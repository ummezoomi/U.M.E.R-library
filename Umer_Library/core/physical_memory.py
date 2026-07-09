# Umer_Library/core/physical_memory.py

import pycuda.driver as cuda
from pycuda.compiler import SourceModule
import numpy as np
from .primitives_3d import CUDA_PHYSICS_CORE

class UMER_Physical_Context:
    def __init__(self, n_particles, buckets=524288, block_size=256):
        self.N = n_particles
        self.BUCKETS = buckets
        self.MASK = np.int32(self.BUCKETS - 1)
        self.BS = block_size
        
        self.L1_BLKS = (self.BUCKETS + self.BS - 1) // self.BS
        self.L2_BLKS = (self.L1_BLKS  + self.BS - 1) // self.BS
        self.SHM = self.BS * 4
        
        self.BLOCK = (self.BS, 1, 1)
        self.PGRID = ((self.N + self.BS - 1) // self.BS, 1)

        print(f"[UMER 3D CORE] Initializing Physical Environment: {self.N:,} particles, {self.BUCKETS:,} buckets.")

        # Compile Immutable Core
        self.core_module = SourceModule(CUDA_PHYSICS_CORE, options=["-O3", "--use_fast_math"])
        self.k_hist_full  = self.core_module.get_function("build_histogram_full")
        self.k_hist_delta = self.core_module.get_function("update_histogram_delta")
        self.k_scan       = self.core_module.get_function("scan_blocks")
        self.k_add        = self.core_module.get_function("add_offsets")
        self.k_scat_full  = self.core_module.get_function("scatter_full")
        self.k_scat_movers = self.core_module.get_function("scatter_movers")

        # Global Memory Allocations (One-time setup)
        self.d_hist      = cuda.mem_alloc(self.BUCKETS * 4)
        self.d_offsets   = cuda.mem_alloc(self.BUCKETS * 4)
        self.d_local     = cuda.mem_alloc(self.BUCKETS * 4)
        self.d_prev_h    = cuda.mem_alloc(self.N * 4)
        self.d_is_mover  = cuda.mem_alloc(self.N * 4)
        self.d_mover_cnt = cuda.mem_alloc(4)
        self.d_L1_sums   = cuda.mem_alloc(self.L1_BLKS * 4)
        self.d_L2_sums   = cuda.mem_alloc(self.L2_BLKS * 4)
        self.d_L2_out    = cuda.mem_alloc(self.L1_BLKS * 4)
        self.d_L3_out    = cuda.mem_alloc(self.L2_BLKS * 4)
        self.d_dummy     = cuda.mem_alloc(4)

        self.k_dynamic_motion = None

    def _gpu_prefix_sum(self, d_in, d_out):
        self.k_scan(d_in, d_out, self.d_L1_sums, np.int32(self.BUCKETS), block=self.BLOCK, grid=(self.L1_BLKS, 1), shared=self.SHM)
        self.k_scan(self.d_L1_sums, self.d_L2_out, self.d_L2_sums, np.int32(self.L1_BLKS), block=self.BLOCK, grid=(self.L2_BLKS, 1), shared=self.SHM)
        self.k_scan(self.d_L2_sums, self.d_L3_out, self.d_dummy, np.int32(self.L2_BLKS), block=self.BLOCK, grid=(1, 1), shared=self.SHM)
        self.k_add(self.d_L2_out, self.d_L3_out, np.int32(self.L1_BLKS), block=self.BLOCK, grid=(self.L2_BLKS, 1))
        self.k_add(d_out, self.d_L2_out, np.int32(self.BUCKETS), block=self.BLOCK, grid=(self.L1_BLKS, 1))

    def cold_start(self, d_pos, d_vel, d_pos_out, d_vel_out):
        """Full spatial hash build for frame 0."""
        cuda.memset_d32(self.d_hist, 0, self.BUCKETS)
        self.k_hist_full(d_pos, self.d_hist, self.d_prev_h, np.int32(self.N), self.MASK, block=self.BLOCK, grid=self.PGRID)
        self._gpu_prefix_sum(self.d_hist, self.d_offsets)
        cuda.memset_d32(self.d_local, 0, self.BUCKETS)
        self.k_scat_full(d_pos, d_vel, self.d_offsets, self.d_local, d_pos_out, d_vel_out, np.int32(self.N), self.MASK, block=self.BLOCK, grid=self.PGRID)

    def temporal_update(self, d_pos, d_vel, d_pos_out, d_vel_out):
        """Ultra-fast delta update tracking only 'movers'."""
        cuda.memset_d32(self.d_mover_cnt, 0, 1)
        self.k_hist_delta(d_pos, self.d_hist, self.d_prev_h, self.d_is_mover, self.d_mover_cnt, np.int32(self.N), self.MASK, block=self.BLOCK, grid=self.PGRID)
        self._gpu_prefix_sum(self.d_hist, self.d_offsets)
        cuda.memset_d32(self.d_local, 0, self.BUCKETS)
        self.k_scat_movers(d_pos, d_vel, self.d_is_mover, self.d_offsets, self.d_local, d_pos_out, d_vel_out, np.int32(self.N), self.MASK, block=self.BLOCK, grid=self.PGRID)

    def inject_physics(self, motion_c_string, function_name):
        """JIT Compiles LLM-generated physical behavior (gravity, boids, collisions)."""
        full_code = "#define DOMAIN 1000.0f\n" + motion_c_string
        self.dynamic_module = SourceModule(full_code, options=["-O3", "--use_fast_math"])
        self.k_dynamic_motion = self.dynamic_module.get_function(function_name)
        print(f"[UMER JIT] Dynamic Physics '{function_name}' locked in.")

    def cleanup(self):
        self.d_hist.free(); self.d_offsets.free(); self.d_local.free(); self.d_prev_h.free()
        self.d_is_mover.free(); self.d_mover_cnt.free(); self.d_L1_sums.free()
        self.d_L2_sums.free(); self.d_L2_out.free(); self.d_L3_out.free(); self.d_dummy.free()