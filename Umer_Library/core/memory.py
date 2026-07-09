# umer_engine/core/memory.py

import pycuda.autoinit
import pycuda.driver as cuda
from pycuda.compiler import SourceModule
import numpy as np
import itertools
from .primitives import CUDA_CASCADE, CUDA_PGP, PRIME_POOL

class UMER_Context:
    def __init__(self, n_particles, total_dims, hash_buckets=4096, dims_per_tree=4, stride=2, max_trees=None):
        """Initializes the GPU Memory and Compiles the Immutable Core."""
        self.N = n_particles
        self.TOTAL_DIMS = total_dims
        self.DIMS_PER_TREE = dims_per_tree
        self.STRIDE = stride
        self.HASH_BUCKETS = hash_buckets
        self.NUM_TREES = ((self.TOTAL_DIMS - self.DIMS_PER_TREE) // self.STRIDE) + 1
        if max_trees is not None:
            self.NUM_TREES = min(self.NUM_TREES, max_trees)
        
        print(f"[UMER CORE] Initializing Environment: {self.N} agents, {self.TOTAL_DIMS}D Space, {self.NUM_TREES} Grids.")

        # 1. Compile Immutable Core
        core_code = CUDA_CASCADE.replace("{num_trees}", str(self.NUM_TREES))
        core_code = core_code.replace("{dims_per_tree}", str(self.DIMS_PER_TREE))
        self.core_module = SourceModule(core_code)
        
        self.k_h = self.core_module.get_function("kernel_histogram")
        self.k_p = self.core_module.get_function("kernel_prefix_sum")
        self.k_s = self.core_module.get_function("kernel_scatter")

        # 2. Compile Optimization Core (PGP)
        self.pgp_module = SourceModule(CUDA_PGP)
        self.k_pgp = self.pgp_module.get_function("kernel_pgp_histogram")

        # 3. Global Memory Allocations (Eliminates leaks/re-allocations)
        self._allocate_memory()
        
        # 4. State tracking
        self.dynamic_module = None
        self.k_dynamic = None

    def _allocate_memory(self):
        """Locks in all necessary VRAM for the topology."""
        self.d_pr = cuda.mem_alloc(16)
        cuda.memcpy_htod(self.d_pr, np.array(PRIME_POOL[:4], dtype=np.int32))
        
        self.d_fm = cuda.mem_alloc(self.NUM_TREES * self.DIMS_PER_TREE * 4)
        self.d_tw = cuda.mem_alloc(self.NUM_TREES * 4)
        
        # Massive topological buffers
        self.d_Af = cuda.mem_alloc(self.NUM_TREES * self.DIMS_PER_TREE * self.N * 4)
        self.d_Am = cuda.mem_alloc(self.NUM_TREES * self.N * 4)
        
        self.d_H = cuda.mem_alloc(self.NUM_TREES * self.HASH_BUCKETS * 4)
        self.d_O = cuda.mem_alloc(self.NUM_TREES * self.HASH_BUCKETS * 4)
        self.d_C = cuda.mem_alloc(self.NUM_TREES * self.HASH_BUCKETS * 4)

    def optimize_topology(self, d_features, d_mass, cell_size, repulsion_decay):
        """Runs the PGP sweep to map the dimensions and calculate tree weights."""
        pairs = list(itertools.combinations(range(self.TOTAL_DIMS), 2))
        num_pairs = len(pairs)
        px_d = np.array([p[0] for p in pairs], dtype=np.int32)
        py_d = np.array([p[1] for p in pairs], dtype=np.int32)
        
        d_px = cuda.mem_alloc(px_d.nbytes)
        d_py = cuda.mem_alloc(py_d.nbytes)
        cuda.memcpy_htod(d_px, px_d)
        cuda.memcpy_htod(d_py, py_d)
        
        d_Ho = cuda.mem_alloc(num_pairs * self.HASH_BUCKETS * 4)
        cuda.memset_d32(d_Ho, 0, num_pairs * self.HASH_BUCKETS)
        
        # Run PGP
        self.k_pgp(d_features, d_mass, d_px, d_py, d_Ho, np.int32(self.N), 
                   np.float32(cell_size), np.int32(self.HASH_BUCKETS), np.int32(num_pairs), 
                   block=(256,1,1), grid=(int((self.N+255)//256),1))
        
        ho = np.zeros(num_pairs * self.HASH_BUCKETS, dtype=np.float32)
        cuda.memcpy_dtoh(ho, d_Ho)
        ho_m = ho.reshape(num_pairs, self.HASH_BUCKETS)
        syn_scores = np.sum(ho_m**2, axis=1)

        # Build Feature Map
        best_p = np.argmax(syn_scores)
        f_list, used = list(pairs[best_p]), set(pairs[best_p])
        while len(f_list) < self.TOTAL_DIMS:
            best_s, best_f = -1.0, -1
            tail = f_list[-2:]
            for f in range(self.TOTAL_DIMS):
                if f in used: continue
                curr_s = sum(syn_scores[i] for i, p in enumerate(pairs) if (p[0] == f and p[1] in tail) or (p[1] == f and p[0] in tail))
                if curr_s > best_s: best_s, best_f = curr_s, f
            f_list.append(best_f)
            used.add(best_f)

        f_map = np.array([f_list[t*self.STRIDE : t*self.STRIDE+self.DIMS_PER_TREE] for t in range(self.NUM_TREES)], dtype=np.int32).flatten()
        raw_t_weights = np.array([sum(syn_scores[i] for i, p in enumerate(pairs) if p[0] in f_list[t*self.STRIDE:t*self.STRIDE+4] and p[1] in f_list[t*self.STRIDE:t*self.STRIDE+4]) for t in range(self.NUM_TREES)])
        
        raw_t_weights = (raw_t_weights / np.max(raw_t_weights)).astype(np.float32)
        d_weights = (raw_t_weights * np.exp(-repulsion_decay * np.arange(self.NUM_TREES))).astype(np.float32)

        # Upload optimal map and weights
        cuda.memcpy_htod(self.d_fm, f_map)
        cuda.memcpy_htod(self.d_tw, d_weights)
        
        # Free local pointers
        d_px.free(); d_py.free(); d_Ho.free()
        print("[UMER CORE] Topology Optimized and Locked.")

    def build_hash_grid(self, d_features, d_mass, cell_size):
        """Executes the standard 3-step U.M.E.R. topology build."""
        cuda.memset_d32(self.d_H, 0, self.NUM_TREES * self.HASH_BUCKETS)
        cuda.memset_d32(self.d_O, 0, self.NUM_TREES * self.HASH_BUCKETS)
        cuda.memset_d32(self.d_C, 0, self.NUM_TREES * self.HASH_BUCKETS)

        grid_size = int((self.N + 255) // 256)
        
        self.k_h(d_features, self.d_pr, self.d_fm, self.d_H, np.int32(self.N), np.float32(cell_size), np.int32(self.HASH_BUCKETS), block=(256,1,1), grid=(grid_size,1))
        self.k_p(self.d_H, self.d_O, np.int32(self.HASH_BUCKETS), block=(self.NUM_TREES,1,1), grid=(1,1))
        self.k_s(d_features, d_mass, self.d_fm, self.d_Af, self.d_Am, self.d_pr, self.d_H, self.d_O, self.d_C, np.int32(self.N), np.float32(cell_size), np.int32(self.HASH_BUCKETS), block=(256,1,1), grid=(grid_size,1))

    def inject_logic(self, c_string, function_name):
        """JIT Compiles LLM-generated logic and attaches it to the engine."""
        # We inject the engine definitions so the LLM doesn't have to write them
        headers = f"""
        #include <stdint.h>
        #include <math.h>
        #define NUM_TREES {self.NUM_TREES}
        #define DIMS_PER_TREE {self.DIMS_PER_TREE}
        #define TOTAL_DIMS {self.TOTAL_DIMS}
        """
        full_code = headers + c_string
        self.dynamic_module = SourceModule(full_code)
        self.k_dynamic = self.dynamic_module.get_function(function_name)
        print(f"[UMER JIT] Dynamic Logic '{function_name}' compiled successfully.")

    def cleanup(self):
        """Call this when shutting down to cleanly return memory."""
        self.d_pr.free()
        self.d_fm.free()
        self.d_tw.free()
        self.d_Af.free()
        self.d_Am.free()
        self.d_H.free()
        self.d_O.free()
        self.d_C.free()
        print("[UMER CORE] Memory buffers cleared.")