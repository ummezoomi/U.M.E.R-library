# Umer_Library/core/neural_memory.py

import numpy as np
import pycuda.driver as cuda
from pycuda.compiler import SourceModule

# =====================================================================
# IMMUTABLE NEURAL CORE: Amortized Hash Updates & Early Termination
# =====================================================================
CUDA_NEURAL_CORE = """
#define CELL_W {cell_width}f

__device__ __forceinline__ int hash3d(float x, float y, float z, int mask) {
    int cx = (int)(x / CELL_W);
    int cy = (int)(y / CELL_W);
    int cz = (int)(z / CELL_W);
    return (cx * 73856093 ^ cy * 19349663 ^ cz * 83492791) & mask;
}

__global__ void build_neural_topology(
    const float4* __restrict__ pos4, 
    int* __restrict__ histogram, 
    int n, int mask)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    float4 p = pos4[idx];
    atomicAdd(&histogram[hash3d(p.x, p.y, p.z, mask)], 1);
}

__global__ void clear_dynamic_topology(
    const float4* __restrict__ pos4, 
    int* __restrict__ histogram, 
    int n, int mask)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    float4 p = pos4[idx];
    atomicSub(&histogram[hash3d(p.x, p.y, p.z, mask)], 1);
}
"""

class UMER_Neural_Context:
    def __init__(self, buckets=67108864, block_size=256, cell_width=1.0):
        self.BUCKETS = buckets
        self.MASK = np.int32(self.BUCKETS - 1)
        self.BS = block_size
        self.CELL_W = float(cell_width)

        print(f"[UMER NEURAL CORE] Initializing Topology: {self.BUCKETS:,} buckets, Cell Width: {self.CELL_W}")

        core_code = CUDA_NEURAL_CORE.replace("{cell_width}", str(self.CELL_W))
        self.core_module = SourceModule(core_code, options=["-O3", "--use_fast_math"])

        self.k_build = self.core_module.get_function("build_neural_topology")
        self.k_clear = self.core_module.get_function("clear_dynamic_topology")

        self.d_hist = cuda.mem_alloc(self.BUCKETS * 4)

        self.dynamic_eval_module = None
        self.k_eval_shader = None

    def initialize_static_geometry(self, d_pos_static, n_static):
        """Hashes the static universe (the 90%) into the topology once."""
        cuda.memset_d32(self.d_hist, 0, self.BUCKETS)
        grid = ((n_static + self.BS - 1) // self.BS, 1)
        self.k_build(d_pos_static, self.d_hist, np.int32(n_static), self.MASK, block=(self.BS, 1, 1), grid=grid)

    def amortized_dynamic_update(self, d_pos_dynamic_old, d_pos_dynamic_new, n_dynamic):
        """The core architectural speedup: Only update the moving 10%."""
        grid = ((n_dynamic + self.BS - 1) // self.BS, 1)
        self.k_clear(d_pos_dynamic_old, self.d_hist, np.int32(n_dynamic), self.MASK, block=(self.BS, 1, 1), grid=grid)
        self.k_build(d_pos_dynamic_new, self.d_hist, np.int32(n_dynamic), self.MASK, block=(self.BS, 1, 1), grid=grid)

    def inject_neural_evaluator(self, shader_string, function_name):
        """
        JIT Compiles the LLM-generated splat logic.
        The shader string may begin with #define statements (e.g. for step counts).
        We prepend <math.h> and the AABB helper automatically.
        """
        headers = f"""#include <math.h>
#define CELL_W {self.CELL_W}f

__device__ __forceinline__ int hash3d(float x, float y, float z, int mask) {{
    int cx = (int)(x / CELL_W);
    int cy = (int)(y / CELL_W);
    int cz = (int)(z / CELL_W);
    return (cx * 73856093 ^ cy * 19349663 ^ cz * 83492791) & mask;
}}

__device__ __forceinline__ float intersect_aabb(float3 ro, float3 rd, float3 boxMin, float3 boxMax) {{
    float tx1 = (boxMin.x - ro.x) / rd.x;
    float tx2 = (boxMax.x - ro.x) / rd.x;
    float tmin = fminf(tx1, tx2);
    float tmax = fmaxf(tx1, tx2);

    float ty1 = (boxMin.y - ro.y) / rd.y;
    float ty2 = (boxMax.y - ro.y) / rd.y;
    tmin = fmaxf(tmin, fminf(ty1, ty2));
    tmax = fminf(tmax, fmaxf(ty1, ty2));

    float tz1 = (boxMin.z - ro.z) / rd.z;
    float tz2 = (boxMax.z - ro.z) / rd.z;
    tmin = fmaxf(tmin, fminf(tz1, tz2));
    tmax = fminf(tmax, fmaxf(tz1, tz2));

    if (tmax >= tmin && tmax > 0.0f) {{ return fmaxf(0.0f, tmin); }}
    return -1.0f; 
}}
"""
        full_code = headers + shader_string
        self.dynamic_eval_module = SourceModule(full_code, options=["-O3", "--use_fast_math"])
        self.k_eval_shader = self.dynamic_eval_module.get_function(function_name)
        print(f"[UMER JIT] Neural Evaluator '{function_name}' locked into the pipeline.")

    def render_volume(self, d_ro, d_rd, d_bmin, d_bmax, d_colors, num_rays, d_out_buffer, d_evals_taken=None):
        """Executes the Ray Marching & Early Alpha Termination."""
        if self.k_eval_shader is None:
            raise RuntimeError("[UMER ERROR] Inject evaluator first before rendering.")

        grid = ((num_rays + self.BS - 1) // self.BS, 1)
        d_evals_ptr = d_evals_taken if d_evals_taken else np.intp(0)

        self.k_eval_shader(
            d_ro, d_rd, d_bmin, d_bmax, self.d_hist, d_colors,
            d_out_buffer, d_evals_ptr, np.int32(num_rays), self.MASK,
            block=(self.BS, 1, 1), grid=grid
        )

    def cleanup(self):
        self.d_hist.free()
        print("[UMER NEURAL CORE] VRAM Buffers safely released.")