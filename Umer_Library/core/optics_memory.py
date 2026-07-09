import pycuda.driver as cuda
from pycuda.compiler import SourceModule
import numpy as np

# =====================================================================
# IMMUTABLE OPTICS CORE: Wavefront Compaction & Teleportation
# =====================================================================
CUDA_OPTICS_CORE = """
#define CELL_W       1.0f
#define MACRO_W      8.0f   
#define RAY_STEPS    200

__device__ __forceinline__ int hash3d(float x, float y, float z, int mask) {
    int cx = (int)(x / CELL_W);
    int cy = (int)(y / CELL_W);
    int cz = (int)(z / CELL_W);
    return (cx * 73856093 ^ cy * 19349663 ^ cz * 83492791) & mask;
}

__device__ __forceinline__ int macro_hash3d(float x, float y, float z, int mask) {
    int cx = (int)(x / MACRO_W);
    int cy = (int)(y / MACRO_W);
    int cz = (int)(z / MACRO_W);
    return (cx * 73856093 ^ cy * 19349663 ^ cz * 83492791) & mask;
}

// ---------------------------------------------------------
// 1. MACRO-TOPOLOGY BUILDER (The 'Teleportation' Map)
// ---------------------------------------------------------
__global__ void build_macro_grid(
    const float4* __restrict__ pos4, 
    unsigned char* __restrict__ macro_grid, 
    int n, int macro_mask)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    
    float4 p = pos4[idx];
    int mh = macro_hash3d(p.x, p.y, p.z, macro_mask);
    macro_grid[mh] = 1; 
}

// ---------------------------------------------------------
// 2. PHASE 1: 3D DDA & STREAM COMPACTION
// ---------------------------------------------------------
__global__ void pass1_macro_dda_compaction(
    const float4* __restrict__ ray_origins,
    const float4* __restrict__ ray_dirs,
    const unsigned char* __restrict__ macro_grid,
    int* __restrict__ active_ray_count,
    float4* __restrict__ queue_pos,
    float4* __restrict__ queue_dir,
    int* __restrict__ queue_id,
    int num_rays, int macro_mask)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_rays) return;
    
    float3 pos = make_float3(ray_origins[idx].x, ray_origins[idx].y, ray_origins[idx].z);
    float3 dir = make_float3(ray_dirs[idx].x, ray_dirs[idx].y, ray_dirs[idx].z);
    
    int X = floorf(pos.x / MACRO_W);
    int Y = floorf(pos.y / MACRO_W);
    int Z = floorf(pos.z / MACRO_W);

    int stepX = (dir.x > 0.0f) ? 1 : -1;
    int stepY = (dir.y > 0.0f) ? 1 : -1;
    int stepZ = (dir.z > 0.0f) ? 1 : -1;

    float tDeltaX = (dir.x == 0.0f) ? 1e30f : fabsf(MACRO_W / dir.x);
    float tDeltaY = (dir.y == 0.0f) ? 1e30f : fabsf(MACRO_W / dir.y);
    float tDeltaZ = (dir.z == 0.0f) ? 1e30f : fabsf(MACRO_W / dir.z);

    float tMaxX = (dir.x == 0.0f) ? 1e30f : ((stepX > 0) ? ((X + 1) * MACRO_W - pos.x) : (pos.x - X * MACRO_W)) / fabsf(dir.x);
    float tMaxY = (dir.y == 0.0f) ? 1e30f : ((stepY > 0) ? ((Y + 1) * MACRO_W - pos.y) : (pos.y - Y * MACRO_W)) / fabsf(dir.y);
    float tMaxZ = (dir.z == 0.0f) ? 1e30f : ((stepZ > 0) ? ((Z + 1) * MACRO_W - pos.z) : (pos.z - Z * MACRO_W)) / fabsf(dir.z);

    int steps = 0;
    int max_macro_steps = RAY_STEPS / 8;

    while (steps < max_macro_steps) {
        int mh = macro_hash3d(X * MACRO_W, Y * MACRO_W, Z * MACRO_W, macro_mask);
        
        if (macro_grid[mh] == 1) {
            // MATTER DETECTED: Stream Scatter / Warp Compaction
            int slot = atomicAdd(active_ray_count, 1);
            
            float dist = fminf(tMaxX, fminf(tMaxY, tMaxZ));
            float hit_x = pos.x + dir.x * dist;
            float hit_y = pos.y + dir.y * dist;
            float hit_z = pos.z + dir.z * dist;
            
            queue_pos[slot] = make_float4(hit_x, hit_y, hit_z, 0.0f);
            queue_dir[slot] = make_float4(dir.x, dir.y, dir.z, 0.0f);
            queue_id[slot]  = idx;
            return; 
        }

        if (tMaxX < tMaxY) {
            if (tMaxX < tMaxZ) { X += stepX; tMaxX += tDeltaX; }
            else               { Z += stepZ; tMaxZ += tDeltaZ; }
        } else {
            if (tMaxY < tMaxZ) { Y += stepY; tMaxY += tDeltaY; }
            else               { Z += stepZ; tMaxZ += tDeltaZ; }
        }
        steps++;
    }
}
"""

class UMER_Optics_Context:
    def __init__(self, max_rays, macro_buckets=262144, block_size=256, macro_width=8.0):
        self.MACRO_W = float(macro_width)
        """Initializes the GPU Memory for Raytracing and Optical Compaction."""
        self.MAX_RAYS = max_rays
        self.MACRO_BUCKETS = macro_buckets
        self.MACRO_MASK = np.int32(self.MACRO_BUCKETS - 1)
        self.BS = block_size
        
        self.RAY_BLOCK = (self.BS, 1, 1)
        self.RAY_GRID = ((self.MAX_RAYS + self.BS - 1) // self.BS, 1)

        print(f"[UMER OPTICS CORE] Initializing Optical Context: {self.MAX_RAYS:,} max rays, {self.MACRO_BUCKETS:,} macro-buckets.")

        # 1. Compile Immutable Optics Core
        core_code = CUDA_OPTICS_CORE.replace("#define MACRO_W      8.0f", f"#define MACRO_W      {self.MACRO_W}f")
        self.core_module = SourceModule(core_code, options=["-O3", "--use_fast_math"])
        self.k_build_macro = self.core_module.get_function("build_macro_grid")
        self.k_pass1 = self.core_module.get_function("pass1_macro_dda_compaction")

        # 2. Global Memory Allocations for Wavefront Queues
        # Using 8-bit ints (unsigned chars) for the macro grid to save massive VRAM
        self.d_macro_grid   = cuda.mem_alloc(self.MACRO_BUCKETS) 
        self.d_active_count = cuda.mem_alloc(4)
        
        # Stream Compaction Buffers
        self.d_queue_pos = cuda.mem_alloc(self.MAX_RAYS * 16)
        self.d_queue_dir = cuda.mem_alloc(self.MAX_RAYS * 16)
        self.d_queue_id  = cuda.mem_alloc(self.MAX_RAYS * 4)

        # Dynamic specific pointers
        self.dynamic_shader_module = None
        self.k_pass2_shader = None

    def update_optical_topology(self, d_pos, n_particles):
        """Builds the low-resolution Macro-Grid used for Ray Teleportation."""
        cuda.memset_d8(self.d_macro_grid, 0, self.MACRO_BUCKETS)
        pgrid = ((n_particles + self.BS - 1) // self.BS, 1)
        self.k_build_macro(d_pos, self.d_macro_grid, np.int32(n_particles), self.MACRO_MASK, block=self.RAY_BLOCK, grid=pgrid)

    def inject_shader(self, shader_c_string, function_name):
        """JIT Compiles LLM-generated lighting, reflection, and BRDF math for Phase 2."""
        headers = """#include <math.h>
    #define CELL_W       1.0f
    __device__ __forceinline__ int hash3d(float x, float y, float z, int mask) {
        int cx = (int)(x / CELL_W);
        int cy = (int)(y / CELL_W);
        int cz = (int)(z / CELL_W);
        return (cx * 73856093 ^ cy * 19349663 ^ cz * 83492791) & mask;
    }
    """
        full_code = headers + shader_c_string
        self.dynamic_shader_module = SourceModule(full_code, options=["-O3", "--use_fast_math"])
        self.k_pass2_shader = self.dynamic_shader_module.get_function(function_name)
        print(f"[UMER JIT] Dynamic Optics Shader '{function_name}' locked into Phase 2.")

    def trace_rays(self, d_ray_origins, d_ray_dirs, num_active_rays, d_image_buffer, d_histogram):
        """Executes the Two-Pass Wavefront Compaction and executes the injected shader.
           d_histogram must be a valid GPU pointer to the macro grid (float3* positions)."""
        if self.k_pass2_shader is None:
            raise RuntimeError("[UMER ERROR] No Phase 2 Shader injected. Call inject_shader() first.")

        r_grid = ((num_active_rays + self.BS - 1) // self.BS, 1)

        # Reset the atomic queue counter
        cuda.memset_d32(self.d_active_count, 0, 1)

        # PHASE 1: Traverse empty space and compact rays that hit the macro-grid
        self.k_pass1(d_ray_origins, d_ray_dirs, self.d_macro_grid, self.d_active_count,
                     self.d_queue_pos, self.d_queue_dir, self.d_queue_id,
                     np.int32(num_active_rays), self.MACRO_MASK,
                     block=self.RAY_BLOCK, grid=r_grid)

        # PHASE 2: Execute User-Injected Shader – always pass the valid d_histogram
        self.k_pass2_shader(self.d_active_count, self.d_queue_pos, self.d_queue_dir, self.d_queue_id,
                            d_image_buffer, d_histogram,
                            np.int32(num_active_rays),
                            block=self.RAY_BLOCK, grid=r_grid)

    def cleanup(self):
        """Safely returns VRAM allocations."""
        self.d_macro_grid.free()
        self.d_active_count.free()
        self.d_queue_pos.free()
        self.d_queue_dir.free()
        self.d_queue_id.free()