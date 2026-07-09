# Umer_Library/modules/physics.py

import numpy as np
import pycuda.driver as cuda
import time
from Umer_Library.core.physical_memory import UMER_Physical_Context

# =====================================================================
# THE DYNAMIC PAYLOAD: Physical Motion
# If you ask Nethanial for "Gravity", it rewrites this string.
# If you ask for "Boids Flocking", it rewrites this string. 
# =====================================================================
DYNAMIC_MOTION = """
__global__ void apply_motion(float4* __restrict__ pos4, const float4* __restrict__ vel4, float dt, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    float4 p = pos4[idx]; 
    float4 v = vel4[idx];
    
    // Default Linear Motion with Domain Wrapping
    p.x = fmodf(p.x + v.x * dt + DOMAIN, DOMAIN);
    p.y = fmodf(p.y + v.y * dt + DOMAIN, DOMAIN);
    p.z = fmodf(p.z + v.z * dt + DOMAIN, DOMAIN);
    
    pos4[idx] = p;
}
"""

def run_temporal_simulation():
    N_PARTICLES = 4_000_000
    DOMAIN = 1000.0
    DT = 0.1
    ITERS = 100

    print(f"\n[U.M.E.R] Booting Physics Simulation ({N_PARTICLES:,} Agents)...")

    # Setup Host Memory
    pos_host = np.zeros((N_PARTICLES, 4), dtype=np.float32)
    vel_host = np.zeros((N_PARTICLES, 4), dtype=np.float32)
    
    pos_host[:, 0:3] = np.random.rand(N_PARTICLES, 3).astype(np.float32) * DOMAIN
    vel_host[:, 0:3] = (np.random.rand(N_PARTICLES, 3).astype(np.float32) - 0.5) * 5.0 # Fast gas velocity

    # Setup GPU Memory
    d_pos_in  = cuda.mem_alloc(pos_host.nbytes)
    d_vel_in  = cuda.mem_alloc(pos_host.nbytes)
    d_pos_out = cuda.mem_alloc(pos_host.nbytes)
    d_vel_out = cuda.mem_alloc(pos_host.nbytes)

    cuda.memcpy_htod(d_pos_in, pos_host)
    cuda.memcpy_htod(d_vel_in, vel_host)

    # 1. INITIALIZE ENGINE
    engine = UMER_Physical_Context(n_particles=N_PARTICLES)
    
    # 2. INJECT LLM LOGIC
    engine.inject_physics(DYNAMIC_MOTION, "apply_motion")

    # 3. WARMUP (Cold Start Topology)
    engine.cold_start(d_pos_in, d_vel_in, d_pos_out, d_vel_out)
    cuda.Context.synchronize()

    # 4. EXECUTION LOOP
    print("\n--- INITIATING TEMPORAL COHERENCE LOOP ---")
    start = time.perf_counter()
    
    for step in range(ITERS):
        # The LLM injected math moves the particles
        engine.k_dynamic_motion(d_pos_in, d_vel_in, np.float32(DT), np.int32(N_PARTICLES), block=engine.BLOCK, grid=engine.PGRID)
        
        # The Immutable core handles the memory sorting
        engine.temporal_update(d_pos_in, d_vel_in, d_pos_out, d_vel_out)
        
    cuda.Context.synchronize()
    elapsed = time.perf_counter() - start

    ms_per_frame = (elapsed / ITERS) * 1000
    throughput = N_PARTICLES / (ms_per_frame / 1000) / 1e6
    
    print(f"[RESULTS] Average Time: {ms_per_frame:.3f} ms/frame")
    print(f"[RESULTS] Throughput:   {throughput:.1f} MCell/s")

    # 5. SAFE SHUTDOWN
    engine.cleanup()
    d_pos_in.free(); d_vel_in.free(); d_pos_out.free(); d_vel_out.free()

if __name__ == "__main__":
    run_temporal_simulation()