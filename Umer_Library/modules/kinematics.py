# Umer_Library/modules/kinematics.py

import numpy as np
import pycuda.driver as cuda
import time
from Umer_Library.core.memory import UMER_Context

# =====================================================================
# 1. THE DYNAMIC PAYLOAD 
# (In the future, your LLM generates ONLY this string based on your prompt)
# =====================================================================
DYNAMIC_CPP = """
__global__ void kernel_infer_vector(
    float* current_states, float* out_vector, int* primes, float* tree_weights, int* feature_map, 
    float* A_features, float* A_mass, int* H, int* O, 
    int n_env, int n_test, float cell_size, int num_buckets, float gauss_var
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_test) return;
    
    // Clear the vector
    for(int d=0; d<TOTAL_DIMS; d++) out_vector[idx * TOTAL_DIMS + d] = 0.0f;
    
    for (int t = 0; t < NUM_TREES; t++) {
        int hash = 0;
        for (int d = 0; d < DIMS_PER_TREE; d++) hash ^= ((int)(current_states[feature_map[t * DIMS_PER_TREE + d] * n_test + idx] / cell_size) * primes[d]);
        hash = hash % num_buckets; if (hash < 0) hash += num_buckets;
        
        int start = O[t * num_buckets + hash] - H[t * num_buckets + hash], count = H[t * num_buckets + hash];
        for (int i = 0; i < count; i++) {
            float d_l1 = 0.0f;
            for (int d = 0; d < DIMS_PER_TREE; d++) {
                d_l1 += fabsf(A_features[t * DIMS_PER_TREE * n_env + d * n_env + (start + i)] - current_states[feature_map[t * DIMS_PER_TREE + d] * n_test + idx]);
            }
            
            float energy = expf(-(d_l1 * d_l1) / gauss_var); 
            float physical_mass = A_mass[t * n_env + start + i];
            float pull_magnitude = (physical_mass * energy) * tree_weights[t];
            
            for (int d = 0; d < DIMS_PER_TREE; d++) {
                int global_dim = feature_map[t * DIMS_PER_TREE + d];
                float direction = current_states[global_dim * n_test + idx] - A_features[t * DIMS_PER_TREE * n_env + d * n_env + start + i];
                atomicAdd(&out_vector[idx * TOTAL_DIMS + global_dim], direction * pull_magnitude);
            }
        }
    }
}
"""

def run_kinematic_driver():
    TOTAL_DIMS = 30
    NUM_OBSTACLES = 20000
    CELL_SIZE = 0.20
    GAUSS_VAR = 0.05
    LEARNING_RATE = 0.05

    print("\n[U.M.E.R] Booting 30D Kinematic Driver...")

    # --- Setup Environment Data ---
    obstacles = np.random.rand(NUM_OBSTACLES, TOTAL_DIMS).astype(np.float32)
    y_mass = np.full(NUM_OBSTACLES, -1.0, dtype=np.float32)
    target = np.full((1, TOTAL_DIMS), 0.8, dtype=np.float32)
    X_robot = np.full((1, TOTAL_DIMS), 0.2, dtype=np.float32)

    obstacles_flat = np.ascontiguousarray(obstacles.T).flatten()

    # Upload to GPU
    d_xe = cuda.mem_alloc(obstacles_flat.nbytes)
    d_me = cuda.mem_alloc(y_mass.nbytes)
    cuda.memcpy_htod(d_xe, obstacles_flat)
    cuda.memcpy_htod(d_me, y_mass)

    # =====================================================================
    # 2. INSTANTIATE THE ENGINE & BUILD IMMUTABLE TOPOLOGY
    # =====================================================================
    engine = UMER_Context(n_particles=NUM_OBSTACLES, total_dims=TOTAL_DIMS)
    
    engine.optimize_topology(d_xe, d_me, cell_size=CELL_SIZE, repulsion_decay=0.50)
    engine.build_hash_grid(d_xe, d_me, cell_size=CELL_SIZE)

    # =====================================================================
    # 3. JIT INJECT THE DYNAMIC LOGIC
    # =====================================================================
    engine.inject_logic(DYNAMIC_CPP, "kernel_infer_vector")

    # =====================================================================
    # 4. EXECUTION LOOP
    # =====================================================================
    d_out_vector = cuda.mem_alloc(TOTAL_DIMS * 4) 
    print("\n--- ENGAGING U.M.E.R. GRAVITY DRIVE ---")
    
    for step in range(120):
        X_rob_flat = np.ascontiguousarray(X_robot.T).flatten()
        d_x_rob = cuda.mem_alloc(X_rob_flat.nbytes)
        cuda.memcpy_htod(d_x_rob, X_rob_flat)
        cuda.memset_d32(d_out_vector, 0, TOTAL_DIMS)

        # Notice how we pass the engine's internal pointers (engine.d_pr, engine.d_fm, etc.)
        engine.k_dynamic(
            d_x_rob, d_out_vector, engine.d_pr, engine.d_tw, engine.d_fm, 
            engine.d_Af, engine.d_Am, engine.d_H, engine.d_O, 
            np.int32(NUM_OBSTACLES), np.int32(1), np.float32(CELL_SIZE), 
            np.int32(engine.HASH_BUCKETS), np.float32(GAUSS_VAR), 
            block=(1,1,1), grid=(1,1)
        )
        
        repulsion_vector = np.zeros(TOTAL_DIMS, dtype=np.float32)
        cuda.memcpy_dtoh(repulsion_vector, d_out_vector)
        attraction_vector = target[0] - X_robot[0]
        
        norm_rep = np.linalg.norm(repulsion_vector)
        if norm_rep > 0: repulsion_vector /= norm_rep
            
        norm_att = np.linalg.norm(attraction_vector)
        if norm_att > 0: attraction_vector /= norm_att
            
        final_vector = (attraction_vector * 1.5) + (repulsion_vector * 1.0)
        X_robot[0] += final_vector * LEARNING_RATE
        
        dist = np.linalg.norm(X_robot[0] - target[0])
        
        if step % 15 == 0:
            print(f"Step {step:03d} | 30D Distance: {dist:.4f} | Dodging: {norm_rep > 0.01}")
            
        d_x_rob.free() # No dead contexts here!

        if dist < 0.15:
            print(f"\n[SUCCESS] Target acquired at Step {step}. Pathfinding Complete.")
            break

    # =====================================================================
    # 5. SAFE SHUTDOWN
    # =====================================================================
    engine.cleanup()
    d_xe.free()
    d_me.free()
    d_out_vector.free()

if __name__ == "__main__":
    run_kinematic_driver()