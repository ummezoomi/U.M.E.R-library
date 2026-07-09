# Umer_Library/modules/spatial_ml.py

import numpy as np
import pycuda.driver as cuda
from sklearn.metrics import accuracy_score
from Umer_Library.core.memory import UMER_Context

# =====================================================================
# THE DYNAMIC PAYLOAD: Abstract Spatial Intelligence
# Class 0 mass = -1.0, class 1 mass = +1.0 – decision by sign.
# =====================================================================
DYNAMIC_CPP = """
__global__ void kernel_infer_cascade(
    float* test_features, float* out_momentum, float* out_m0, float* out_m1, 
    int* primes, float* tree_weights, int* feature_map, 
    float* A_features, float* A_mass, int* H, int* O, 
    int n_train, int n_test, float cell_size, int num_buckets, float search_radius, float gauss_var
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_test) return;

    float momentum = 0.0f; 

    for (int t = 0; t < NUM_TREES; t++) {
        int hash = 0;
        for (int d = 0; d < DIMS_PER_TREE; d++)
            hash ^= ((int)(test_features[feature_map[t * DIMS_PER_TREE + d] * n_test + idx] / cell_size) * primes[d]);
        hash = hash % num_buckets;
        if (hash < 0) hash += num_buckets;

        int start = O[t * num_buckets + hash] - H[t * num_buckets + hash];
        int count = H[t * num_buckets + hash];
        float m0 = 0.0f, m1 = 0.0f;

        for (int i = 0; i < count; i++) {
            float d_l1 = 0.0f;
            for (int d = 0; d < DIMS_PER_TREE; d++)
                d_l1 += fabsf(A_features[t * DIMS_PER_TREE * n_train + d * n_train + (start + i)]
                               - test_features[feature_map[t * DIMS_PER_TREE + d] * n_test + idx]);
            if (d_l1 <= search_radius) {
                float e = expf(-(d_l1 * d_l1) / gauss_var); 
                float mass_val = A_mass[t * n_train + start + i];
                // mass < 0 → class 0   |   mass > 0 → class 1
                if (mass_val < 0.0f) m0 += e;
                else                  m1 += e;
            }
        }

        out_m0[idx * NUM_TREES + t] = m0;
        out_m1[idx * NUM_TREES + t] = m1;

        if ((m0 + m1) > 0.0f)
            momentum += ((m1 - m0) / (m1 + m0)) * tree_weights[t];
    }
    out_momentum[idx] = momentum;
}
"""


def run_abstract_engine(X_train, y_train, X_test, y_test, feature_names=None, max_trees=None):
    TOTAL_DIMS = X_train.shape[1]
    NUM_TRAIN = len(X_train)
    NUM_TEST = len(X_test)

    # Engine Parameters (kept identical to the proven 97.37% setup)
    CELL_SIZE = 0.20
    SEARCH_RADIUS = 0.09
    GAUSS_VAR = 0.10
    DECAY_RATE = 0.05

    print(f"\n[U.M.E.R] Booting Abstract Spatial Intelligence ({TOTAL_DIMS}D)...")

    # Flatten and upload training data
    X_tr_flat = np.ascontiguousarray(X_train.T).astype(np.float32).flatten()
    # Convert integer labels to signed float mass: -1 for class 0, +1 for class 1
    y_mass = np.where(y_train == 0, -1.0, 1.0).astype(np.float32)
    y_mass_flat = np.ascontiguousarray(y_mass)  # already flat (1D)

    d_xt = cuda.mem_alloc(X_tr_flat.nbytes)
    d_mass = cuda.mem_alloc(y_mass_flat.nbytes)
    cuda.memcpy_htod(d_xt, X_tr_flat)
    cuda.memcpy_htod(d_mass, y_mass_flat)

    # 1. INSTANTIATE THE ENGINE
    engine = UMER_Context(n_particles=NUM_TRAIN, total_dims=TOTAL_DIMS, max_trees=max_trees)

    # 2. BUILD TOPOLOGY (now passes float mass instead of raw labels)
    engine.optimize_topology(d_xt, d_mass, cell_size=CELL_SIZE, repulsion_decay=DECAY_RATE)
    engine.build_hash_grid(d_xt, d_mass, cell_size=CELL_SIZE)

    # 3. JIT INJECT the corrected dynamic kernel
    engine.inject_logic(DYNAMIC_CPP, "kernel_infer_cascade")

    # 4. EXECUTE INFERENCE
    X_te_flat = np.ascontiguousarray(X_test.T).astype(np.float32).flatten()
    d_x_te = cuda.mem_alloc(X_te_flat.nbytes)
    cuda.memcpy_htod(d_x_te, X_te_flat)

    d_m = cuda.mem_alloc(NUM_TEST * 4)
    d_m0 = cuda.mem_alloc(NUM_TEST * engine.NUM_TREES * 4)
    d_m1 = cuda.mem_alloc(NUM_TEST * engine.NUM_TREES * 4)

    engine.k_dynamic(
        d_x_te, d_m, d_m0, d_m1,
        engine.d_pr, engine.d_tw, engine.d_fm,
        engine.d_Af, engine.d_Am, engine.d_H, engine.d_O,
        np.int32(NUM_TRAIN), np.int32(NUM_TEST),
        np.float32(CELL_SIZE), np.int32(engine.HASH_BUCKETS),
        np.float32(SEARCH_RADIUS), np.float32(GAUSS_VAR),
        block=(256, 1, 1), grid=(int((NUM_TEST + 255) // 256), 1)
    )
    cuda.Context.synchronize()

    # Retrieve Results
    m_h = np.zeros(NUM_TEST, dtype=np.float32)
    cuda.memcpy_dtoh(m_h, d_m)
    preds = (m_h > 0.0).astype(np.int32)
    acc = accuracy_score(y_test, preds) * 100

    print(f"\n[RESULTS] Classification Accuracy: {acc:.2f}%")

    # 5. SAFE SHUTDOWN
    engine.cleanup()
    d_xt.free();
    d_mass.free();
    d_x_te.free();
    d_m.free();
    d_m0.free();
    d_m1.free()

    return m_h, preds