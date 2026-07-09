# Umer_Library/telemetry/sweeps.py

import pycuda.autoinit
import pycuda.driver as cuda
from pycuda.compiler import SourceModule
import numpy as np
import itertools
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score
import plotly.graph_objects as go
import plotly.io as pio

PRIME_POOL = [73856093, 19349663, 83492791, 23948573]

# FIX: Removed the hardcoded '30' dimension limit to allow true N-D scaling
CUDA_PGP = """
#include <stdint.h>
__global__ void kernel_pgp_histogram(
    float* features_in, int* labels_in, int* pairs_x, int* pairs_y,
    int* H0, int* H1, int n_train, float cell_size, int num_buckets, int num_pairs
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_train) return;
    int label = labels_in[idx];

    #pragma unroll 4
    for (int p = 0; p < num_pairs; p++) {
        int dx = pairs_x[p], dy = pairs_y[p];

        int ix = (int)(features_in[dx * n_train + idx] / cell_size);
        int iy = (int)(features_in[dy * n_train + idx] / cell_size);

        int hash = ((ix * 73856093) ^ (iy * 19349663)) % num_buckets;
        if (hash < 0) hash += num_buckets;

        // Dynamic bound check based on the number of pairs provided
        if (p * num_buckets + hash < num_pairs * num_buckets) {
            if (label == 0) atomicAdd(&H0[p * num_buckets + hash], 1);
            else atomicAdd(&H1[p * num_buckets + hash], 1);
        }
    }
}
"""

CUDA_CASCADE = """
#include <stdint.h>
#include <math.h>
#define NUM_TREES {num_trees}
#define DIMS_PER_TREE {dims_per_tree}

__global__ void kernel_histogram(float* features_in, int* primes, int* feature_map, int* H, int n_train, float cell_size, int num_buckets) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_train) return;
    for (int t = 0; t < NUM_TREES; t++) {
        int hash = 0;
        for (int d = 0; d < DIMS_PER_TREE; d++) {
            hash ^= ((int)(features_in[feature_map[t * DIMS_PER_TREE + d] * n_train + idx] / cell_size) * primes[d]);
        }
        hash = hash % num_buckets; if (hash < 0) hash += num_buckets;
        atomicAdd(&H[t * num_buckets + hash], 1);
    }
}

__global__ void kernel_prefix_sum(int* H, int* O, int num_buckets) {
    int t = threadIdx.x; if (t < NUM_TREES) {
        int sum = 0;
        for (int i = 0; i < num_buckets; i++) { sum += H[t * num_buckets + i]; O[t * num_buckets + i] = sum; }
    }
}

__global__ void kernel_scatter(float* features_in, int* labels_in, int* feature_map, float* A_features, int* A_label, int* primes, int* H, int* O, int* C, int n_train, float cell_size, int num_buckets) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_train) return;
    for (int t = 0; t < NUM_TREES; t++) {
        int hash = 0;
        for (int d = 0; d < DIMS_PER_TREE; d++) {
            hash ^= ((int)(features_in[feature_map[t * DIMS_PER_TREE + d] * n_train + idx] / cell_size) * primes[d]);
        }
        hash = hash % num_buckets; if (hash < 0) hash += num_buckets;
        int loc = (O[t * num_buckets + hash] - H[t * num_buckets + hash]) + atomicAdd(&C[t * num_buckets + hash], 1);
        for (int d = 0; d < DIMS_PER_TREE; d++) A_features[t * DIMS_PER_TREE * n_train + d * n_train + loc] = features_in[feature_map[t * DIMS_PER_TREE + d] * n_train + idx];
        A_label[t * n_train + loc] = labels_in[idx];
    }
}

__global__ void kernel_infer_cascade(
    float* test_features, float* out_momentum, int* primes, float* tree_weights, int* feature_map, 
    float* A_features, int* A_label, int* H, int* O, 
    int n_train, int n_test, float cell_size, int num_buckets, float search_radius, float gauss_var
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_test) return;
    float momentum = 0.0f; 
    for (int t = 0; t < NUM_TREES; t++) {
        int hash = 0;
        for (int d = 0; d < DIMS_PER_TREE; d++) hash ^= ((int)(test_features[feature_map[t * DIMS_PER_TREE + d] * n_test + idx] / cell_size) * primes[d]);
        hash = hash % num_buckets; if (hash < 0) hash += num_buckets;
        int start = O[t * num_buckets + hash] - H[t * num_buckets + hash], count = H[t * num_buckets + hash];
        float m0 = 0.0f, m1 = 0.0f;
        for (int i = 0; i < count; i++) {
            float d_l1 = 0.0f;
            for (int d = 0; d < DIMS_PER_TREE; d++) d_l1 += fabsf(A_features[t * DIMS_PER_TREE * n_train + d * n_train + (start + i)] - test_features[feature_map[t * DIMS_PER_TREE + d] * n_test + idx]);
            if (d_l1 <= search_radius) {
                float e = expf(-(d_l1 * d_l1) / gauss_var); 
                if (A_label[t * n_train + start + i] == 0) m0 += e; else m1 += e;
            }
        }
        if ((m0 + m1) > 0.0f) momentum += ((m1 - m0) / (m1 + m0)) * tree_weights[t];
    }
    out_momentum[idx] = momentum;
}
"""


def run_auto_tuner(X, y, output_html="umer_resonance_map.html"):
    """Accepts custom datasets and finds the Spatial Resonance Peak."""
    TOTAL_DIMS = X.shape[1]
    DIMS_PER_TREE, STRIDE = 4, 2
    NUM_TREES = ((TOTAL_DIMS - DIMS_PER_TREE) // STRIDE) + 1
    HASH_BUCKETS, CELL_SIZE, GAUSS_VAR = 2048, 0.20, 0.10

    # 1. Prepare Data
    scaler = MinMaxScaler()
    X_norm = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_norm, y, test_size=0.2, random_state=42)

    X_tr_flat = np.ascontiguousarray(X_train.T).astype(np.float32).flatten()
    X_te_flat = np.ascontiguousarray(X_test.T).astype(np.float32).flatten()
    y_tr_flat = y_train.astype(np.int32)

    # 2. PGP Phase (Feature Mapping)
    pairs = list(itertools.combinations(range(TOTAL_DIMS), 2))
    num_pairs = len(pairs)
    px_d = np.array([p[0] for p in pairs], dtype=np.int32)
    py_d = np.array([p[1] for p in pairs], dtype=np.int32)

    mod_pgp = SourceModule(CUDA_PGP)
    k_pgp = mod_pgp.get_function("kernel_pgp_histogram")

    d_xt = cuda.mem_alloc(X_tr_flat.nbytes);
    d_lt = cuda.mem_alloc(y_tr_flat.nbytes)
    cuda.memcpy_htod(d_xt, X_tr_flat);
    cuda.memcpy_htod(d_lt, y_tr_flat)
    d_px = cuda.mem_alloc(px_d.nbytes);
    d_py = cuda.mem_alloc(py_d.nbytes)
    cuda.memcpy_htod(d_px, px_d);
    cuda.memcpy_htod(d_py, py_d)

    d_H0 = cuda.mem_alloc(num_pairs * HASH_BUCKETS * 4)
    d_H1 = cuda.mem_alloc(num_pairs * HASH_BUCKETS * 4)
    cuda.memset_d32(d_H0, 0, num_pairs * HASH_BUCKETS)
    cuda.memset_d32(d_H1, 0, num_pairs * HASH_BUCKETS)

    grid_size = (int((len(X_train) + 255) // 256), 1)
    k_pgp(d_xt, d_lt, d_px, d_py, d_H0, d_H1, np.int32(len(X_train)), np.float32(CELL_SIZE), np.int32(HASH_BUCKETS),
          np.int32(num_pairs), block=(256, 1, 1), grid=grid_size)

    h0 = np.zeros(num_pairs * HASH_BUCKETS, dtype=np.int32);
    h1 = np.zeros(num_pairs * HASH_BUCKETS, dtype=np.int32)
    cuda.memcpy_dtoh(h0, d_H0);
    cuda.memcpy_dtoh(h1, d_H1)
    syn_scores = np.sum(((h1.reshape(num_pairs, HASH_BUCKETS) - h0.reshape(num_pairs, HASH_BUCKETS)) ** 2) /
                        (h1.reshape(num_pairs, HASH_BUCKETS) + h0.reshape(num_pairs, HASH_BUCKETS) + 1e-9), axis=1)

    # 3. Topological Weave
    best_p = np.argmax(syn_scores)
    f_list, used = [pairs[best_p][0], pairs[best_p][1]], {pairs[best_p][0], pairs[best_p][1]}
    while len(f_list) < TOTAL_DIMS:
        best_s, best_f = -1.0, -1
        tail = f_list[-2:]
        for f in range(TOTAL_DIMS):
            if f in used: continue
            curr_s = sum(syn_scores[i] for i, p in enumerate(pairs) if
                         (p[0] == f and p[1] in tail) or (p[1] == f and p[0] in tail))
            if curr_s > best_s: best_s, best_f = curr_s, f
        f_list.append(best_f);
        used.add(best_f)

    f_map = np.array([f_list[t * STRIDE: t * STRIDE + DIMS_PER_TREE] for t in range(NUM_TREES)],
                     dtype=np.int32).flatten()
    raw_t_weights = np.array([sum(syn_scores[i] for i, p in enumerate(pairs) if
                                  p[0] in f_list[t * 2:t * 2 + 4] and p[1] in f_list[t * 2:t * 2 + 4]) for t in
                              range(NUM_TREES)])
    raw_t_weights = (raw_t_weights / np.max(raw_t_weights)).astype(np.float32)

    # 4. Resonance Sweep
    search_radii = np.arange(0.05, 0.40, 0.02).astype(np.float32)
    decay_rates = np.arange(0.00, 0.50, 0.02).astype(np.float32)
    results = np.zeros((len(decay_rates), len(search_radii)))

    mod_c = SourceModule(
        CUDA_CASCADE.replace("{num_trees}", str(NUM_TREES)).replace("{dims_per_tree}", str(DIMS_PER_TREE)))
    k_h, k_p, k_s, k_i = mod_c.get_function("kernel_histogram"), mod_c.get_function(
        "kernel_prefix_sum"), mod_c.get_function("kernel_scatter"), mod_c.get_function("kernel_infer_cascade")

    d_x_te = cuda.mem_alloc(X_te_flat.nbytes);
    d_m = cuda.mem_alloc(len(X_test) * 4)
    cuda.memcpy_htod(d_x_te, X_te_flat)
    d_fm = cuda.mem_alloc(f_map.nbytes);
    d_pr = cuda.mem_alloc(16)
    cuda.memcpy_htod(d_fm, f_map);
    cuda.memcpy_htod(d_pr, np.array(PRIME_POOL[:4], dtype=np.int32))

    d_Af = cuda.mem_alloc(NUM_TREES * 4 * len(X_train) * 4);
    d_Al = cuda.mem_alloc(NUM_TREES * len(X_train) * 4)
    d_H = cuda.mem_alloc(NUM_TREES * HASH_BUCKETS * 4);
    d_O = cuda.mem_alloc(NUM_TREES * HASH_BUCKETS * 4);
    d_C = cuda.mem_alloc(NUM_TREES * HASH_BUCKETS * 4)

    cuda.memset_d32(d_H, 0, NUM_TREES * HASH_BUCKETS);
    cuda.memset_d32(d_O, 0, NUM_TREES * HASH_BUCKETS);
    cuda.memset_d32(d_C, 0, NUM_TREES * HASH_BUCKETS)
    k_h(d_xt, d_pr, d_fm, d_H, np.int32(len(X_train)), np.float32(CELL_SIZE), np.int32(HASH_BUCKETS), block=(256, 1, 1),
        grid=grid_size)
    k_p(d_H, d_O, np.int32(HASH_BUCKETS), block=(NUM_TREES, 1, 1), grid=(1, 1))
    k_s(d_xt, d_lt, d_fm, d_Af, d_Al, d_pr, d_H, d_O, d_C, np.int32(len(X_train)), np.float32(CELL_SIZE),
        np.int32(HASH_BUCKETS), block=(256, 1, 1), grid=grid_size)

    d_tw = cuda.mem_alloc(NUM_TREES * 4)
    best_acc, best_coords = 0.0, (0, 0)

    for i, dr in enumerate(decay_rates):
        d_weights = (raw_t_weights * np.exp(-dr * np.arange(NUM_TREES))).astype(np.float32)
        cuda.memcpy_htod(d_tw, d_weights)
        for j, sr in enumerate(search_radii):
            k_i(d_x_te, d_m, d_pr, d_tw, d_fm, d_Af, d_Al, d_H, d_O, np.int32(len(X_train)), np.int32(len(X_test)),
                np.float32(CELL_SIZE), np.int32(HASH_BUCKETS), np.float32(sr), np.float32(GAUSS_VAR), block=(256, 1, 1),
                grid=(int((len(X_test) + 255) // 256), 1))
            m_h = np.zeros(len(X_test), dtype=np.float32);
            cuda.memcpy_dtoh(m_h, d_m)
            acc = accuracy_score(y_test, (m_h > 0.0).astype(np.int32)) * 100
            results[i, j] = acc
            if acc > best_acc: best_acc = acc; best_coords = (dr, sr)

    print(f"✅ Auto-Tuner Peak: {best_acc:.2f}% (Decay: {best_coords[0]:.2f}, Radius: {best_coords[1]:.2f})")

    # 5. Export Dashboard
    fig = go.Figure(data=[go.Surface(z=results, x=search_radii, y=decay_rates, colorscale='Magma')])
    fig.update_layout(title=f"U.M.E.R. Resonance Scan ({TOTAL_DIMS}D)",
                      scene=dict(xaxis_title='Radius', yaxis_title='Decay', zaxis_title='Accuracy %'),
                      paper_bgcolor='rgb(10,10,10)', font=dict(color='white'))
    pio.write_html(fig, file=output_html, auto_open=False)

    # Cleanup
    for d in [d_xt, d_lt, d_px, d_py, d_H0, d_H1, d_x_te, d_m, d_fm, d_pr, d_Af, d_Al, d_H, d_O, d_C, d_tw]: d.free()
    return best_acc, best_coords[0], best_coords[1]