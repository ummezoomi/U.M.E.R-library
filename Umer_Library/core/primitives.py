# umer_engine/core/primitives.py

PRIME_POOL = [73856093, 19349663, 83492791, 23948573]

CUDA_PGP = """
#include <stdint.h>
__global__ void kernel_pgp_histogram(
    float* features_in, float* mass_in, int* pairs_x, int* pairs_y,
    float* H_obstacle, int n_env, float cell_size, int num_buckets, int num_pairs
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_env) return;
    float mass = mass_in[idx];
    
    #pragma unroll 4
    for (int p = 0; p < num_pairs; p++) {
        int dx = pairs_x[p], dy = pairs_y[p];
        int ix = (int)(features_in[dx * n_env + idx] / cell_size);
        int iy = (int)(features_in[dy * n_env + idx] / cell_size);
        int hash = ((ix * 73856093) ^ (iy * 19349663)) % num_buckets;
        if (hash < 0) hash += num_buckets;
        
        atomicAdd(&H_obstacle[p * num_buckets + hash], -mass); 
    }
}
"""

CUDA_CASCADE = """
#include <stdint.h>
#include <math.h>
#define NUM_TREES {num_trees}
#define DIMS_PER_TREE {dims_per_tree}

__global__ void kernel_histogram(float* features_in, int* primes, int* feature_map, int* H, int n_env, float cell_size, int num_buckets) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_env) return;
    for (int t = 0; t < NUM_TREES; t++) {
        int hash = 0;
        for (int d = 0; d < DIMS_PER_TREE; d++) {
            hash ^= ((int)(features_in[feature_map[t * DIMS_PER_TREE + d] * n_env + idx] / cell_size) * primes[d]);
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

__global__ void kernel_scatter(float* features_in, float* mass_in, int* feature_map, float* A_features, float* A_mass, int* primes, int* H, int* O, int* C, int n_env, float cell_size, int num_buckets) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_env) return;
    for (int t = 0; t < NUM_TREES; t++) {
        int hash = 0;
        for (int d = 0; d < DIMS_PER_TREE; d++) {
            hash ^= ((int)(features_in[feature_map[t * DIMS_PER_TREE + d] * n_env + idx] / cell_size) * primes[d]);
        }
        hash = hash % num_buckets; if (hash < 0) hash += num_buckets;
        int loc = (O[t * num_buckets + hash] - H[t * num_buckets + hash]) + atomicAdd(&C[t * num_buckets + hash], 1);
        
        for (int d = 0; d < DIMS_PER_TREE; d++) A_features[t * DIMS_PER_TREE * n_env + d * n_env + loc] = features_in[feature_map[t * DIMS_PER_TREE + d] * n_env + idx];
        A_mass[t * n_env + loc] = mass_in[idx];
    }
}
"""