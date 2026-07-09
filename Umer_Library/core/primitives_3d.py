# Umer_Library/core/primitives_3d.py

CUDA_PHYSICS_CORE = """
#define CELL_W  12.0f
#define DOMAIN  1000.0f

__device__ __forceinline__ int hash3d(float x, float y, float z, int mask) {
    int cx = (int)(x / CELL_W);
    int cy = (int)(y / CELL_W);
    int cz = (int)(z / CELL_W);
    return (cx * 73856093 ^ cy * 19349663 ^ cz * 83492791) & mask;
}

__global__ void build_histogram_full(
    const float4* __restrict__ pos4, int* __restrict__ histogram,
    int* __restrict__ prev_hash, int n, int mask)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    float4 p = pos4[idx];
    int h = hash3d(p.x, p.y, p.z, mask);
    atomicAdd(&histogram[h], 1);
    prev_hash[idx] = h;
}

__global__ void update_histogram_delta(
    const float4* __restrict__ pos4, int* __restrict__ histogram,
    int* __restrict__ prev_hash, int* __restrict__ is_mover,
    int* __restrict__ mover_count, int n, int mask)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    float4 p  = pos4[idx];
    int new_h = hash3d(p.x, p.y, p.z, mask);
    int old_h = prev_hash[idx];
    if (new_h == old_h) { is_mover[idx] = 0; return; }
    atomicAdd(&histogram[old_h], -1);
    atomicAdd(&histogram[new_h], +1);
    prev_hash[idx] = new_h;
    is_mover[idx]  = 1;
    atomicAdd(mover_count, 1);
}

__global__ void scan_blocks(
    const int* __restrict__ in, int* __restrict__ out,
    int* __restrict__ block_sums, int n)
{
    extern __shared__ int smem[];
    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + tid;
    smem[tid] = (gid < n) ? in[gid] : 0;
    __syncthreads();
    for (int s = 1; s < blockDim.x; s <<= 1) {
        int i = (tid + 1) * (s << 1) - 1;
        if (i < blockDim.x) smem[i] += smem[i - s];
        __syncthreads();
    }
    if (tid == blockDim.x - 1) {
        if (block_sums) block_sums[blockIdx.x] = smem[tid];
        smem[tid] = 0;
    }
    __syncthreads();
    for (int s = blockDim.x >> 1; s > 0; s >>= 1) {
        int i = (tid + 1) * (s << 1) - 1;
        if (i < blockDim.x) {
            int t = smem[i - s]; smem[i - s] = smem[i]; smem[i] += t;
        }
        __syncthreads();
    }
    if (gid < n) out[gid] = smem[tid];
}

__global__ void add_offsets(int* __restrict__ data, const int* __restrict__ offsets, int n) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid < n) data[gid] += offsets[blockIdx.x];
}

__global__ void scatter_full(
    const float4* __restrict__ pos4_in, const float4* __restrict__ vel4_in,
    const int* __restrict__ offsets, int* __restrict__ local_counts,
    float4* __restrict__ pos4_out, float4* __restrict__ vel4_out,
    int n, int mask)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    float4 p = pos4_in[idx]; float4 v = vel4_in[idx];
    int h = hash3d(p.x, p.y, p.z, mask);
    int slot = offsets[h] + atomicAdd(&local_counts[h], 1);
    pos4_out[slot] = p; vel4_out[slot] = v;
}

__global__ void scatter_movers(
    const float4* __restrict__ pos4_in, const float4* __restrict__ vel4_in,
    const int* __restrict__ is_mover, const int* __restrict__ offsets,
    int* __restrict__ local_counts,
    float4* __restrict__ pos4_out, float4* __restrict__ vel4_out,
    int n, int mask)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    if (!is_mover[idx]) return;
    float4 p = pos4_in[idx]; float4 v = vel4_in[idx];
    int h = hash3d(p.x, p.y, p.z, mask);
    int slot = offsets[h] + atomicAdd(&local_counts[h], 1);
    pos4_out[slot] = p; vel4_out[slot] = v;
}
"""