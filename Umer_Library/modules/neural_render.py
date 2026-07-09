# Umer_Library/modules/neural_render.py

import pycuda.autoinit
import pycuda.driver as cuda
import numpy as np
import imageio
from Umer_Library.core.neural_memory import UMER_Neural_Context

# =====================================================================
# THE DYNAMIC SPLAT SHADER (now fully parameterised)
# =====================================================================
DYNAMIC_SPLAT_SHADER = """
__global__ void evaluate_splats(
    const float4* __restrict__ ray_origins, const float4* __restrict__ ray_dirs,
    const float3* __restrict__ global_min, const float3* __restrict__ global_max,
    const int* __restrict__ histogram, const float4* __restrict__ splat_colors,
    unsigned char* __restrict__ image, int* __restrict__ dummy_evals, int num_rays, int mask)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_rays) return;

    float3 pos = make_float3(ray_origins[idx].x, ray_origins[idx].y, ray_origins[idx].z);
    float3 dir = make_float3(ray_dirs[idx].x, ray_dirs[idx].y, ray_dirs[idx].z);

    float t_hit = intersect_aabb(pos, dir, global_min[0], global_max[0]);
    if (t_hit > 0.0f) { pos.x += dir.x * t_hit; pos.y += dir.y * t_hit; pos.z += dir.z * t_hit; }

    float accum_alpha = 0.0f;
    float3 final_color = make_float3(0.0f, 0.0f, 0.0f);

    // NEURAL_MAX_STEPS and NEURAL_COLOR_COUNT are injected as macros
    for(int i = 0; i < NEURAL_MAX_STEPS; i++) {
        pos.x += dir.x * CELL_W; pos.y += dir.y * CELL_W; pos.z += dir.z * CELL_W;
        int h = hash3d(pos.x, pos.y, pos.z, mask);

        if (histogram[h] > 0) {
            float4 s_data = splat_colors[h % NEURAL_COLOR_COUNT];
            float weight = s_data.w * (1.0f - accum_alpha);
            final_color.x += s_data.x * weight;
            final_color.y += s_data.y * weight;
            final_color.z += s_data.z * weight;
            accum_alpha += weight;
            if (accum_alpha >= 0.99f) break;
        }
    }
    image[idx * 3]     = (unsigned char)(fminf(final_color.x, 1.0f) * 255.0f);
    image[idx * 3 + 1] = (unsigned char)(fminf(final_color.y, 1.0f) * 255.0f);
    image[idx * 3 + 2] = (unsigned char)(fminf(final_color.z, 1.0f) * 255.0f);
}
"""


def generate_camera_rays(width, height):
    num_rays = width * height
    ray_origins = np.zeros((num_rays, 4), dtype=np.float32)
    ray_dirs = np.zeros((num_rays, 4), dtype=np.float32)
    ray_origins[:, 0] = 500.0
    ray_origins[:, 1] = 500.0
    ray_origins[:, 2] = 200.0

    u = np.linspace(-1, 1, width)
    v = np.linspace(-1, 1, height) * (height / width)
    uu, vv = np.meshgrid(u, v)
    ray_dirs[:, 0] = uu.flatten()
    ray_dirs[:, 1] = -vv.flatten()
    ray_dirs[:, 2] = 1.0
    norms = np.linalg.norm(ray_dirs[:, :3], axis=1, keepdims=True)
    ray_dirs[:, :3] = ray_dirs[:, :3] / norms
    return ray_origins, ray_dirs


def run_neural_rendering_pipeline(width=640, height=480,
                                  n_static=1_800_000, n_dynamic=200_000,
                                  max_steps=600, fps=30, frames=60,
                                  output_file='/kaggle/working/umer_neural_render.mp4'):
    WIDTH, HEIGHT = width, height
    NUM_RAYS = WIDTH * HEIGHT
    N_STATIC, N_DYNAMIC = n_static, n_dynamic
    FPS, FRAMES = fps, frames

    rng = np.random.default_rng(seed=42)
    pos_host = np.zeros((N_STATIC + N_DYNAMIC, 4), dtype=np.float32)
    pos_host[:, 0] = rng.uniform(400.0, 600.0, N_STATIC + N_DYNAMIC)
    pos_host[:, 1] = rng.uniform(400.0, 600.0, N_STATIC + N_DYNAMIC)
    pos_host[:, 2] = rng.uniform(400.0, 600.0, N_STATIC + N_DYNAMIC)

    # Color palette size – change this to any number you like
    num_colors = 10000
    splat_colors = np.zeros((num_colors, 4), dtype=np.float32)
    splat_colors[:, 0] = rng.uniform(0.1, 1.0, num_colors)
    splat_colors[:, 1] = rng.uniform(0.1, 0.5, num_colors)
    splat_colors[:, 2] = rng.uniform(0.5, 1.0, num_colors)
    splat_colors[:, 3] = rng.uniform(0.05, 0.25, num_colors)

    d_colors = cuda.mem_alloc(splat_colors.nbytes)
    cuda.memcpy_htod(d_colors, splat_colors)

    d_pos_static = cuda.mem_alloc(N_STATIC * 16)
    d_pos_dynamic = cuda.mem_alloc(N_DYNAMIC * 16)
    cuda.memcpy_htod(d_pos_static, pos_host[:N_STATIC])
    cuda.memcpy_htod(d_pos_dynamic, pos_host[N_STATIC:])

    ray_orig, ray_dir = generate_camera_rays(WIDTH, HEIGHT)
    d_ro = cuda.mem_alloc(ray_orig.nbytes)
    d_rd = cuda.mem_alloc(ray_dir.nbytes)
    cuda.memcpy_htod(d_ro, ray_orig)
    cuda.memcpy_htod(d_rd, ray_dir)

    bmin = np.array([[399.0, 399.0, 399.0]], dtype=np.float32)
    bmax = np.array([[601.0, 601.0, 601.0]], dtype=np.float32)
    d_bmin = cuda.mem_alloc(bmin.nbytes)
    d_bmax = cuda.mem_alloc(bmax.nbytes)
    cuda.memcpy_htod(d_bmin, bmin)
    cuda.memcpy_htod(d_bmax, bmax)

    image_bytes = NUM_RAYS * 3
    d_image = cuda.mem_alloc(image_bytes)
    image_host = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)

    # --- Engine setup ---
    engine = UMER_Neural_Context()

    # Inject the macros directly before the shader string
    shader_with_defines = (
        f"#define NEURAL_MAX_STEPS {max_steps}\n"
        f"#define NEURAL_COLOR_COUNT {num_colors}\n"
        + DYNAMIC_SPLAT_SHADER
    )
    engine.inject_neural_evaluator(shader_with_defines, "evaluate_splats")

    engine.initialize_static_geometry(d_pos_static, N_STATIC)
    writer = imageio.get_writer(output_file, fps=FPS)

    for f in range(FRAMES):
        # Animate dynamic particles
        pos_host[N_STATIC:, 0] += np.sin(f * 0.1) * 2.0
        pos_host[N_STATIC:, 1] += np.cos(f * 0.1) * 2.0
        d_pos_dynamic_new = cuda.mem_alloc(N_DYNAMIC * 16)
        cuda.memcpy_htod(d_pos_dynamic_new, pos_host[N_STATIC:])

        engine.amortized_dynamic_update(d_pos_dynamic, d_pos_dynamic_new, N_DYNAMIC)
        engine.render_volume(d_ro, d_rd, d_bmin, d_bmax, d_colors, NUM_RAYS, d_image)
        cuda.Context.synchronize()

        cuda.memcpy_dtoh(image_host, d_image)
        writer.append_data(image_host)
        d_pos_dynamic.free()
        d_pos_dynamic = d_pos_dynamic_new

    writer.close()
    engine.cleanup()
    d_pos_static.free()
    d_pos_dynamic.free()
    d_ro.free()
    d_rd.free()
    d_image.free()
    d_colors.free()
    d_bmin.free()
    d_bmax.free()
    print(f"\n✅ RENDER COMPLETE! Output saved to: {output_file}")