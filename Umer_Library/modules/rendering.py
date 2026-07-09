# Umer_Library/modules/rendering.py

import pycuda.autoinit
import pycuda.driver as cuda
import numpy as np
import imageio
from Umer_Library.core.optics_memory import UMER_Optics_Context

# =====================================================================
# MULTI‑SHAPE SDF SHADER (unchanged)
# =====================================================================
DYNAMIC_SHADER = r"""
__device__ float3 sub(float3 a, float3 b) { return make_float3(a.x - b.x, a.y - b.y, a.z - b.z); }
__device__ float3 add(float3 a, float3 b) { return make_float3(a.x + b.x, a.y + b.y, a.z + b.z); }
__device__ float3 mul(float3 a, float b)  { return make_float3(a.x * b, a.y * b, a.z * b); }
__device__ float dot(float3 a, float3 b)  { return a.x * b.x + a.y * b.y + a.z * b.z; }
__device__ float3 normalize(float3 v)     { float inv = rsqrtf(dot(v, v)); return mul(v, inv); }

__device__ float sdf_sphere(float3 p, float r) {
    return sqrtf(p.x*p.x + p.y*p.y + p.z*p.z) - r;
}
__device__ float sdf_box(float3 p, float3 half_ext) {
    float3 q = sub(make_float3(fabsf(p.x), fabsf(p.y), fabsf(p.z)), half_ext);
    return fminf(fmaxf(q.x, fmaxf(q.y, q.z)), 0.0f) +
           sqrtf(fmaxf(q.x,0.f)*fmaxf(q.x,0.f)+fmaxf(q.y,0.f)*fmaxf(q.y,0.f)+fmaxf(q.z,0.f)*fmaxf(q.z,0.f));
}
__device__ float sdf_torus(float3 p, float main_r, float tube_r) {
    float2 horizontal = make_float2(sqrtf(p.x*p.x + p.z*p.z) - main_r, p.y);
    return sqrtf(horizontal.x*horizontal.x + horizontal.y*horizontal.y) - tube_r;
}
__device__ float sdf_cylinder(float3 p, float r, float half_h) {
    float2 d = make_float2(sqrtf(p.x*p.x + p.z*p.z) - r, fabsf(p.y) - half_h);
    return fminf(fmaxf(d.x, d.y), 0.f) + sqrtf(fmaxf(d.x,0.f)*fmaxf(d.x,0.f)+fmaxf(d.y,0.f)*fmaxf(d.y,0.f));
}
__device__ float sdf_mandelbulb(float3 p, float power, int max_iters) {
    float3 z = p; float dr = 1.f; float r = 0.f;
    for (int i = 0; i < max_iters; ++i) {
        r = sqrtf(z.x*z.x + z.y*z.y + z.z*z.z);
        if (r > 2.f) break;
        float theta = acosf(z.z / r);
        float phi   = atan2f(z.y, z.x);
        dr  = powf(r, power - 1.f) * power * dr + 1.f;
        float zr  = powf(r, power);
        theta *= power; phi *= power;
        z = make_float3(zr * sinf(theta)*cosf(phi), zr * sinf(theta)*sinf(phi), zr * cosf(theta));
        z = add(z, p);
    }
    return 0.5f * logf(r) * r / dr;
}

__global__ void execute_materials(
    const int* __restrict__ active_ray_count,
    const float4* __restrict__ queue_pos,
    const float4* __restrict__ queue_dir,
    const int* __restrict__ queue_id,
    unsigned char* __restrict__ image,
    const float4* __restrict__ scene_data,
    int num_active_rays)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= *active_ray_count || idx >= num_active_rays) return;

    float3 pos = make_float3(queue_pos[idx].x, queue_pos[idx].y, queue_pos[idx].z);
    float3 dir = make_float3(queue_dir[idx].x, queue_dir[idx].y, queue_dir[idx].z);
    int pixel_id = queue_id[idx];

    float t_min = 99999.f;
    float3 final_color = make_float3(0.05f, 0.05f, 0.05f);
    float3 light_pos = make_float3(505.f, 510.f, 505.f);

    for (int obj = 0; obj < NUM_OBJECTS; obj++) {
        float4 d0 = scene_data[obj * 4 + 0];
        float4 d1 = scene_data[obj * 4 + 1];
        float4 d2 = scene_data[obj * 4 + 2];

        int   type   = (int)d0.x;
        float3 center = make_float3(d0.y, d0.z, d0.w);
        float paramA  = d1.x;
        float paramB  = d1.y;
        float paramC  = d1.z;
        float3 obj_color = make_float3(d2.x, d2.y, d2.z);
        float spec_power = d2.w;

        float3 local_pos = sub(pos, center);
        float t = 0.f;
        bool hit = false;
        float hit_t = 0.f;
        float3 hit_normal = make_float3(0,0,0);

        for (int step = 0; step < MAX_MARCH_STEPS; step++) {
            float3 p = add(local_pos, mul(dir, t));
            float d = 1e10f;
            switch (type) {
                case 1: d = sdf_sphere(p, paramA); break;
                case 2: d = sdf_torus(p, paramA, paramB); break;
                case 3: d = sdf_box(p, make_float3(paramA, paramB, paramC)); break;
                case 4: d = sdf_cylinder(p, paramA, paramB); break;
                case 5: d = sdf_mandelbulb(p, paramA, (int)paramB); break;
            }
            if (d < 0.001f) {
                hit = true; hit_t = t;
                float eps = 0.001f;
                float dx1,dx2,dy1,dy2,dz1,dz2;
                switch (type) {
                    case 1: dx1 = sdf_sphere(add(p, make_float3(eps,0,0)), paramA); dx2 = sdf_sphere(sub(p, make_float3(eps,0,0)), paramA);
                            dy1 = sdf_sphere(add(p, make_float3(0,eps,0)), paramA); dy2 = sdf_sphere(sub(p, make_float3(0,eps,0)), paramA);
                            dz1 = sdf_sphere(add(p, make_float3(0,0,eps)), paramA); dz2 = sdf_sphere(sub(p, make_float3(0,0,eps)), paramA); break;
                    case 2: dx1 = sdf_torus(add(p, make_float3(eps,0,0)), paramA, paramB); dx2 = sdf_torus(sub(p, make_float3(eps,0,0)), paramA, paramB);
                            dy1 = sdf_torus(add(p, make_float3(0,eps,0)), paramA, paramB); dy2 = sdf_torus(sub(p, make_float3(0,eps,0)), paramA, paramB);
                            dz1 = sdf_torus(add(p, make_float3(0,0,eps)), paramA, paramB); dz2 = sdf_torus(sub(p, make_float3(0,0,eps)), paramA, paramB); break;
                    case 3: dx1 = sdf_box(add(p, make_float3(eps,0,0)), make_float3(paramA,paramB,paramC)); dx2 = sdf_box(sub(p, make_float3(eps,0,0)), make_float3(paramA,paramB,paramC));
                            dy1 = sdf_box(add(p, make_float3(0,eps,0)), make_float3(paramA,paramB,paramC)); dy2 = sdf_box(sub(p, make_float3(0,eps,0)), make_float3(paramA,paramB,paramC));
                            dz1 = sdf_box(add(p, make_float3(0,0,eps)), make_float3(paramA,paramB,paramC)); dz2 = sdf_box(sub(p, make_float3(0,0,eps)), make_float3(paramA,paramB,paramC)); break;
                    case 4: dx1 = sdf_cylinder(add(p, make_float3(eps,0,0)), paramA, paramB); dx2 = sdf_cylinder(sub(p, make_float3(eps,0,0)), paramA, paramB);
                            dy1 = sdf_cylinder(add(p, make_float3(0,eps,0)), paramA, paramB); dy2 = sdf_cylinder(sub(p, make_float3(0,eps,0)), paramA, paramB);
                            dz1 = sdf_cylinder(add(p, make_float3(0,0,eps)), paramA, paramB); dz2 = sdf_cylinder(sub(p, make_float3(0,0,eps)), paramA, paramB); break;
                    case 5: dx1 = sdf_mandelbulb(add(p, make_float3(eps,0,0)), paramA, (int)paramB); dx2 = sdf_mandelbulb(sub(p, make_float3(eps,0,0)), paramA, (int)paramB);
                            dy1 = sdf_mandelbulb(add(p, make_float3(0,eps,0)), paramA, (int)paramB); dy2 = sdf_mandelbulb(sub(p, make_float3(0,eps,0)), paramA, (int)paramB);
                            dz1 = sdf_mandelbulb(add(p, make_float3(0,0,eps)), paramA, (int)paramB); dz2 = sdf_mandelbulb(sub(p, make_float3(0,0,eps)), paramA, (int)paramB); break;
                    default: dx1=dx2=dy1=dy2=dz1=dz2=0.f; break;
                }
                hit_normal = normalize(make_float3(dx1-dx2, dy1-dy2, dz1-dz2));
                break;
            }
            t += d * MARCH_STEP_SIZE;
            if (t > t_min || t > 99999.f) break;
        }

        if (hit && hit_t < t_min) {
            t_min = hit_t;
            float3 surface_pt = add(pos, mul(dir, hit_t));
            float3 light_dir = normalize(sub(light_pos, surface_pt));
            float diff = fmaxf(dot(hit_normal, light_dir), 0.1f);
            float3 view_dir = normalize(sub(pos, surface_pt));
            float3 reflect_dir = sub(mul(hit_normal, 2.f * dot(hit_normal, light_dir)), light_dir);
            float spec = powf(fmaxf(dot(view_dir, reflect_dir), 0.f), spec_power);
            final_color = make_float3(obj_color.x * diff + spec,
                                      obj_color.y * diff + spec,
                                      obj_color.z * diff + spec);
        }
    }

    if (t_min < 99999.f) {
        image[pixel_id * 3]   = (unsigned char)(fminf(final_color.x, 1.f) * 255.f);
        image[pixel_id * 3+1] = (unsigned char)(fminf(final_color.y, 1.f) * 255.f);
        image[pixel_id * 3+2] = (unsigned char)(fminf(final_color.z, 1.f) * 255.f);
    }
}
"""

# =====================================================================
# SCENE BUILDER – fully dynamic
# =====================================================================
def build_scene_from_objects(objects):
    """
    Args:
        objects: list of tuples (type, center, params, color_spec)
                 type   = 1..5
                 center = (x,y,z)
                 params = (paramA, paramB, paramC)  (meaning depends on type)
                 color_spec = (r,g,b, specular_power)
    Returns:
        scene_data : np.array (num_objects*4, 4) float32
        centers_4d : np.array (num_objects, 4) float32  (for macro grid, w=0)
        num_objects: int
    """
    num_objects = len(objects)
    scene_data = np.zeros((num_objects * 4, 4), dtype=np.float32)
    centers_4d = np.zeros((num_objects, 4), dtype=np.float32)

    for i, (typ, center, params, color_spec) in enumerate(objects):
        scene_data[i*4]     = [float(typ), center[0], center[1], center[2]]
        scene_data[i*4 + 1] = [params[0], params[1], params[2], 0.0]
        scene_data[i*4 + 2] = [color_spec[0], color_spec[1], color_spec[2], color_spec[3]]
        centers_4d[i, :3] = center   # w stays 0

    return scene_data, centers_4d, num_objects


def default_scene_objects():
    """Returns the default demo scene as a list of object tuples."""
    return [
        # type 1 = sphere, 2 = torus, 3 = box, 4 = cylinder, 5 = mandelbulb
        (1, (500., 500., 500.), (1.0, 0, 0), (1.0, 0.2, 0.2, 32.)),      # red sphere
        (2, (498., 498., 502.), (2.0, 0.5, 0), (0.2, 1.0, 0.2, 64.)),    # green torus
        (3, (503., 503., 497.), (1.0, 0.8, 0.6), (0.2, 0.2, 1.0, 32.)),  # blue box
        (4, (495., 495., 505.), (1.2, 1.5, 0), (1.0, 1.0, 0.2, 40.)),    # yellow cylinder
        (5, (500., 502., 500.), (8.0, 20.0, 0), (0.9, 0.9, 0.9, 64.))    # white Mandelbulb
    ]

# =====================================================================
# CAMERA
# =====================================================================
def generate_camera_rays(width, height, cam_pos=(500.0, 500.0, 520.0), tilt=0.0):
    print(f"[SYSTEM] Calibrating Ray Matrix. Camera: {cam_pos}")
    num_rays = width * height
    ray_origins = np.zeros((num_rays, 4), dtype=np.float32)
    ray_dirs = np.zeros((num_rays, 4), dtype=np.float32)

    ray_origins[:, 0] = cam_pos[0]
    ray_origins[:, 1] = cam_pos[1]
    ray_origins[:, 2] = cam_pos[2]

    u = np.linspace(-1, 1, width)
    v = np.linspace(-1, 1, height) * (height / width)
    uu, vv = np.meshgrid(u, v)
    ray_dirs[:, 0] = uu.flatten()
    ray_dirs[:, 1] = -vv.flatten() + tilt
    ray_dirs[:, 2] = -1.0

    norms = np.linalg.norm(ray_dirs[:, :3], axis=1, keepdims=True)
    ray_dirs[:, :3] = ray_dirs[:, :3] / norms
    return ray_origins, ray_dirs


# =====================================================================
# MAIN RENDERING PIPELINE – now accepts optional scene_objects
# =====================================================================
def run_rendering_pipeline(width=640, height=640, fps=30, duration=5,
                           scene_objects=None,
                           max_march_steps=64,
                           march_step_size=0.8,
                           macro_width=8.0,
                           output_file='/kaggle/working/umer_wavefront_render.mp4'):
    """
    If scene_objects is None, a default demo scene of 5 objects is used.
    Otherwise, pass a list of (type, center, params, color_spec) tuples.
    """
    NUM_RAYS = width * height
    FRAMES = fps * duration

    print(f"\n[U.M.E.R] Booting 3D Optical Engine ({width}x{height} Resolution)...")

    # Use provided scene or default
    objects = scene_objects if scene_objects is not None else default_scene_objects()
    scene_data, centers_4d, num_objects = build_scene_from_objects(objects)
    scene_flat = scene_data.reshape(-1)

    # Upload scene buffer (for SDF ray‑marching)
    d_scene = cuda.mem_alloc(scene_flat.nbytes)
    cuda.memcpy_htod(d_scene, scene_flat)

    # Upload macro‑grid positions (float4, w=0)
    d_pos = cuda.mem_alloc(centers_4d.nbytes)
    cuda.memcpy_htod(d_pos, centers_4d)

    # Camera rays
    ray_orig_host, ray_dir_host = generate_camera_rays(width, height)
    d_ro = cuda.mem_alloc(ray_orig_host.nbytes)
    d_rd = cuda.mem_alloc(ray_dir_host.nbytes)
    cuda.memcpy_htod(d_ro, ray_orig_host)
    cuda.memcpy_htod(d_rd, ray_dir_host)

    # Output image
    image_bytes = width * height * 3
    d_image = cuda.mem_alloc(image_bytes)
    image_host = np.zeros((height, width, 3), dtype=np.uint8)

    # Engine
    engine = UMER_Optics_Context(max_rays=NUM_RAYS,macro_width=macro_width)

    shader_with_defines = (
            f"#define NUM_OBJECTS {num_objects}\n"
            f"#define MAX_MARCH_STEPS {max_march_steps}\n"
            f"#define MARCH_STEP_SIZE {march_step_size}f\n"
            + DYNAMIC_SHADER
    )
    engine.inject_shader(shader_with_defines, "execute_materials")

    writer = imageio.get_writer(output_file, fps=fps)

    for f in range(FRAMES):
        # Optional animation per frame: modify object parameters here
        # e.g., move first object's y coordinate
        if len(objects) > 0:
            # animate sphere y
            scene_data[0][2] = 500.0 + np.sin(f * 0.1) * 2.0   # bump up visibility
        scene_flat = scene_data.reshape(-1)
        cuda.memcpy_htod(d_scene, scene_flat)

        # Update macro grid if centers changed (extract from scene_data)
        centers_anim = scene_data[0::4, 1:4].copy()
        centers_4d[:, :3] = centers_anim
        cuda.memcpy_htod(d_pos, centers_4d)

        cuda.memset_d8(d_image, 25, image_bytes)

        engine.update_optical_topology(d_pos, num_objects)
        engine.trace_rays(d_ro, d_rd, NUM_RAYS, d_image, d_histogram=d_scene)
        cuda.Context.synchronize()

        cuda.memcpy_dtoh(image_host, d_image)
        writer.append_data(image_host)

    writer.close()
    engine.cleanup()
    d_pos.free()
    d_scene.free()
    d_ro.free()
    d_rd.free()
    d_image.free()
    print(f"\n✅ RENDER COMPLETE! Output saved to: {output_file}")