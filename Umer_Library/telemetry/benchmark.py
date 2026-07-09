# Umer_Library/telemetry/benchmark.py

import time
import threading
import numpy as np
import pycuda.driver as cuda

from Umer_Library.core.physical_memory import UMER_Physical_Context

try:
    import pynvml

    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    NVML_OK = True
except Exception:
    NVML_OK = False


class PowerSampler:
    """Samples NVML power every 5ms in a background thread."""

    def __init__(self):
        self.samples = []
        self._stop = False
        self._thread = None

    def start(self):
        self.samples = []
        self._stop = False
        self._thread = threading.Thread(target=self._sample, daemon=True)
        self._thread.start()

    def _sample(self):
        while not self._stop:
            if NVML_OK:
                mw = pynvml.nvmlDeviceGetPowerUsage(handle)
                self.samples.append(mw / 1000.0)
            time.sleep(0.005)

    def stop(self):
        self._stop = True
        if self._thread:
            self._thread.join(timeout=1.0)

    def mean_watts(self):
        if not self.samples:
            return 0.0
        trim = max(1, len(self.samples) // 10)
        return float(np.mean(self.samples[trim:-trim] or self.samples))


def print_efficiency_report(label, ms_per_frame, power_w, throughput, edp, movers_pct):
    """Standardized output for your IEEE paper formatting."""
    print(f"{label}  [{movers_pct:.1f}% movers]:")
    print(f"  Throughput : {throughput:.1f} MCell/s")
    print(f"  Power      : {power_w:.1f} W")
    if power_w > 0:
        print(f"  Efficiency : {(throughput / power_w):.1f} MCell/W")
    print(f"  EDP        : {edp:.6f} J·s")
    print("-" * 50)


# Dynamic payload for the benchmark
BENCHMARK_MOTION = """
__global__ void apply_motion(float4* __restrict__ pos4, const float4* __restrict__ vel4, float dt, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    float4 p = pos4[idx]; float4 v = vel4[idx];
    p.x = fmodf(p.x + v.x * dt + DOMAIN, DOMAIN);
    p.y = fmodf(p.y + v.y * dt + DOMAIN, DOMAIN);
    p.z = fmodf(p.z + v.z * dt + DOMAIN, DOMAIN);
    pos4[idx] = p;
}
"""


def run_suite(n_particles=4_000_000, domain=1000.0, dt=0.1, iters=200):
    print("=========================================================")
    print("   U.M.E.R. ECOLOGICAL EFFICIENCY BENCHMARK")
    print("=========================================================")

    pos_host = np.zeros((n_particles, 4), dtype=np.float32)
    pos_host[:, 0:3] = np.random.rand(n_particles, 3).astype(np.float32) * domain

    d_pos_in = cuda.mem_alloc(pos_host.nbytes)
    d_vel_in = cuda.mem_alloc(pos_host.nbytes)
    d_pos_out = cuda.mem_alloc(pos_host.nbytes)
    d_vel_out = cuda.mem_alloc(pos_host.nbytes)

    engine = UMER_Physical_Context(n_particles=n_particles)
    engine.inject_physics(BENCHMARK_MOTION, "apply_motion")

    sampler = PowerSampler()

    VEL_SCALES = [
        (0.5, "UMER TC vel=0.5 (dense SPH)"),
        (2.0, "UMER TC vel=2.0 (Boids/cloth)"),
        (5.0, "UMER TC vel=5.0 (fast gas)"),
        (15.0, "UMER TC vel=15 (explosion)"),
    ]

    for vel_scale, label in VEL_SCALES:
        vel_host = np.zeros((n_particles, 4), dtype=np.float32)
        vel_host[:, 0:3] = (np.random.rand(n_particles, 3).astype(np.float32) - 0.5) * vel_scale

        cuda.memcpy_htod(d_pos_in, pos_host)
        cuda.memcpy_htod(d_vel_in, vel_host)

        engine.cold_start(d_pos_in, d_vel_in, d_pos_out, d_vel_out)
        cuda.Context.synchronize()

        # Warmup
        for _ in range(10):
            engine.k_dynamic_motion(d_pos_in, d_vel_in, np.float32(dt), np.int32(n_particles), block=engine.BLOCK,
                                    grid=engine.PGRID)
            engine.temporal_update(d_pos_in, d_vel_in, d_pos_out, d_vel_out)
        cuda.Context.synchronize()

        sampler.start()
        t0 = time.perf_counter()

        for _ in range(iters):
            engine.k_dynamic_motion(d_pos_in, d_vel_in, np.float32(dt), np.int32(n_particles), block=engine.BLOCK,
                                    grid=engine.PGRID)
            engine.temporal_update(d_pos_in, d_vel_in, d_pos_out, d_vel_out)

        cuda.Context.synchronize()
        elapsed = time.perf_counter() - t0
        sampler.stop()

        ms_per_frame = (elapsed / iters) * 1000
        power_w = sampler.mean_watts()
        throughput = n_particles / (ms_per_frame / 1000) / 1e6
        edp = power_w * ((ms_per_frame / 1000) ** 2)

        # Get movers percentage
        mc = np.zeros(1, dtype=np.int32)
        cuda.memcpy_dtoh(mc, engine.d_mover_cnt)
        movers_pct = (mc[0] / n_particles) * 100

        print_efficiency_report(label, ms_per_frame, power_w, throughput, edp, movers_pct)

    engine.cleanup()
    d_pos_in.free();
    d_vel_in.free();
    d_pos_out.free();
    d_vel_out.free()


if __name__ == "__main__":
    run_suite()