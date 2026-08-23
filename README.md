U.M.E.R (Uniform Memory Encoded Representation) — a breakdown of what's actually in the library right now, file by file.

core/primitives.py Defines the base CUDA kernels: a 3-pass cascade (histogram → prefix sum → scatter) that sorts points into a hash grid, plus a separate kernel (CUDA_PGP) that scores every pair of dimensions by how well they separate positive/negative mass, used for feature selection.

core/memory.py — UMER_Context Wraps those kernels into a usable class. optimize_topology() runs the pairwise scoring and picks the best dimension ordering across N-dimensional data, building a set of overlapping hash-grid "trees." build_hash_grid() runs the actual sort. inject_logic() takes a CUDA string, JIT-compiles it, and attaches it as a callable kernel that has access to the grid's internal buffers.

core/primitives_3d.py + core/physical_memory.py — UMER_Physical_Context A 3D-specific version of the same sorting idea, but built for particles that move every frame. cold_start() does a full sort. temporal_update() only re-sorts particles whose hash bucket actually changed since the last frame (tracked via a prev_hash buffer and an is_mover flag) — everything else keeps its old bucket. inject_physics() JIT-compiles a motion kernel (e.g. gravity, boids) that runs before each sort.

core/neural_memory.py — UMER_Neural_Context Same delta-update idea, split explicitly into static and dynamic populations. initialize_static_geometry() hashes the non-moving majority of a scene once. amortized_dynamic_update() clears and re-inserts only the moving subset. inject_neural_evaluator() compiles a ray-marching/splatting shader against that hash table.

core/optics_memory.py — UMER_Optics_Context Two-pass ray tracing. Pass 1 (pass1_macro_dda_compaction) walks a coarse "macro-grid" using 3D DDA stepping and compacts only the rays that actually hit something into a queue (stream compaction) — rays through empty space are dropped early. Pass 2 runs a user-injected shader (inject_shader()) only on the compacted, active rays.

modules/kinematics.py run_kinematic_driver() — a 30-dimensional path-planning demo. 20,000 obstacle points are given negative mass, a target point positive mass. A JIT-compiled kernel computes a repulsion/attraction vector at each step by querying the hash grid, and the "robot" position is updated along that combined vector for up to 120 steps.

modules/physics.py run_temporal_simulation() — spins up 4,000,000 particles with random velocities, applies a JIT-injected motion kernel each frame, re-sorts using temporal_update(), and prints average frame time and throughput in MCell/s.

modules/rendering.py Signed-distance-field ray marcher supporting five primitive types (sphere, torus, box, cylinder, mandelbulb). build_scene_from_objects() and default_scene_objects() construct scene data; generate_camera_rays() builds a pinhole camera's ray origins/directions; run_rendering_pipeline() runs the two-pass optics engine per frame and writes an MP4 via imageio.

modules/neural_render.py Same two-pass idea applied to a splatting shader instead of raw SDFs — designed for scenes with a large static population and a smaller dynamic one, using UMER_Neural_Context's amortized updates.

modules/perception.py Not GPU code — a classical CV feature extractor. abstract_image() takes a BGR image, crops 10% border, applies CLAHE, splits it into a 3×3 grid, and pulls 12 features per cell (mean, std, skew, kurtosis, Sobel edge density/variance, GLCM contrast/homogeneity/energy/correlation, max edge, median) plus 3 global contour features (circularity, solidity, aspect ratio) — 111 features total. get_feature_names() returns matching labels for downstream analysis.

modules/spatial_ml.py run_abstract_engine() — binary classification using the same hash-grid cascade. Training labels are converted to signed mass (-1 / +1). At inference, each test point sums Gaussian-weighted mass from nearby training points per tree, weighted by that tree's learned importance; the sign of the total determines the predicted class. Reports accuracy via sklearn.

telemetry/autopsy.py export_trajectory_csv() dumps a path-planning trajectory to CSV. plot_kinematic_pca() projects high-D obstacles/trajectory/target into 2D via PCA and saves a PNG. diagnose_failed_cases() prints per-tree mass breakdowns for misclassified test points. export_dynamic_sandbox() generates a standalone HTML/JS particle demo with injectable JS physics.

telemetry/benchmark.py run_suite() runs the physics engine at four velocity presets, samples GPU power draw via NVML in a background thread (PowerSampler), and reports throughput (MCell/s), power (W), efficiency (MCell/W), energy-delay product, and percentage of particles that moved per frame.

telemetry/sweeps.py run_auto_tuner() takes an arbitrary labeled dataset, runs the same PGP feature-selection + cascade build, then grid-searches over search radius and mass-decay rate to find the combination with highest test accuracy. Outputs an interactive 3D Plotly surface of accuracy vs. hyperparameters.

umer_dashboard.py A DearPyGui desktop app scaffold — sidebar with sliders (cell size, radius) and two live plots (particle scatter, throughput line). Currently wired to placeholder/random values, not the live engine — it's a UI shell built ahead of the backend hookup.

umer_engine-2.0.1-cp312-cp312-linux_x86_64.whl A prebuilt Python wheel of the compiled engine for Python 3.12 on Linux x86_64.

library_documentation.docx Full API reference covering all classes and functions above.

That's the whole repo as it stands — one commit, ~1,900 lines of Python/CUDA. It's not a finished product; it's a working prototype of one core mechanism (delta-updated spatial hashing with JIT-injectable logic) applied across five different problem types.
