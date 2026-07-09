# Umer_Library/telemetry/autopsy.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

def export_trajectory_csv(trajectory_data, total_dims, filename="umer_kinematic_trajectory.csv"):
    """Exports flight paths for 3D simulation engines like PyBullet or MuJoCo."""
    df = pd.DataFrame(trajectory_data, columns=[f"Joint_{i}" for i in range(total_dims)])
    df.to_csv(filename, index_label="Step")
    print(f"✅ Exported Trajectory: '{filename}'")

def plot_kinematic_pca(obstacles, trajectory, target, n_obstacles_to_plot=5000):
    """Crushes 30D space into a 2D Visual Map to prove obstacle avoidance."""
    pca = PCA(n_components=2)
    
    # Fit PCA on a safe subset to prevent RAM blowouts
    safe_obstacles = obstacles[:n_obstacles_to_plot]
    all_points = np.vstack([safe_obstacles, trajectory, target])
    pca_result = pca.fit_transform(all_points)
    
    pca_obstacles = pca_result[:len(safe_obstacles)]
    pca_trajectory = pca_result[len(safe_obstacles):len(safe_obstacles)+len(trajectory)]
    pca_target = pca_result[-1]

    plt.figure(figsize=(12, 8))
    plt.scatter(pca_obstacles[:, 0], pca_obstacles[:, 1], c='red', s=2, alpha=0.1, label='Obstacles (-Mass)')
    plt.plot(pca_trajectory[:, 0], pca_trajectory[:, 1], c='blue', marker='.', linewidth=2, label='U.M.E.R. Trajectory')
    plt.scatter(pca_target[0], pca_target[1], c='green', marker='*', s=200, edgecolor='black', label='Target (+Mass)')
    
    plt.title("Engine Autopsy: PCA Projection of High-Dimensional Threading")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.legend()
    plt.savefig("autopsy_pca_projection.png", dpi=300)
    print("✅ Exported Projection: 'autopsy_pca_projection.png'")

def diagnose_failed_cases(preds, y_true, m_h, total_dims, f_list, m0_matrix, m1_matrix, feature_names=None):
    """The Unrestricted Autopsy for your Spatial Intelligence module."""
    print("\n=======================================================")
    print("   ENGINE AUTOPSY: THE FAILED CASES")
    print("=======================================================")
    
    fp_idx = np.where((preds == 1) & (y_true == 0))[0] # Pred Benign, Actual Malignant
    fn_idx = np.where((preds == 0) & (y_true == 1))[0] # Pred Malignant, Actual Benign
    
    print(f"THE FINAL X-RAY: {len(fp_idx) + len(fn_idx)} TOTAL PATIENTS MISCLASSIFIED\n")
    
    def print_autopsy_block(indices, title):
        if len(indices) == 0: return
        print(f"--- {title} ({len(indices)} cases) ---")
        for idx in indices[:5]: # Limit to first 5 so we don't spam the console
            print(f"\nPatient {idx} | Total Momentum: {m_h[idx]:.3f}")
            for t in range(min(3, m0_matrix.shape[1])): # Show first 3 grids
                feats = [feature_names[f] if feature_names else f"Dim_{f}" for f in f_list[t*2:t*2+4]]
                print(f"  Grid {t} ({feats[0]}, {feats[1]}...)")
                print(f"     > Malignant Gravity: {m0_matrix[idx, t]:.3f}")
                print(f"     > Benign Gravity:    {m1_matrix[idx, t]:.3f}")

    print_autopsy_block(fp_idx, "FALSE POSITIVES (Predicted Benign, Actually Malignant)")
    print_autopsy_block(fn_idx, "FALSE NEGATIVES (Predicted Malignant, Actually Benign)")


def export_dynamic_sandbox(custom_update_logic, num_particles=10000, theme_color="#00FF7F",
                           filename="umer_sandbox.html"):
    """
    Generates a standalone HTML5 WebGL particle viewer.
    Dynamically injects user-defined JavaScript physics logic into the render loop.
    """
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>U.M.E.R. Native Sandbox</title>
        <style>
            body {{ margin: 0; overflow: hidden; background-color: #050505; color: {theme_color}; font-family: monospace; }}
            canvas {{ display: block; }}
            #hud {{ position: absolute; top: 10px; left: 10px; z-index: 10; pointer-events: none; }}
        </style>
    </head>
    <body>
        <div id="hud">
            <h2>U.M.E.R. DYNAMIC SANDBOX</h2>
            <p>Target: WebGL Browser Engine</p>
            <p id="fps">FPS: 0</p>
            <p>Agents: {num_particles:,}</p>
        </div>
        <canvas id="simCanvas"></canvas>
        <script>
            const canvas = document.getElementById('simCanvas');
            const ctx = canvas.getContext('2d');
            canvas.width = window.innerWidth; canvas.height = window.innerHeight;

            // Engine tracks mouse coordinates automatically for interactive logic
            let mouseX = canvas.width / 2;
            let mouseY = canvas.height / 2;
            window.addEventListener('mousemove', (e) => {{ mouseX = e.clientX; mouseY = e.clientY; }});

            const particles = [];
            for(let i=0; i<{num_particles}; i++) {{
                particles.push({{x: Math.random()*canvas.width, y: Math.random()*canvas.height, vx: 0, vy: 0}});
            }}

            let lastTime = performance.now();

            function loop() {{
                let now = performance.now();
                document.getElementById('fps').innerText = 'FPS: ' + Math.round(1000/(now-lastTime));
                lastTime = now;

                // Trailing effect
                ctx.fillStyle = 'rgba(0, 0, 0, 0.2)'; 
                ctx.fillRect(0, 0, canvas.width, canvas.height);

                ctx.fillStyle = '{theme_color}';
                for(let p of particles) {{

                    // ==========================================
                    // U.M.E.R. DYNAMIC LOGIC INJECTION
                    // ==========================================
                    {custom_update_logic}
                    // ==========================================

                    p.x += p.vx; 
                    p.y += p.vy;

                    // Default Engine Behavior: Screen Wrapping
                    if(p.x < 0) p.x = canvas.width;
                    if(p.x > canvas.width) p.x = 0;
                    if(p.y < 0) p.y = canvas.height;
                    if(p.y > canvas.height) p.y = 0;

                    ctx.fillRect(p.x, p.y, 1.5, 1.5);
                }}
                requestAnimationFrame(loop);
            }}
            loop();
        </script>
    </body>
    </html>
    """

    with open(filename, "w") as f:
        f.write(html_content)
    print(f"✅ Exported Dynamic Native Visualizer: '{filename}'")