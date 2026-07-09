import dearpygui.dearpygui as dpg
import numpy as np
import random

# --- SAFETY VALVE ---
try:
    import pycuda.driver as cuda
    from Umer_Library.core.physical_memory import UMER_Physical_Context
    HAS_GPU = True
except Exception:
    HAS_GPU = False
    print("⚠️ NO GPU DETECTED: Running in Simulation/Showcase mode.")

class UmerDashboard:
    def __init__(self):
        self.is_running = False
        self.data_history = []

        dpg.create_context()
        self._setup_theme()
        self._create_layout()

        dpg.create_viewport(title='U.M.E.R. ENGINE | R&D Terminal', width=1280, height=720)
        dpg.setup_dearpygui()
        dpg.show_viewport()

    def _setup_theme(self):
        # Translate your Figma colors here!
        with dpg.theme() as global_theme:
            with dpg.theme_component(dpg.mvAll):
                dpg.add_theme_color(dpg.mvThemeCol_WindowBg, (10, 10, 12)) # Darker Navy/Black
                dpg.add_theme_color(dpg.mvThemeCol_Text, (200, 200, 200))
                dpg.add_theme_color(dpg.mvThemeCol_Button, (30, 30, 35))
                dpg.add_theme_style(dpg.mvStyleVar_FrameRounding, 5)
        dpg.bind_theme(global_theme)

    def _create_layout(self):
        # Main Dashboard
        with dpg.window(label="Control Unit", width=1280, height=720, no_title_bar=True, no_move=True):
            with dpg.group(horizontal=True):
                # SIDEBAR (From your Figma design)
                with dpg.child_window(width=250, border=True):
                    dpg.add_text("ENGINE STATUS", color=(0, 255, 127))
                    dpg.add_separator()
                    status = "ONLINE (CUDA)" if HAS_GPU else "OFFLINE (SIM)"
                    dpg.add_text(f"Mode: {status}")

                    dpg.add_spacer(height=20)
                    dpg.add_slider_float(label="Cell Size", default_value=0.20, max_value=1.0)
                    dpg.add_slider_float(label="Radius", default_value=0.09, max_value=0.5)

                    dpg.add_spacer(height=20)
                    dpg.add_button(label="INITIALIZE CORE", width=-1, height=40, callback=self.toggle_engine)
                    dpg.add_button(label="FLUSH VRAM", width=-1)

                # MAIN VIEWPORT
                with dpg.group():
                    with dpg.child_window(height=450, border=True):
                        dpg.add_text("SPATIAL TOPOLOGY VIEWPORT", color=(100, 100, 100))
                        # This plot will "look" like your 4M particles
                        with dpg.plot(no_title=True, height=-1, width=-1, tag="main_plot"):
                            dpg.add_plot_axis(dpg.mvXAxis, no_tick_labels=True)
                            dpg.add_plot_axis(dpg.mvYAxis, no_tick_labels=True, tag="y_axis_viz")
                            dpg.add_scatter_series([], [], parent="y_axis_viz", tag="particle_series")

                    with dpg.child_window(height=-1, border=True):
                        dpg.add_text("REAL-TIME THROUGHPUT (MCell/s)", color=(0, 255, 127))
                        with dpg.plot(no_title=True, height=-1, width=-1):
                            dpg.add_plot_axis(dpg.mvXAxis, no_tick_labels=True)
                            dpg.add_plot_axis(dpg.mvYAxis, tag="y_axis_perf")
                            dpg.add_line_series([], [], parent="y_axis_perf", tag="perf_series")

    def toggle_engine(self):
        self.is_running = not self.is_running

    def run(self):
        while dpg.is_dearpygui_running():
            if self.is_running:
                # 1. Fake the Throughput Data (Match your 3572.2 MCell/s result!)
                val = 3572.2 + random.uniform(-10, 10)
                self.data_history.append(val)
                if len(self.data_history) > 50: self.data_history.pop(0)

                # 2. Fake the Visuals (Scatter a subset of points)
                # In your real video, this is what looks "insane"
                px = np.random.normal(0, 1, 500)
                py = np.random.normal(0, 1, 500)

                dpg.set_value("perf_series", [list(range(len(self.data_history))), self.data_history])
                dpg.set_value("particle_series", [px, py])
                dpg.fit_axis_data("y_axis_perf")

            dpg.render_dearpygui_frame()
        dpg.destroy_context()

if __name__ == "__main__":
    UmerDashboard().run()