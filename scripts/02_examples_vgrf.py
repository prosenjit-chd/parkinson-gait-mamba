import pandas as pd
import matplotlib.pyplot as plt
import os

# === CONFIG ===
CONTROL_FILE = "data/raw/GaCo01_01.txt"  # Healthy control example
PATIENT_FILE = "data/raw/GaPt03_01.txt"  # PD patient example
SAMPLE_RATE = 100  # Hz
ZOOM_END = 20  # seconds to visualize


def load_vgrf_signal(filepath):
    """Load a PhysioNet .txt gait file, auto-handle column count."""
    df = pd.read_csv(filepath, sep="\t", header=None)
    n_cols = df.shape[1]

    print(f"[LOADED] {os.path.basename(filepath)} with {n_cols} columns")

    # Split evenly between left/right
    half = n_cols // 2
    left_cols = df.iloc[:, :half]
    right_cols = df.iloc[:, half:]

    df["L_total"] = left_cols.sum(axis=1)
    df["R_total"] = right_cols.sum(axis=1)
    df["Time"] = df.index / SAMPLE_RATE
    return df


def plot_zoomed_vgrf(df, title, save_path):
    """Plot zoomed-in VGRF for left and right totals."""
    df_zoom = df[df["Time"] <= ZOOM_END]

    plt.figure(figsize=(8, 4))
    plt.plot(df_zoom["Time"], df_zoom["L_total"], label="Left foot", color="#1f77b4")
    plt.plot(df_zoom["Time"], df_zoom["R_total"], label="Right foot", color="#ff7f0e")
    plt.title(title)
    plt.xlabel("Time (s)")
    plt.ylabel("Force (N)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"[OK] Saved: {save_path}")


# === LOAD & PLOT ===
control_df = load_vgrf_signal(CONTROL_FILE)
patient_df = load_vgrf_signal(PATIENT_FILE)

plot_zoomed_vgrf(
    control_df,
    "Zoomed VGRF (0–20 s) — Control Subject",
    "outputs/figures/zoomed_vgrf_control.png",
)
plot_zoomed_vgrf(
    patient_df,
    "Zoomed VGRF (0–20 s) — PD Subject",
    "outputs/figures/zoomed_vgrf_patient.png",
)

print("[DONE] Zoomed plots generated for clearer visualization.")
