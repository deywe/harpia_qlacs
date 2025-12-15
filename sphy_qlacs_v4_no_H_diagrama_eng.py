import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
import sys

# Define the output directory
OUTPUT_DIR = "logs_harpia_diagram"

def generate_harpia_3d_diagram(csv_file_path):
    """
    Reads the HARPIA/Qulacs simulation log (Hilbertless) and generates a 
    3D trajectory plot of the symbiotic control.
    
    Axes: X=Frame (Time), Y=Coherence (SPHY %), Z=Boost (Correction)
    Color (4D): Uncertainty (I)
    """
    
    # 1. File Verification and Loading
    if not os.path.exists(csv_file_path):
        print(f"❌ Error: File not found at path: {csv_file_path}", file=sys.stderr)
        return

    try:
        # Tries to load the CSV, accommodating different separators
        df = pd.read_csv(csv_file_path, sep=None, engine='python')
    except Exception as e:
        print(f"❌ Error reading the CSV: {e}", file=sys.stderr)
        return

    # 2. Column Normalization and Verification
    # Normalizes column names to match the expected format (e.g., 'sphy (%)' -> 'sphy%')
    df.columns = df.columns.str.strip().str.lower().str.replace('[^a-z0-9%]', '', regex=True)
    
    # Expected columns in your log:
    COLUMNS = {
        'x': 'frame',
        'y': 'sphy%',
        'z': 'boost',
        'c': 'i' # Color = Uncertainty/Entropy (I)
    }

    data_cols = {}
    
    try:
        for key, col_name in COLUMNS.items():
            if col_name in df.columns:
                data_cols[key] = df[col_name].values
            else:
                raise KeyError(f"Required column '{col_name}' not found.")
                
    except KeyError as e:
        print(f"\n❌ Error: {e}. Ensure the CSV contains all necessary headers.", file=sys.stderr)
        return

    # 3. Create Output Directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 4. Generate the 3D Plot
    
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Use 'I' (Uncertainty/Entropy) to color the points (4th Dimension)
    colors = data_cols['c'] 
    
    # Plot the 3D trajectory (Scatter plot)
    scatter = ax.scatter(
        data_cols['x'], 
        data_cols['y'], 
        data_cols['z'], 
        c=colors, 
        cmap='plasma', # Vibrant color map
        s=5,           # Point size
        alpha=0.8
    )

    # Connect the points with a subtle line to show the trajectory (evolution over time)
    ax.plot(
        data_cols['x'], 
        data_cols['y'], 
        data_cols['z'], 
        color='gray', 
        linewidth=0.5, 
        alpha=0.3
    )

    # 5. Visual Adjustments and Titles
    
    # Color Bar
    cbar = fig.colorbar(scatter, ax=ax, pad=0.1)
    cbar.set_label('Uncertainty/Entropy (I)', rotation=270, labelpad=20)

    # Axis Labels
    ax.set_xlabel("X-Axis: Simulation Frame (Time)", fontsize=12)
    ax.set_ylabel("Y-Axis: SPHY Coherence (%)", fontsize=12)
    ax.set_zlabel("Z-Axis: Boost (Gravitational Correction)", fontsize=12)
    ax.set_title("3D Diagram of HARPIA Symbiotic Control Trajectory", fontsize=16)

    # Adjust view angle for better perspective
    ax.view_init(elev=20, azim=130) 

    plt.tight_layout()
    
    # 6. Save and Show
    file_base_name = os.path.basename(csv_file_path).replace('.csv', '')
    output_filename = os.path.join(OUTPUT_DIR, f"diagram_3d_harpia_{file_base_name}.png")
    plt.savefig(output_filename, dpi=300)
    
    print("-" * 50)
    print(f"✅ 3D plot successfully generated!")
    print(f"💾 File saved to: {output_filename}")
    print("-" * 50)

    plt.show()

# --- EXECUTION ---

if __name__ == "__main__":
    
    # Request the CSV path from the user
    csv_path = input("🔗 Enter the full path of the CSV file (e.g., logs/qghz_RF_hilbertless_60q_log.csv): ")
    
    if csv_path:
        generate_harpia_3d_diagram(csv_path.strip())
    else:
        print("Operation cancelled.")