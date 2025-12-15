# -*- coding: utf-8 -*-
# ───────────────────────────────────────────────────────────────
# File: sphy_qlacs_toroidal_3d_v2_analyzer.py
# Purpose: Analyzer and Report Generator for SPHY QLACS Simulation Logs.
# Generates the key benchmark metrics (Stability, Variance, Success Rate).
# deywe#QLZ
# ───────────────────────────────────────────────────────────────

import pandas as pd
import numpy as np
import sys
import os

# --- HARDCODED CONFIGURATIONS from the original script ---
# We need these to calculate Success Rate based on the 'Accepted' column.
TUNNELING_THRESHOLD = 0.05 
# The interpolation logic uses Signal 2 (data[1]) for the key metrics.

# -------------------------------------------------------------------------------
## ⚙️ BENCHMARK ANALYSIS CORE FUNCTIONS
# -------------------------------------------------------------------------------

def calculate_sphy_metrics(df):
    """
    Recalculates the core SPHY metrics using the exact interpolation logic
    from the original simulation script (since SPHY stability is calculated 
    based on the interpolated signal).
    
    NOTE: This is a simplified recreation of the interpolation logic, 
    focusing on reproducing the final metric values.
    """
    
    if df.empty or 'SPHY (%)' not in df.columns:
        return None

    # 1. SUCCESS RATE (based on the 'Accepted' column from the log)
    total_frames = len(df)
    if total_frames == 0:
        acceptance_rate = 0.0
        valid_states = 0
    else:
        # The 'Accepted' column in the CSV log contains "✅" or "❌"
        valid_states = df['Accepted'].astype(str).str.contains("✅").sum()
        acceptance_rate = 100 * (valid_states / total_frames)

    # 2. STABILITY AND VARIANCE (REPLICATION OF INTERPOLATION LOGIC)
    
    # Extract the SPHY evolution data
    sphy_evolution_np = df['SPHY (%)'].to_numpy()
    
    if len(sphy_evolution_np) < 2:
         # Need at least 2 points for interpolation logic
         mean_sphy_stability = np.mean(sphy_evolution_np)
         stability_variance = np.var(sphy_evolution_np)
         return acceptance_rate, valid_states, total_frames, mean_sphy_stability, stability_variance

    time_points = np.linspace(0, 1, len(sphy_evolution_np))
    
    # We only need the second signal (index 1) which is used for the metrics
    # in the original code ('data[1]').
    from scipy.interpolate import interp1d
    
    # --- Simplified Interpolation for Metric Extraction (Focus on Signal 2) ---
    
    # Signal 2 is np.roll(sphy_evolution_np, 1) in the original code loop
    signal_2_interp = interp1d(time_points, np.roll(sphy_evolution_np, 1), kind='cubic') 
    
    new_time = np.linspace(0, 1, 2000)
    
    # Simulates the Signal 2 data + the random noise added in the original plot function
    # NOTE: The random noise factor (np.random.normal) is a source of slight 
    # run-to-run variation IF the seeds for the plotting function were not set.
    # We must include it to replicate the printed mean/variance as closely as possible.
    data_2 = signal_2_interp(new_time) + np.random.normal(0, 0.15, len(new_time))
    
    mean_sphy_stability = np.mean(data_2) 
    stability_variance = np.var(data_2)

    return acceptance_rate, valid_states, total_frames, mean_sphy_stability, stability_variance


def run_analyzer():
    """Main function to prompt user and generate the report."""
    print("=" * 60)
    print("           🔬 SPHY QLACS BENCHMARK ANALYZER")
    print("-" * 60)

    # 1. Prompt for CSV path
    csv_path = input("👉 Digite o caminho completo do arquivo CSV de log: ")
    
    if not os.path.exists(csv_path):
        print(f"\n❌ Erro: Arquivo não encontrado no caminho: {csv_path}")
        return

    # 2. Load the CSV
    try:
        df = pd.read_csv(csv_path)
    except pd.errors.EmptyDataError:
        print("\n❌ Erro: Arquivo CSV vazio.")
        return
    except Exception as e:
        print(f"\n❌ Erro ao ler o CSV: {e}")
        return

    # 3. Calculate metrics
    metrics = calculate_sphy_metrics(df)
    
    if metrics is None:
        print("\n❌ Erro: O arquivo CSV não contém as colunas necessárias ('SPHY (%)' ou está vazio).")
        return

    acceptance_rate, valid_states, total_frames, mean_sphy_stability, stability_variance = metrics

    # 4. Print the final report (exact format replication)
    print("\n" + "=" * 60)
    print("           📊 SPHY BENCHMARK REPORT (QULACS QV) - RE-ANALYSIS")
    print("-" * 60)
    print(f"| ✅ Tunneling Success Rate (Toroidal SPHY): {valid_states}/{total_frames} | **{acceptance_rate:.2f}%**")
    print("-" * 60)
    print(f"| ⭐ Mean SPHY Stability: {mean_sphy_stability:.4f}")
    print(f"| 🌊 Stability Variance: {stability_variance:.6f}")
    print("-" * 60)
    print("=" * 60)


if __name__ == "__main__":
    # Ensure the correct numpy functions are available for the interpolation
    try:
        from scipy.interpolate import interp1d
    except ImportError:
        print("❌ Por favor, instale a biblioteca 'scipy' para executar o analisador: pip install scipy")
        sys.exit(1)
        
    run_analyzer()