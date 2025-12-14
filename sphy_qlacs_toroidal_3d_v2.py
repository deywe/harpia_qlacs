# -*- coding: utf-8 -*-
# ───────────────────────────────────────────────────────────────
# File: sphy_qlacs_toroidal_3d_v2.py
# Purpose: QUANTUM TUNNELING IN A TOROIDAL LATTICE (QV Equivalent) + 
# SPHY FIELD ENGINEERING 
# deywe#QLZ
# ───────────────────────────────────────────────────────────────

# 1️⃣ Import necessary modules
from meissner_core import meissner_correction_step 

# ⚛️ Qulacs & QV Imports
from qulacs import QuantumCircuit, DensityMatrix
from qulacs.gate import H, CNOT, X, Y, Z, RZ
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from multiprocessing import Pool, Manager
from datetime import datetime
import os, random, sys, hashlib, csv
from scipy.interpolate import interp1d

# === SPHY Toroidal Lattice Configuration (QV Equivalent) ===
GRID_SIZE = 2 
NUM_QUBITS = GRID_SIZE * GRID_SIZE # 4 Qubits

# Target qubit
TARGET_QUBIT = 0 

TUNNELING_THRESHOLD = 0.05     

# === Log Directory
LOG_DIR = "logs_sphy_toroidal_qlacs_v2"
os.makedirs(LOG_DIR, exist_ok=True)

INITIAL_COHERENCE = 90.0

# -------------------------------------------------------------------------------
## ⚙️ SETUP AND HELPER FUNCTIONS
# -------------------------------------------------------------------------------

def get_user_parameters():
    """Retrieves user parameters for the simulation."""
    try:
        num_qubits = NUM_QUBITS
        print(f"🔢 Number of Qubits (Lattice {GRID_SIZE}x{GRID_SIZE}): {num_qubits}")
        total_pairs = int(input("🔁 Total Frames to simulate: ")) 
        
        barrier_strength_input = float(input("🚧 Barrier Strength (0.0 to 1.0): "))
        if not (0.0 <= barrier_strength_input <= 1.0):
             print("❌ Barrier Strength must be between 0.0 and 1.0.")
             exit(1)
             
        # Converts strength to RZ rotation angle (0 to pi/2)
        barrier_strength_theta = barrier_strength_input * np.pi / 2 
        
        return num_qubits, total_pairs, barrier_strength_theta
    except ValueError:
        print("❌ Invalid input. Please enter integers/floats.")
        exit(1)

# -------------------------------------------------------------------------------
## ⚛️ QULACS FUNCTIONS AND QV MODELING
# -------------------------------------------------------------------------------

def apply_depolarizing_noise_manually(circuit, target_qubit, noise_prob):
    """ Adds probabilistic X, Y, or Z error (Noise Simulation) """
    if random.random() < noise_prob:
        op = random.choice(['X', 'Y', 'Z'])
        if op == 'X':
            circuit.add_gate(X(target_qubit))
        elif op == 'Y':
            circuit.add_gate(Y(target_qubit))
        else:
            circuit.add_gate(Z(target_qubit))

def toroidal_tunneling_program_qlacs(num_qubits, barrier_theta, sphy_perturbation_angle, noise_prob):
    """
    Creates the QV circuit simulating toroidal connections and tunneling.
    """
    circuit = QuantumCircuit(num_qubits)
    
    # State Preparation (Entanglement)
    circuit.add_gate(H(0))
    apply_depolarizing_noise_manually(circuit, 0, noise_prob)
    
    # Toroidal Coupling (Connections: 0->1, 1->3, 3->2, 2->0)
    connections = [(0, 1), (1, 3), (3, 2), (2, 0)]
    
    for c1, c2 in connections:
        circuit.add_gate(CNOT(c1, c2))
        apply_depolarizing_noise_manually(circuit, c1, noise_prob)
        apply_depolarizing_noise_manually(circuit, c2, noise_prob)

    # Barrier (RZ gate on target qubit)
    circuit.add_gate(RZ(TARGET_QUBIT, barrier_theta)) 
    
    # SPHY Field Perturbation
    circuit.add_gate(RZ(TARGET_QUBIT, sphy_perturbation_angle))
    
    # Perturbation to adjacent qubits
    for qubit in [1, 2, 3]:
        circuit.add_gate(RZ(qubit, sphy_perturbation_angle / 2))
             
    return circuit

def simulate_frame_qlacs(frame_data):
    frame, num_qubits, total_frames, noise_prob, sphy_coherence, barrier_theta = frame_data
    
    random.seed(os.getpid() * frame) 
    
    sphy_perturbation_angle = 0.0
    if random.random() < noise_prob:
        sphy_perturbation_angle = random.uniform(-np.pi/8, np.pi/8)
    
    program = toroidal_tunneling_program_qlacs(num_qubits, barrier_theta, sphy_perturbation_angle, noise_prob)
    
    try:
        initial_state = DensityMatrix(num_qubits)
        program.update_quantum_state(initial_state)
        state_dm = initial_state
        
    except Exception as e:
        return None, None, None, f"\nCritical Error running Qulacs Engine on frame {frame}: {e}"

    # --- EXPECTATION VALUE & STATE DATA COLLECTION ---
    
    # 1. TUNNELING PROXY (Probability of P(|1>) of the target qubit)
    measured_values = [2] * num_qubits
    measured_values[TARGET_QUBIT] = 1 # Define 1 for qubit 0 (target)
    
    prob_1 = state_dm.get_marginal_probability(measured_values) 
    
    # Tunneling Proxy: Deviation from the base probability (0.5)
    proxy_mag = abs(prob_1 - 0.5) 

    result_raw = 1 if proxy_mag > TUNNELING_THRESHOLD else 0
    ideal_state = 1

    # 2. STATE DATA FOR PURITY (REMOVED)
    # purity = state_dm.get_squared_norm() # Purity line removed

    # === SPHY/Meissner Logic ===
    H = random.uniform(0.95, 1.0) 
    S = random.uniform(0.95, 1.0) 
    C = sphy_coherence / 100    
    I = abs(H - S)             
    T = frame                   
    psi_state = [3.0, 3.0, 1.2, 1.2, 0.6, 0.5] 

    try:
        boost, phase_impact, psi_state = meissner_correction_step(H, S, C, I, T, psi_state)
    except Exception as e:
        return None, None, None, f"\nCritical Error on frame {frame} (AI Meissner): {e}"

    delta = boost * 0.7
    new_coherence = min(100, sphy_coherence + delta)
    activated = delta > 0 
    accepted = (result_raw == ideal_state) and activated
    
    # Log entry
    current_timestamp = datetime.utcnow().isoformat()
    data_to_hash = f"{frame}:{result_raw}:{H:.4f}:{S:.4f}:{C:.4f}:{I:.4f}:{boost:.4f}:{new_coherence:.4f}:{current_timestamp}"
    sha256_signature = hashlib.sha256(data_to_hash.encode('utf-8')).hexdigest()

    # adjusted state_data_log (Purity removed)
    state_data_log = (prob_1,)

    # adjusted log_entry (Purity removed)
    log_entry = [
        frame, result_raw,
        round(prob_1, 4), 
        round(proxy_mag, 4),
        # round(purity, 6), # Purity removed
        round(H, 4), round(S, 4), round(C, 4), round(I, 4), round(boost, 4),
        round(new_coherence, 4), "✅" if accepted else "❌",
        sha256_signature, current_timestamp
    ]
    return log_entry, new_coherence, state_data_log, None

# -------------------------------------------------------------------------------
## 🖼️ PLOTTING FUNCTIONS
# -------------------------------------------------------------------------------

# The plot_qubit_purity_histogram function was REMOVED

def plot_tunneling_histogram(df, threshold, fig_filename_hist):
    """ Plots the histogram of the Tunneling Proxy magnitude (Proxy_Mag) """
    if df.empty:
        print("❌ Error: Empty DataFrame for Histogram plotting.")
        return

    proxy_data = df['Proxy_Mag']
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.hist(proxy_data, bins=30, edgecolor='black', alpha=0.7, color='skyblue', 
            label='Tunneling Proxy Magnitude ( $|\Delta P(|1\\rangle)| $)')
    
    ax.axvline(threshold, color='red', linestyle='--', linewidth=2, 
               label=f'Tunneling Threshold ({threshold})')
    
    success_count = (proxy_data >= threshold).sum()
    total_count = len(proxy_data)
    success_rate = 100 * (success_count / total_count)
    
    ax.text(0.95, 0.90, f'Total Success: {success_rate:.2f}%', 
            transform=ax.transAxes, 
            bbox=dict(boxstyle="round,pad=0.5", fc="white", alpha=0.6),
            horizontalalignment='right', fontsize=12, color='darkgreen', weight='bold')

    ax.set_title(f'Performance Distribution (Tunneling Proxy) over {total_count} Frames', fontsize=14)
    ax.set_xlabel('Tunneling Proxy Magnitude ( $|\Delta P(|1\\rangle)| $)')
    ax.set_ylabel('Frequency of Occurrence (Frames)')
    ax.legend()
    ax.grid(axis='y', alpha=0.5)
    
    plt.savefig(fig_filename_hist, dpi=300)
    print(f"🖼️ Tunneling Histogram saved: {fig_filename_hist}")


# -------------------------------------------------------------------------------
## 🚀 MAIN EXECUTION FUNCTION
# -------------------------------------------------------------------------------

def execute_simulation_multiprocessing_qlacs(num_qubits, total_frames, barrier_theta, noise_prob=1.0, num_processes=4):
    print("=" * 60)
    print(f" ⚛️ SPHY WAVES (QULACS QV): Toroidal Tunneling ({GRID_SIZE}x{GRID_SIZE}) • {total_frames:,} Frames")
    print(f" 🚧 Barrier Strength: {barrier_theta*180/np.pi:.2f} degrees RZ (QV Analog)")
    print("=" * 60)

    timecode = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_filename = os.path.join(LOG_DIR, f"toroidal_{num_qubits}q_log_{timecode}.csv")
    fig_filename = os.path.join(LOG_DIR, f"toroidal_{num_qubits}q_graph_2D_{timecode}.png")
    # fig_filename_purity_hist removed
    fig_filename_hist = os.path.join(LOG_DIR, f"toroidal_{num_qubits}q_HISTOGRAM_{timecode}.png")

    manager = Manager()
    sphy_coherence = manager.Value('f', INITIAL_COHERENCE)
    log_data, sphy_evolution = manager.list(), manager.list()
    valid_states = manager.Value('i', 0)
    # last_purity removed
    
    frame_inputs = [
        (f, num_qubits, total_frames, noise_prob, sphy_coherence.value, barrier_theta) 
        for f in range(1, total_frames + 1)
    ]

    print(f"🔄 Using {num_processes} processes for simulation...")
    with Pool(processes=num_processes) as pool:
        for log_entry, new_coherence, state_data_log, error in tqdm(pool.imap_unordered(simulate_frame_qlacs, frame_inputs),
                                            total=total_frames, desc="⏳ Simulating Toroidal SPHY (QULACS)"):
            if error:
                print(f"\n{error}", file=sys.stderr)
                pool.terminate()
                break
            if log_entry:
                log_data.append(log_entry)
                sphy_evolution.append(new_coherence)
                sphy_coherence.value = new_coherence 
                if log_entry[-3] == "✅":
                    valid_states.value += 1
                
                # last_purity is no longer updated/used

    # --- Metric Calculation and Report ---
    acceptance_rate = 100 * (valid_states.value / total_frames) if total_frames > 0 else 0
    sphy_evolution_list = list(sphy_evolution)
    
    if not sphy_evolution_list:
        print("❌ Simulation aborted or no data to calculate metrics.")
        return

    sphy_evolution_np = np.array(sphy_evolution_list)
    time_points = np.linspace(0, 1, len(sphy_evolution_np))
    
    # Interpolation Logic (unchanged)
    n_redundancies = 2 
    signals = [interp1d(time_points, np.roll(sphy_evolution_np, i), kind='cubic') for i in range(n_redundancies)]
    new_time = np.linspace(0, 1, 2000)
    
    data = [signal(new_time) + np.random.normal(0, 0.15, len(new_time)) for signal in signals]
    weights = np.linspace(1, 1.5, n_redundancies)
    tunneling_stability = np.average(data, axis=0, weights=weights) 
    
    mean_sphy_stability = np.mean(data[1]) 
    stability_variance = np.var(data[1])

    # 3. Print Metrics to Console (Complete Report)
    print("\n" + "=" * 60)
    print("           📊 SPHY BENCHMARK REPORT (QULACS QV)")
    print("-" * 60)
    print(f"| ✅ Tunneling Success Rate (Toroidal SPHY): {valid_states.value}/{total_frames} | **{acceptance_rate:.2f}%**")
    print("-" * 60)
    print(f"| ⭐ Mean SPHY Stability: {mean_sphy_stability:.4f}")
    print(f"| 🌊 Stability Variance: {stability_variance:.6f}")
    print("-" * 60)
    # Purity line removed
    print("=" * 60)
    
    # --- CSV Writing ---
    # "Purity" column removed from header
    header = [
        "Frame", "Result", 
        "Prob_1", 
        "Proxy_Mag",
        "H", "S", "C", "I", "Boost", "SPHY (%)", "Accepted", 
        "SHA256_Signature", "Timestamp"
    ]
    with open(csv_filename, mode="w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(header)
        writer.writerows(list(log_data))
    print(f"🧾 CSV saved: {csv_filename}")

    # --- Plotting Graphs ---
    df_results = pd.DataFrame(list(log_data), columns=header)
    
    # plot_qubit_purity_histogram removed
    plot_tunneling_histogram(df_results, TUNNELING_THRESHOLD, fig_filename_hist)

    # === 2D STABILITY PLOTTING CODE ===
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12))
    
    ax1.set_title("SPHY Coherence Evolution (Signal 1: Amplitude)")
    for i in range(n_redundancies): 
        ax1.plot(new_time, data[i], alpha=0.3, color='blue')  
    ax1.plot(new_time, tunneling_stability, 'k--', linewidth=2, label="Weighted Average Stability")
    ax1.set_xlabel("Normalized Time")
    ax1.set_ylabel("Coherence/Amplitude")
    ax1.legend()
    ax1.grid()

    ax2.set_title("SPHY Coherence Evolution (Signal 2: Stability)")
    ax2.plot(new_time, data[1], color='red', alpha=0.7, label='Coherence Signal (2)')
    
    ax2.axhline(mean_sphy_stability, color='green', linestyle='--', label=f"Mean: {mean_sphy_stability:.2f}")
    ax2.axhline(mean_sphy_stability + np.sqrt(stability_variance), color='orange', linestyle='--', label=f"± Variance")
    ax2.axhline(mean_sphy_stability - np.sqrt(stability_variance), color='orange', linestyle='--')

    ax2.set_xlabel("Normalized Time")
    ax2.set_ylabel("Coherence/Amplitude")
    ax2.legend()
    ax2.grid()

    fig.suptitle(f"Quantum Tunneling Simulation (Qulacs QV): {total_frames} Attempts (Toroidal SPHY)", fontsize=16)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    plt.savefig(fig_filename, dpi=300)
    print(f"🖼️ 2D Stability Graph saved: {fig_filename}")
    plt.show(block=True) 


if __name__ == "__main__":
    qubits, pairs, barrier_theta = get_user_parameters()
    
    execute_simulation_multiprocessing_qlacs(num_qubits=qubits, total_frames=pairs, barrier_theta=barrier_theta, noise_prob=1.0, num_processes=4)