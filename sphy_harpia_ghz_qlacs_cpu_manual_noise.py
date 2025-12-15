# ───────────────────────────────────────────────────────────────
# File: sphy_harpia_ghz_qlacs_cpu_manual_noise.py
# Purpose: Simulação GHZ + HARPIA (QULACS/CPU) + Ruído Manual
# Author: deywe@QLZ | Versão com ruído manual
# ───────────────────────────────────────────────────────────────

import os
import csv
import sys
import re
import random
import numpy as np
from datetime import datetime
from tqdm import tqdm
import matplotlib.pyplot as plt
import subprocess
import hashlib
from qulacs import QuantumState, QuantumCircuit
from qulacs.gate import H, CNOT, X, Y, Z, Measurement

# 🔧 Set up log directory
LOG_DIR = "logs_harpia_qlacs_cpu_manual_noise"
os.makedirs(LOG_DIR, exist_ok=True)

# 🧠 Parameter input
def get_parameters():
    try:
        num_qubits = int(input("🔢 Number of Qubits in the GHZ circuit: "))
        total_frames = int(input("🔁 Total GHZ states to simulate: "))
        return num_qubits, total_frames
    except ValueError:
        print("❌ Invalid input. Please enter integers.")
        sys.exit(1)

# 🧬 GHZ circuit generation with manual noise
def create_noisy_ghz_circuit(num_qubits, noise_prob):
    circuit = QuantumCircuit(num_qubits)
    
    # Apply Hadamard to the first qubit
    circuit.add_gate(H(0))
    
    # Manually simulate depolarizing noise after the H gate
    apply_depolarizing_noise_manually(circuit, 0, noise_prob)

    # Apply CNOTs to create the GHZ state
    for i in range(1, num_qubits):
        circuit.add_gate(CNOT(0, i))
        # Manually simulate depolarizing noise after the CNOT gate
        apply_depolarizing_noise_manually(circuit, 0, noise_prob)
        apply_depolarizing_noise_manually(circuit, i, noise_prob)
        
    return circuit

def apply_depolarizing_noise_manually(circuit, target_qubit, noise_prob):
    """Applies depolarizing noise on a target qubit manually."""
    if random.random() < noise_prob:
        op = random.choice(['X', 'Y', 'Z'])
        if op == 'X':
            circuit.add_gate(X(target_qubit))
        elif op == 'Y':
            circuit.add_gate(Y(target_qubit))
        else: # op == 'Z'
            circuit.add_gate(Z(target_qubit))
            
# 🧪 Measure the state with Qulacs
def measure(circuit, num_qubits):
    """Executes the circuit and returns the measured bitstring."""
    state = QuantumState(num_qubits)
    state.set_zero_state()
    circuit.update_quantum_state(state)
    
    amplitudes = state.get_vector()
    probabilities = np.abs(amplitudes) ** 2
    
    max_prob_index = np.argmax(probabilities)
    result = format(max_prob_index, f'0{num_qubits}b')

    return result

# ⚙️ Call to external HARPIA binary (Rust): Simbiotic AI which resolves decoherence control.
def calculate_F_opt(H, S, C, I, T):
    try:
        result = subprocess.run(
            ["./sphy_simbiotic_entangle_ai", str(H), str(S), str(C), str(I), str(T)],
            capture_output=True, text=True, check=True
        )
        match = re.search(r"([-+]?\d*\.\d+|\d+)", result.stdout)
        if match:
            return float(match.group(0))
        else:
            raise ValueError(f"❌ Failed to extract output value: {result.stdout}")
    except FileNotFoundError:
        print("❌ Error: The binary 'sphy_simbiotic_entangle_ai' was not found.")
        sys.exit(1)
    except subprocess.CalledProcessError as e:
        print(f"❌ Error executing the binary: {e.stderr}")
        sys.exit(1)

# 🚀 Main simulation loop
def run_simulation(num_qubits, total_frames=100000, noise_prob=0.99):
    print("=" * 60)
    print(f"    🧿 HARPIA QGHZ STABILIZER • {num_qubits} Qubits • {total_frames:,} Frames (Qulacs/CPU)")
    print("=" * 60)

    timecode = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_file = os.path.join(LOG_DIR, f"qghz_{num_qubits}q_log_{timecode}.csv")
    fig_file = os.path.join(LOG_DIR, f"qghz_{num_qubits}q_graph_{timecode}.png")

    sphy_coherence = 90.0
    accepted_frames = 0
    log_data = []
    sphy_evolution = []
    ideal_states = ['0' * num_qubits, '1' * num_qubits]

    for frame in tqdm(range(1, total_frames + 1), desc="⏳ Simulating GHZ"):
        circuit = create_noisy_ghz_circuit(num_qubits, noise_prob)
        result = measure(circuit, num_qubits)

        H = random.uniform(0.95, 1.0)
        S = random.uniform(0.95, 1.0)
        C = sphy_coherence / 100
        I = abs(H - S)
        T = frame

        boost = calculate_F_opt(H, S, C, I, T)
        delta = boost * 0.7
        new_coherence = min(100, sphy_coherence + delta)
        activated = delta > 0
        accepted = (result in ideal_states) and activated
        if accepted:
            accepted_frames += 1

        sphy_coherence = new_coherence
        sphy_evolution.append(sphy_coherence)

        log_line = [
            frame, result,
            round(H, 4), round(S, 4),
            round(C, 4), round(I, 4),
            round(boost, 4), round(sphy_coherence, 4),
            "✅" if accepted else "❌"
        ]

        hash_input = ",".join(map(str, log_line))
        uid_sha256 = hashlib.sha256(hash_input.encode()).hexdigest()

        log_line.append(uid_sha256)
        log_data.append(log_line)

        sys.stdout.flush()

    acceptance_rate = 100 * (accepted_frames / total_frames)
    print(f"\n✅ Accepted GHZ States: {accepted_frames}/{total_frames}|{acceptance_rate:.2f}%")

    with open(csv_file, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([
            "Frame", "Result", "H", "S", "C", "I",
            "Boost", "SPHY (%)", "Accepted", "UID_SHA256"
        ])
        writer.writerows(log_data)
    print(f"🧾 CSV saved: {csv_file}")

    plt.figure(figsize=(12, 5))
    plt.plot(range(1, total_frames + 1), sphy_evolution, color="darkcyan", label="⧉ SPHY Coherence")
    plt.scatter(
        range(1, total_frames + 1), sphy_evolution,
        c=['green' if row[-2] == "✅" else 'red' for row in log_data],
        s=8, alpha=0.6
    )
    plt.axhline(90, color='gray', linestyle="dotted", linewidth=1, label="Threshold")
    plt.title(f"📡 HARPIA SPHY Evolution • {num_qubits} Qubits • {total_frames:,} Frames (Qulacs/CPU)")
    plt.xlabel("Frames")
    plt.ylabel("SPHY Coherence (%)")
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig_file, dpi=300)
    print(f"📊 Graph saved: {fig_file}")
    plt.show()

# Entry point
if __name__ == "__main__":
    qubits, frames = get_parameters()
    run_simulation(num_qubits=qubits, total_frames=frames, noise_prob=1.0)