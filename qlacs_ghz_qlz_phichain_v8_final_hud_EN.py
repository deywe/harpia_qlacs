# File: qlacs_ghz_qlz_phichain_v8_final_hud_EN.py
# Purpose: GHZ + HARPIA (QULACS/CPU) Live Monitor (Final Static HUD)
# Author: deywe@QLZ (Adjusted by Gemini)

import warnings
warnings.filterwarnings("ignore")

import sys
import re
import random
import numpy as np
import time 
import subprocess
import hashlib
import os 
from qulacs import QuantumState, QuantumCircuit
from qulacs.gate import H, CNOT, X, Y, Z

# -------------------------------------------------------------
#                   AUXILIARY AND INPUT FUNCTIONS
# -------------------------------------------------------------

def input_parameters():
    """Requests and returns the number of qubits and the noise probability."""
    try:
        num_qubits = int(input("🔢 Number of Qubits in the GHZ circuit: "))
        # 🎚 Digite o nível de Ruído (0.0 a 1.0) -> Enter the Noise Level (0.0 to 1.0)
        noise_prob = float(input("🎚 Enter the Noise Level (0.0 to 1.0): "))
        if not (0.0 <= noise_prob <= 1.0):
             # Erro: O nível de ruído deve estar entre 0.0 e 1.0. -> Error: Noise level must be between 0.0 and 1.0.
             print("Error: Noise level must be between 0.0 and 1.0.", file=sys.stderr)
             sys.exit(1)
        return num_qubits, noise_prob
    except ValueError:
        # Erro: Entrada inválida. Use números inteiros e decimais. -> Error: Invalid input. Use integers and decimals.
        print("Error: Invalid input. Use integers and decimals.", file=sys.stderr)
        sys.exit(1)

# --- [CIRCUIT AND VALIDATION FUNCTIONS: Strings adjusted] ---

def generate_ghz_circuit_with_noise(nq, noise_prob):
    circuit = QuantumCircuit(nq)
    circuit.add_gate(H(0))
    apply_depolarizing_noise_manually(circuit, 0, noise_prob)
    for i in range(1, nq):
        circuit.add_gate(CNOT(0, i))
        apply_depolarizing_noise_manually(circuit, 0, noise_prob)
        apply_depolarizing_noise_manually(circuit, i, noise_prob)
    return circuit

def apply_depolarizing_noise_manually(circuit, target_qubit, noise_prob):
    if random.random() < noise_prob:
        op = random.choice(['X', 'Y', 'Z'])
        if op == 'X':
            circuit.add_gate(X(target_qubit))
        elif op == 'Y':
            circuit.add_gate(Y(target_qubit))
        else:
            circuit.add_gate(Z(target_qubit))

def measure(circuit, nq):
    state = QuantumState(nq)
    state.set_zero_state()
    circuit.update_quantum_state(state)
    amplitudes = state.get_vector()
    probabilities = np.abs(amplitudes) ** 2
    max_prob_index = np.argmax(probabilities)
    result = format(max_prob_index, f'0{nq}b')
    return result

def calculate_F_opt(H, S, C, I, T):
    try:
        result = subprocess.run(
            ["./sphy_simbiotic_entangle_ai", str(H), str(S), str(C), str(I), str(T)],
            capture_output=True, text=True, check=True
        )
        match = re.search(r"([-+]?\d*\.\d+|\d+)", result.stdout)
        return float(match.group(0)) if match else 0.0 
    except Exception:
        return 0.0

def generate_uid_via_bscore():
    """Adjusted to return 'Accepted' and 'Rejected' status."""
    try:
        result = subprocess.run(
            ["./ai_validator_bscore_uid"],
            capture_output=True, text=True, check=True
        )
        lines = result.stdout.strip().splitlines()
        # 'UID aceita' / 'UID rejeitada' are the search keywords from the Rust binary output
        target_line = next((line for line in reversed(lines) if "UID aceita" in line or "UID rejeitada" in line), None)
        
        if target_line is None:
            return "N/A", 0.0, "Error"

        parts = target_line.split("|")
        uid_part = parts[0].split(":")[1].strip()
        uid_match = re.search(r"Φ_[0-9a-fA-F]+", uid_part)
        uid = uid_match.group(0) if uid_match else "N/A"
        
        bscore = float(parts[1].replace("B(t) =", "").strip())
        status = "Accepted" if "UID aceita" in target_line else "Rejected" 
        
        return uid, bscore, status
        
    except Exception:
        return "-", 0.0, "Error"
        
# -------------------------------------------------------------
#                   MAIN SIMULATION FUNCTION (CONTINUOUS)
# -------------------------------------------------------------

def run_simulation(num_qubits, noise_prob):
    start_time = time.time()
    
    sphy_coherence = 90.0
    accepted_frames = 0
    frame = 0 
    
    # ➡️ TRANSLATED HEADER (Cabeçalho)
    print("\n" + "=" * 73)
    print("============================= SIMBIOTIC =================================")
    print("=")
    print("Gravito-Quantum Proof of Coherence Simbiotic AI. ")
    print(" ")
    print("Anti-Decriptography Security Level")
    print("A Quantum Blockchain Security Anti-Shor Protocol")
    print("GQM Gravitationa Quantum Modulation")
    print("Vibrational Simbiotic AI's ")
    print("Developed by QLZ Solutions® & Harpia Quantum®")
    print("https://www.linkedin.com/company/harpia-quantum/")
    print("=")
    print("=" * 73)
    # Ruído Base -> Base Noise
    print(f"🧬 HARPIA CORE | Qubits: {num_qubits} | Base Noise: {noise_prob:.2f}")
    print("-" * 73)
    
    # Simulação Quantica -> Quantum Simulation
    sys.stdout.write("Quantum Simulation\n") 
    
    sys.stdout.write("".ljust(100)) 
    sys.stdout.flush()

    try:
        while True:
            frame += 1 
            
            # --- SIMULATION AND CALCULATIONS ---
            circuit = generate_ghz_circuit_with_noise(num_qubits, noise_prob)
            measure(circuit, num_qubits) 

            H = random.uniform(0.95, 1.0)
            S = random.uniform(0.95, 1.0)
            C = sphy_coherence / 100
            I = abs(H - S)
            T = frame

            boost = calculate_F_opt(H, S, C, I, T)
            delta = boost * 0.7
            sphy_coherence = min(100, sphy_coherence + delta)

            uid, bscore, uid_status = generate_uid_via_bscore()
            
            is_accepted = bscore >= 0.900
            if is_accepted:
                accepted_frames += 1
                login_symbol = "✅"
            else:
                login_symbol = "❌"
            
            # --- CHRONOMETER AND DYNAMIC OUTPUT (ONE LINE) ---
            
            elapsed_time = time.time() - start_time
            hours, rem = divmod(elapsed_time, 3600)
            minutes, seconds = divmod(rem, 60)
            
            sys.stdout.write('\r') 
            
            # ➡️ TRANSLATED DYNAMIC LINE (Linha Dinâmica)
            # Coerência S(Φ) -> Coherence S(Φ)
            # UID✅: (accepted frames counter)
            # T: (Time)
            status_line = (
                f"| Frame {frame:,} | Login {login_symbol} UID: {uid} ({uid_status})  | "
                f"Coherence S(Φ): {int(sphy_coherence)}% | UID✅: {accepted_frames:,} | "
                f"T: {int(hours)}h {int(minutes)}m {seconds:.1f}s"
            )
            
            sys.stdout.write(status_line.ljust(100))
            sys.stdout.flush()
            
            time.sleep(0.01)

    except KeyboardInterrupt:
        sys.stdout.write('\n\n') 
        # Monitor interrompido pelo usuário (Ctrl+C). Gerando resumo... -> Monitor interrupted by user (Ctrl+C). Generating summary...
        print("Monitor interrupted by user (Ctrl+C). Generating summary...")

    # --- Final Summary (Resumo Final) ---
    end_time = time.time()
    total_time = end_time - start_time
    
    hours, rem = divmod(total_time, 3600)
    minutes, seconds = divmod(rem, 60)
    
    total_frames_processed = frame
    # Taxa de Aceitação -> Acceptance Rate
    taxa_aceitacao = (accepted_frames / total_frames_processed) * 100 if total_frames_processed > 0 else 0.0

    print("\n" + "=" * 65)
    # HARPIA MONITOR ENCERRADO -> HARPIA MONITOR ENDED
    print("                  HARPIA MONITOR ENDED")
    print("=" * 65)
    # UIDs Aceitos (Total) -> Accepted UIDs (Total)
    print(f"✅ Accepted UIDs (Total): {accepted_frames:,}")
    # Total de Frames Processados -> Total Frames Processed
    print(f"🔢 Total Frames Processed: {total_frames_processed:,}")
    # Tempo Total de Monitoramento -> Total Monitoring Time
    print(f"⏱ Total Monitoring Time: {int(hours)}h {int(minutes)}m {seconds:.2f}s")
    # Taxa de Aceitação (Login) -> Acceptance Rate (Login)
    print(f"📊 Acceptance Rate (Login): {taxa_aceitacao:.2f}%")
    print("=============================================================")

if __name__ == "__main__":
    qubits, noise_level = input_parameters()
    run_simulation(num_qubits=qubits, noise_prob=noise_level)