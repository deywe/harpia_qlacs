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

# 🔧 Configura pasta de logs
LOG_DIR = "logs_harpia_qlacs_cpu_noise_manual"
os.makedirs(LOG_DIR, exist_ok=True)

# 🧠 Entrada de parâmetros
def entrada_parametros():
    try:
        num_qubits = int(input("🔢 Número de Qubits no circuito GHZ: "))
        total_pares = int(input("🔁 Total de estados GHZ a simular: "))
        return num_qubits, total_pares
    except ValueError:
        print("❌ Entrada inválida.")
        sys.exit(1)

# 🧬 Geração do circuito GHZ com ruído manual
def gerar_ghz_circuit_com_ruido(nq, noise_prob):
    circuit = QuantumCircuit(nq)
    
    # Aplica Hadamard no primeiro qubit
    circuit.add_gate(H(0))
    
    # Simula ruído de depolarização após a porta H
    apply_depolarizing_noise_manually(circuit, 0, noise_prob)

    # Aplica CNOTs para criar o estado GHZ
    for i in range(1, nq):
        circuit.add_gate(CNOT(0, i))
        # Simula ruído de depolarização após a porta CNOT
        apply_depolarizing_noise_manually(circuit, 0, noise_prob)
        apply_depolarizing_noise_manually(circuit, i, noise_prob)
        
    return circuit

def apply_depolarizing_noise_manually(circuit, target_qubit, noise_prob):
    """Aplica ruído de depolarização em um qubit alvo."""
    if random.random() < noise_prob:
        op = random.choice(['X', 'Y', 'Z'])
        if op == 'X':
            circuit.add_gate(X(target_qubit))
        elif op == 'Y':
            circuit.add_gate(Y(target_qubit))
        else: # op == 'Z'
            circuit.add_gate(Z(target_qubit))
            
# 🧪 Medição no simulador local (com CPU e ruído)
def medir(circuit, nq):
    """Executa o circuito e retorna o bitstring medido."""
    state = QuantumState(nq)
    state.set_zero_state()
    circuit.update_quantum_state(state)
    
    amplitudes = state.get_vector()
    probabilities = np.abs(amplitudes) ** 2
    
    max_prob_index = np.argmax(probabilities)
    result = format(max_prob_index, f'0{nq}b')

    return result

# ⚙️ Chamada ao binário HARPIA externo (Rust)
def calcular_F_opt(H, S, C, I, T):
    try:
        result = subprocess.run(
            ["./sphy_simbiotic_entangle_ai", str(H), str(S), str(C), str(I), str(T)],
            capture_output=True, text=True, check=True
        )
        match = re.search(r"([-+]?\d*\.\d+|\d+)", result.stdout)
        if match:
            return float(match.group(0))
        else:
            raise ValueError(f"❌ Falha ao extrair valor de saída: {result.stdout}")
    except FileNotFoundError:
        print("❌ Erro: O binário 'sphy_simbiotic_entangle_ai' não foi encontrado.")
        sys.exit(1)
    except subprocess.CalledProcessError as e:
        print(f"❌ Erro ao executar o binário: {e.stderr}")
        sys.exit(1)

# 🚀 Simulação principal
def executar_simulacao(num_qubits, total=100000, noise_prob=0.01):
    print("=" * 60)
    print(f"    🧿 HARPIA QGHZ STABILIZER • {num_qubits} Qubits • {total:,} Frames (Qulacs/CPU)")
    print("=" * 60)

    timecode = datetime.now().strftime("%Y%m%d_%H%M%S")
    nome_csv = os.path.join(LOG_DIR, f"qghz_{num_qubits}q_log_{timecode}.csv")
    nome_fig = os.path.join(LOG_DIR, f"qghz_{num_qubits}q_graph_{timecode}.png")

    sphy_coherence = 90.0
    validos = 0
    log_data = []
    sphy_evolution = []
    estados_ideais = ['0' * num_qubits, '1' * num_qubits]

    for frame in tqdm(range(1, total + 1), desc="⏳ Simulando GHZ"):
        circuito = gerar_ghz_circuit_com_ruido(num_qubits, noise_prob)
        resultado = medir(circuito, num_qubits)

        H = random.uniform(0.95, 1.0)
        S = random.uniform(0.95, 1.0)
        C = sphy_coherence / 100
        I = abs(H - S)
        T = frame

        boost = calcular_F_opt(H, S, C, I, T)
        delta = boost * 0.7
        novo = min(100, sphy_coherence + delta)
        ativado = delta > 0
        aceito = (resultado in estados_ideais) and ativado
        if aceito:
            validos += 1

        sphy_coherence = novo
        sphy_evolution.append(sphy_coherence)

        log_line = [
            frame, resultado,
            round(H, 4), round(S, 4),
            round(C, 4), round(I, 4),
            round(boost, 4), round(sphy_coherence, 4),
            "✅" if aceito else "❌"
        ]

        hash_input = ",".join(map(str, log_line))
        uid_sha256 = hashlib.sha256(hash_input.encode()).hexdigest()

        log_line.append(uid_sha256)
        log_data.append(log_line)

        sys.stdout.flush()

    acceptance_rate = 100 * (validos / total)
    print(f"\n✅ GHZ States aceitos: {validos}/{total}|{acceptance_rate:.2f}%")

    with open(nome_csv, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([
            "Frame", "Result", "H", "S", "C", "I",
            "Boost", "SPHY (%)", "Accepted", "UID_SHA256"
        ])
        writer.writerows(log_data)
    print(f"🧾 CSV salvo: {nome_csv}")

    plt.figure(figsize=(12, 5))
    plt.plot(range(1, total + 1), sphy_evolution, color="darkcyan", label="⧉ SPHY Coherence")
    plt.scatter(
        range(1, total + 1), sphy_evolution,
        c=['green' if row[-2] == "✅" else 'red' for row in log_data],
        s=8, alpha=0.6
    )
    plt.axhline(90, color='gray', linestyle="dotted", linewidth=1, label="Threshold")
    plt.title(f"📡 HARPIA SPHY Evolution • {num_qubits} Qubits • {total:,} Frames (Qulacs/CPU)")
    plt.xlabel("Frames")
    plt.ylabel("SPHY Coherence (%)")
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(nome_fig, dpi=300)
    print(f"📊 Gráfico salvo como: {nome_fig}")
    plt.show()

# Ponto de entrada
if __name__ == "__main__":
    qubits, pares = entrada_parametros()
    executar_simulacao(num_qubits=qubits, total=pares, noise_prob=0.30)