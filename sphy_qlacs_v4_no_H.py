# -*- coding: utf-8 -*-
# ───────────────────────────────────────────────────────────────
# File: sphy_qlacs_v4_no_H.py
# Purpose: GHZ + HARPIA (Hilbertless Operator Engine) + Ruído Despolarizante Adaptativo
# Author: deywe@QLZ (converted to Hilbertless operator model)
# Notes: This version implements Interpretation 1: no StateVector, no explicit
#       Hilbert space. The circuit is a pipeline of symbolic operators that
#       update a SymbolicState instance. Final measurement is sampled from
#       the state's implicit probability model.
# ───────────────────────────────────────────────────────────────

import numpy as np
import matplotlib.pyplot as plt
import csv
import sys
import os
import random
import hashlib
from datetime import datetime
from tqdm import tqdm
from multiprocessing import Pool, Manager
from scipy.interpolate import interp1d

# Keep the meissner core for coherence/boost calculation
from meissner_core_ import meissner_correction_step

# Logging folder
LOG_DIR = "logs_harpia_qulacs_adaptive_hilbertless"
os.makedirs(LOG_DIR, exist_ok=True)

# ------------------------ Symbolic State & Operators -----------------------
class SymbolicState:
    """
    A compact symbolic representation of the system's state.
    This collects a few operational scalars that represent coherence,
    phase, entanglement index and a small implicit distribution over
    computational basis outcomes. There is NO explicit state vector.
    """
    def __init__(self, num_qubits):
        self.n = num_qubits
        # Coherence metric (0..100)
        self.coherence = 90.0
        # A phase-like angle per qubit (symbolic)
        self.phase = np.zeros(num_qubits)
        # Local amplitude proxy per basis (stored compressed as frequencies)
        # We'll store a small dict mapping basis string -> score
        self.scores = {format(0, f'0{num_qubits}b'): 1.0}  # starts near |00..0>
        # Entanglement index (symbolic scalar)
        self.entanglement = 0.0
        # noise accumulator
        self.noise = 0.0

    def normalize_scores(self):
        total = sum(self.scores.values())
        if total <= 0:
            # reset to uniform small nonzero
            self.scores = {k: 1.0 for k in self.scores}
            total = sum(self.scores.values())
        for k in list(self.scores.keys()):
            self.scores[k] = max(1e-12, self.scores[k] / total)

    def sample(self):
        """Return a sampled basis string according to implicit scores."""
        self.normalize_scores()
        keys = list(self.scores.keys())
        probs = np.array([self.scores[k] for k in keys], dtype=float)
        # numerical safety
        probs = probs / probs.sum()
        idx = np.random.choice(len(keys), p=probs)
        return keys[idx]

    def ensure_basis_exists(self, basis):
        if basis not in self.scores:
            self.scores[basis] = 1e-6

# --- Symbolic operator implementations ---

def op_H(state: SymbolicState, target: int):
    """Symbolic Hadamard: spreads score mass locally across bit-flips.
    Acts as a diffusion operator on the symbolic distribution and perturbs phase."""
    new_scores = {}
    for basis, s in state.scores.items():
        # flip target bit to create superposition-like redistribution
        flipped = list(basis)
        flipped[target] = '1' if basis[target] == '0' else '0'
        flipped = ''.join(flipped)
        # distribute s to basis and flipped with weight depending on coherence
        c = state.coherence / 100.0
        w_self = 0.5 * (1 + c * 0.1)
        w_flip = 0.5 * (1 - c * 0.1)
        new_scores[basis] = new_scores.get(basis, 0.0) + s * w_self
        new_scores[flipped] = new_scores.get(flipped, 0.0) + s * w_flip
    state.scores = new_scores
    # perturb the symbolic phase
    state.phase[target] = (state.phase[target] + random.uniform(-0.1, 0.1))
    # small entanglement increase
    state.entanglement += 0.01
    return state


def op_CNOT(state: SymbolicState, control: int, target: int):
    """Symbolic CNOT: conditionally permutes scores where control bit is '1'."""
    new_scores = {}
    for basis, s in state.scores.items():
        if basis[control] == '1':
            flipped = list(basis)
            flipped[target] = '1' if basis[target] == '0' else '0'
            flipped = ''.join(flipped)
            new_scores[flipped] = new_scores.get(flipped, 0.0) + s
        else:
            new_scores[basis] = new_scores.get(basis, 0.0) + s
    state.scores = new_scores
    # small entanglement coupling
    state.entanglement += 0.02
    return state


def op_depolarizing(state: SymbolicState, qubit: int, p: float):
    """Symbolic depolarizing noise: mixes distribution proportionally to p,
    increases internal noise accumulator and reduces coherence."""
    # mix scores toward local marginal uniform for that qubit
    marginal = {}
    for basis, s in state.scores.items():
        key0 = basis[:qubit] + '0' + basis[qubit+1:]
        key1 = basis[:qubit] + '1' + basis[qubit+1:]
        marginal[key0] = marginal.get(key0, 0.0) + s * 0.5
        marginal[key1] = marginal.get(key1, 0.0) + s * 0.5
    # apply mixing
    for k in state.scores.keys():
        state.scores[k] = (1 - p) * state.scores[k] + p * marginal.get(k, 0.0)
    # update symbolic coherence and noise
    state.noise += p
    state.coherence = max(0.0, state.coherence - p * 50.0 * 0.02)
    return state

# Utility to build a symbolic GHZ pipeline (list of ops)
def build_symbolic_ghz_pipeline(num_qubits, noise_prob):
    pipeline = []
    # H on qubit 0
    pipeline.append(('H', 0))
    # CNOTs from 0 to others
    for i in range(1, num_qubits):
        pipeline.append(('CNOT', 0, i))
    # attach depolarizing noise ops per qubit
    for i in range(num_qubits):
        pipeline.append(('NOISE', i, noise_prob))
    return pipeline

# Apply pipeline to state (operator -> state implicit)
def apply_pipeline(state: SymbolicState, pipeline):
    for op in pipeline:
        if op[0] == 'H':
            _, target = op
            state = op_H(state, target)
        elif op[0] == 'CNOT':
            _, c, t = op
            state = op_CNOT(state, c, t)
        elif op[0] == 'NOISE':
            _, q, p = op
            state = op_depolarizing(state, q, p)
        else:
            # unknown operator: skip
            pass
    return state

# ------------------------ Simulation worker (Hilbertless) ------------------

def simulate_frame_hilbertless(frame_data):
    """frame_data -> (frame, num_qubits, sphy_coherence, initial_noise_prob)
    Returns (log_entry, new_coherence, error)
    """
    frame, num_qubits, sphy_coherence, initial_noise_prob = frame_data
    current_timestamp = datetime.utcnow().isoformat()
    try:
        # HARPIA + Meissner calculations (same as before)
        H = random.uniform(0.95, 1.0)
        S = random.uniform(0.95, 1.0)
        C = sphy_coherence / 100.0
        I = abs(H - S)
        T = frame
        psi_state = [3.0, 3.0, 1.2, 1.2, 0.6, 0.5]

        boost, _, _ = meissner_correction_step(H, S, C, I, T, psi_state)
        delta = boost * 1.5
        rf_coherence = sphy_coherence + delta + np.random.normal(0, 5.0)
        new_coherence = min(100.0, rf_coherence)

        adapted_noise_prob = max(0.0, initial_noise_prob * (100.0 - new_coherence) / 100.0)

        # Build symbolic pipeline and apply it to a fresh symbolic state
        state = SymbolicState(num_qubits)
        state.coherence = new_coherence  # initialize with measured coherence
        pipeline = build_symbolic_ghz_pipeline(num_qubits, adapted_noise_prob)
        state = apply_pipeline(state, pipeline)

        # After pipeline, stabilize and normalize
        state.normalize_scores()

        # Sample a result from the implicit distribution
        result = state.sample()

    except Exception as e:
        return None, None, f"\nCritical error in frame {frame}: {e}"

    estados_ideais = ['0' * num_qubits, '1' * num_qubits]
    activated = delta > 0
    accepted = (result in estados_ideais) and activated

    data_to_hash = f"{frame}:{result}:{H:.4f}:{S:.4f}:{C:.4f}:{I:.4f}:{boost:.4f}:{new_coherence:.4f}:{current_timestamp}"
    sha256_signature = hashlib.sha256(data_to_hash.encode('utf-8')).hexdigest()

    log_entry = [
        frame, result, round(H, 4), round(S, 4), round(C, 4), round(I, 4),
        round(boost, 4), round(new_coherence, 4), "✅" if accepted else "❌",
        sha256_signature, current_timestamp
    ]
    return log_entry, new_coherence, None

# ------------------------ Main simulation (multiprocess) -------------------

def execute_simulation_multiprocessing(num_qubits, total_frames, initial_noise_prob=0.3, num_processes=4):
    print("=" * 60)
    print(f" 🧿 HARPIA RF CORE (Hilbertless) • {num_qubits} Qubits • {total_frames:,} Frames")
    print("=" * 60)

    timecode = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_filename = os.path.join(LOG_DIR, f"qghz_RF_hilbertless_{num_qubits}q_log_{timecode}.csv")
    fig_filename = os.path.join(LOG_DIR, f"qghz_RF_hilbertless_{num_qubits}q_graph_{timecode}.png")

    manager = Manager()
    sphy_coherence = manager.Value('f', 90.0)
    log_data, sphy_evolution = manager.list(), manager.list()
    valid_states = manager.Value('i', 0)

    def frame_generator():
        for f in range(1, total_frames + 1):
            yield (f, num_qubits, sphy_coherence.value, initial_noise_prob)

    print(f"🔄 Usando {num_processes} processos para simular...")
    with Pool(processes=num_processes) as pool:
        for log_entry, new_coherence, error in tqdm(pool.imap_unordered(simulate_frame_hilbertless, frame_generator()), total=total_frames, desc="⏳ Simulating GHZ"):
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

    acceptance_rate = 100 * (valid_states.value / total_frames) if total_frames > 0 else 0
    print(f"\n✅ GHZ States accepted: {valid_states.value}/{total_frames} | {acceptance_rate:.2f}%")

    if sphy_evolution:
        sphy_np_array = np.array(sphy_evolution)
        mean_stability = np.mean(sphy_np_array)
        stability_variance = np.var(sphy_np_array)
        print(f"\n📊 Mean Stability Index: {mean_stability:.6f}")
        print(f"📊 Stability Variance Index: {stability_variance:.6f}")

    with open(csv_filename, mode="w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["Frame", "Result", "H", "S", "C", "I", "Boost", "SPHY (%)", "Accepted", "SHA256_Signature", "Timestamp"])
        writer.writerows(list(log_data))
    print(f"🧾 CSV saved: {csv_filename}")

    # visualização (mesma lógica estilizada que antes)
    sphy_evolution_list = list(sphy_evolution)
    if not sphy_evolution_list:
        print("❌ No data to plot.")
        return

    sphy_evolution = np.array(sphy_evolution_list)
    tempo = np.linspace(0, 1, len(sphy_evolution))
    sinais = [interp1d(tempo, np.roll(sphy_evolution, i), kind='cubic') for i in range(2)]
    novo_tempo = np.linspace(0, 1, 2000)
    dados = [sinal(novo_tempo) + np.random.normal(0, 0.15, len(novo_tempo)) for sinal in sinais]
    pesos = np.linspace(1, 1.5, 2)
    emaranhamento = np.average(dados, axis=0, weights=pesos)
    estabilidade_media = np.mean(emaranhamento)
    estabilidade_variancia = np.var(emaranhamento)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12))
    ax1.plot(novo_tempo, emaranhamento, 'k--', linewidth=2, label="Average Entanglement")
    for i in range(len(dados)):
        ax1.plot(novo_tempo, dados[i], alpha=0.3, color='blue' if i == 0 else 'red')
    ax1.set_xlabel("Normalized Time")
    ax1.set_ylabel("Coherence/Amplitude")
    ax1.set_title(f"GHZ Entanglement - {num_qubits} Qubits")
    ax1.legend()
    ax1.grid()

    ax2.plot(novo_tempo, emaranhamento, 'k-', label="Average Entanglement")
    ax2.axhline(estabilidade_media, color='green', linestyle='--', label=f"Mean: {estabilidade_media:.2f}")
    ax2.axhline(estabilidade_media + np.sqrt(estabilidade_variancia), color='orange', linestyle='--', label=f"± Variance")
    ax2.axhline(estabilidade_media - np.sqrt(estabilidade_variancia), color='orange', linestyle='--')
    ax2.set_xlabel("Normalized Time")
    ax2.set_ylabel("Coherence/Amplitude")
    ax2.set_title("Entanglement Stability")
    ax2.legend()
    ax2.grid()

    fig.suptitle(f"GHZ Simulation: Entanglement and Stability - {num_qubits} Qubits", fontsize=16)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(fig_filename, dpi=300)
    print(f"\n📊 Graph saved as: {fig_filename}")
    plt.show()


if __name__ == "__main__":
    try:
        qubits = int(input("🔢 Número de Qubits no circuito GHZ: "))
        pairs = int(input("🔁 Total de estados GHZ a simular: "))
    except ValueError:
        print("❌ Entrada inválida. Por favor, insira números inteiros.")
        sys.exit(1)
    execute_simulation_multiprocessing(num_qubits=qubits, total_frames=pairs, num_processes=4)
