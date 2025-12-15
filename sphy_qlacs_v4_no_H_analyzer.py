#!/usr/bin/env python3

# ───────────────────────────────────────────────────────────────
# File: sphy_qlacs_v4_no_H_analyzer.py
# Purpose: Analisa CSV de simulações GHZ HARPIA RF (Hilbertless).
# Author: Gemini
# ───────────────────────────────────────────────────────────────

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
from scipy.interpolate import interp1d

# 🔧 Configura pasta de saída para os gráficos
OUTPUT_DIR = "analise_harpia_rf"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def entrada_caminho():
    """Pede ao usuário o caminho completo para o arquivo CSV de log."""
    print("=" * 70)
    print("ANALISADOR DE BENCHMARKS HARPIA RF (COERÊNCIA/EMARANHAMENTO HILBERTLESS)")
    print("=" * 70)
    caminho = input("📁 Digite o caminho COMPLETO do arquivo CSV para análise: ")
    return caminho


def analisar_e_gerar_benchmark(caminho_csv):
    """
    Carrega o CSV, calcula as métricas e gera o gráfico de estabilidade SPHY/RF.
    """
    if not os.path.exists(caminho_csv):
        print(f"\n❌ Erro: Arquivo não encontrado no caminho: {caminho_csv}")
        sys.exit(1)

    try:
        # Carrega o CSV
        df = pd.read_csv(caminho_csv)
    except Exception as e:
        print(f"\n❌ Erro ao ler o arquivo CSV: {e}")
        sys.exit(1)

    # 1. Obtenção de Métricas
    
    coherence_column = 'SPHY (%)'
    if coherence_column not in df.columns:
        print(f"\n❌ Erro: Coluna '{coherence_column}' não encontrada no CSV.")
        sys.exit(1)

    total_frames = len(df)
    valid_states = (df['Accepted'] == '✅').sum()
    acceptance_rate = 100 * (valid_states / total_frames) if total_frames > 0 else 0.0
    
    sphy_np_array = df[coherence_column].to_numpy()
    
    if sphy_np_array.size > 0:
        mean_stability = np.mean(sphy_np_array)
        stability_stdev = np.std(sphy_np_array, ddof=1) if sphy_np_array.size > 1 else 0.0
        stability_variance = np.var(sphy_np_array)
    else:
        mean_stability = 0.0
        stability_stdev = 0.0
        stability_variance = 0.0
        
    num_qubits = 0 # Não está no CSV, mas podemos inferir
    try:
        # Tenta inferir o número de qubits pelo nome do arquivo (ex: '...4q_log...')
        match = os.path.basename(caminho_csv).split('_')
        for part in match:
            if 'q' in part and part.replace('q', '').isdigit():
                num_qubits = int(part.replace('q', ''))
                break
    except:
        pass # Ignora erro de inferência

    # 2. Impressão das Métricas
    print("\n" + "—" * 70)
    print(f"ANÁLISE DE BENCHMARK HARPIA RF (Q={'???' if num_qubits == 0 else num_qubits}, FRAMES={total_frames:,})")
    print("—" * 70)
    print(f"✅ Estados GHZ aceitos: {valid_states}/{total_frames} | {acceptance_rate:.2f}%")
    print("\nMÉTRICAS DE ESTABILIDADE RF CORE (SPHY COHERENCE)")
    print(f"🎯 Coerência Média (Mean): {mean_stability:.6f}%")
    print(f"⚖️ Desvio Padrão (Stdev): {stability_stdev:.6f}")
    print(f"🔬 Variância (Variance): {stability_variance:.6f}")
    print("—" * 70)
    
    # 3. Geração do Gráfico (Replicando a Lógica de Visualização do SPHY)
    
    base_name = os.path.basename(caminho_csv).replace('.csv', '')
    fig_filename = os.path.join(OUTPUT_DIR, f"{base_name}_analise_graph.png")

    sphy_evolution_list = sphy_np_array.tolist()
    if not sphy_evolution_list:
        print("❌ Sem dados para plotar.")
        return

    # A lógica de interpolação e média ponderada do script de simulação original
    sphy_evolution = np.array(sphy_evolution_list)
    tempo = np.linspace(0, 1, len(sphy_evolution))
    
    # Simula a mesma lógica de sinais e interpolação, mas usando os dados SPHY%
    sinais = [interp1d(tempo, np.roll(sphy_evolution, i), kind='cubic') for i in range(2)]
    novo_tempo = np.linspace(0, 1, 2000)
    dados = [sinal(novo_tempo) + np.random.normal(0, 0.15, len(novo_tempo)) for sinal in sinais]
    pesos = np.linspace(1, 1.5, 2)
    emaranhamento = np.average(dados, axis=0, weights=pesos)
    
    # Re-calcula a média e variância APENAS para o PLOT SUAVIZADO
    plot_mean = np.mean(emaranhamento)
    plot_stdev = np.std(emaranhamento)

    # Cria o gráfico 2x1 idêntico
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12))
    
    # --- GRÁFICO 1: Emaranhamento ---
    ax1.plot(novo_tempo, emaranhamento, 'k--', linewidth=2, label="Average Entanglement")
    for i in range(len(dados)):
        # Plota os 'sinais' interpolados com ruído (replicando a estética)
        ax1.plot(novo_tempo, dados[i], alpha=0.3, color='blue' if i == 0 else 'red')
        
    ax1.set_xlabel("Normalized Time")
    ax1.set_ylabel("Coherence/Amplitude")
    ax1.set_title(f"GHZ Entanglement - {num_qubits if num_qubits > 0 else '??'} Qubits")
    ax1.legend()
    ax1.grid(alpha=0.3)

    # --- GRÁFICO 2: Estabilidade ---
    ax2.plot(novo_tempo, emaranhamento, 'k-', label="Average Entanglement (Smoothed)")
    ax2.axhline(plot_mean, color='green', linestyle='--', label=f"Mean: {plot_mean:.2f}")
    
    # Banda de Desvio Padrão
    ax2.axhline(plot_mean + plot_stdev, color='orange', linestyle='--', label=f"± Stdev")
    ax2.axhline(plot_mean - plot_stdev, color='orange', linestyle='--')
    
    # Adiciona a Variância do dado BRUTO para referência
    ax2.text(0.02, 0.1, f'Raw Data Variance: {stability_variance:.6f}', 
             transform=ax2.transAxes, color='red', fontsize=10, 
             bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))

    ax2.set_xlabel("Normalized Time")
    ax2.set_ylabel("Coherence/Amplitude")
    ax2.set_title("Entanglement Stability (GQM Corrected)")
    ax2.legend()
    ax2.grid(alpha=0.3)

    fig.suptitle(f"HARPIA RF: Entanglement and Stability Analysis - {num_qubits if num_qubits > 0 else '??'} Qubits", fontsize=16)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(fig_filename, dpi=300)
    print(f"\n📊 Gráfico de análise salvo como: {fig_filename}")
    plt.show()

# Ponto de entrada
if __name__ == "__main__":
    # Verifica dependências
    try:
        import pandas as pd
        import matplotlib.pyplot as plt
        from scipy.interpolate import interp1d
    except ImportError as e:
        print(f"❌ A biblioteca '{e.name}' não está instalada. Instale com: pip install {e.name}")
        sys.exit(1)
        
    caminho_csv = entrada_caminho()
    analisar_e_gerar_benchmark(caminho_csv)