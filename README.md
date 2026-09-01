# SA-fWavKAN: State-Adaptive Fractional Wavelet Kolmogorov-Arnold Networks

<!-- Insert Badges Here (e.g., PyPI version, License, Build Status, CUDA Support) -->

This repository contains the official PyTorch implementation of **SA-fWavKAN**, a novel neural architecture designed to systematically resolve the temporal memory deficits and parameter explosion bottlenecks that impede standard Kolmogorov-Arnold Networks (KANs) in dynamic sequence modeling. 

By mathematically synthesizing a globally shared fractional-orthogonal Jacobi wavelet dictionary with continuous selective state space transitions, SA-fWavKAN seamlessly maps the continuous flow of physical sequence dynamics with unprecedented empirical efficacy and structural efficiency.

## 🌟 Key Innovations

*   **Bounded Parameter Complexity:** Bypasses the prohibitive $O(3N^2L)$ combinatorial parameter explosion of baseline Wav-KANs by deploying a globally shared dictionary, compressing the topological footprint to a highly constrained $O(N^2LD)$ scaling.
*   **Continuous Selective State Space:** Elevates state-space gating mechanisms ($B_t$, $C_t$, $\Delta_t$) directly into the fractional wavelet domain, embedding endogenous latent memory to effectively retain boundary constraints over long sequences.
*   **Adaptive Band-Pass Filtering:** Utilizes the Riemann-Liouville fractional derivative operator ($\mu$) optimized during backpropagation to naturally suppress non-stationary noise while preserving volatile temporal shockwaves.
*   **Hardware-Aware Execution:** Bridges continuous ordinary differential equations with discrete algorithmic execution using Zero Order Hold (ZOH) discretization, executing via a custom parallel associative scan to bypass High Bandwidth Memory (HBM) bottlenecks and maintain sub-quadratic $O(L_{seq})$ inference complexity.

## 📈 State-of-the-Art Empirical Performance

SA-fWavKAN establishes new quantifiable benchmarks across highly chaotic, multi-scale temporal domains:

*   **Subseasonal-to-Seasonal (S2S) Climate Prediction (ChaosBench):** Achieves a **1.94°C** RMSE and an ACC of 0.82 across a 44-day forecasting horizon utilizing only 42M parameters.
*   **Temporal Heterogeneous Graphs (TGB 2.0):** Delivers a **0.795** Test MRR in dynamic link prediction with an extraordinarily efficient 4.1 GB memory footprint, overcoming standard catastrophic forgetting.
*   **Zero-Leakage OOD Atmospheric Modeling (Weather-10K):** Attains a pristine **1.18°C** RMSE for 2-meter temperature extremes under strictly out-of-distribution regimes.

---
