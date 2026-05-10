# Experimental Analysis Results

This document summarizes the results of experiments conducted on the DiffTaichi Mass-Spring Robot simulation.

## Experiment 1: Learning Rate Schedule Comparison

The objective was to evaluate the impact of different learning rate schedules on the convergence and final loss of the simulation.

| Schedule | Final Loss | Min Loss |
| :--- | :--- | :--- |
| **Fixed LR=0.01 (Paper)** | 0.036971 | 0.036971 |
| Fixed LR=0.001 | 0.037818 | 0.037818 |
| **Fixed LR=0.1** | **0.009036** | **0.009036** |
| Step Decay (÷2 every 50) | 0.037498 | 0.037498 |
| Cosine Annealing | 0.037466 | 0.037466 |

**Statistical Test (Fixed LR=0.01 vs Cosine Annealing):**
- **t-statistic:** -34.1960
- **p-value:** 0.000000
- **Conclusion:** The difference is statistically significant (p < 0.05).

---

## Experiment 2: Optimizer Comparison (SGD vs Adam)

We compared the performance of the SGD optimizer (standard in paper) against Adam over 3 different random seeds.

### Summary Statistics
| Optimizer | Final Loss (mean) | Final Loss (std) | Best Loss |
| :--- | :--- | :--- | :--- |
| **SGD (lr=0.01)** | -0.0019 | 0.0001 | -0.0020 |
| **Adam (lr=0.001)** | **-0.0027** | **0.0000** | **-0.0027** |

### Statistical Test (SGD vs Adam):
- **t-statistic:** 7.5579
- **p-value:** 0.001642
- **Conclusion:** Adam significantly outperforms SGD in this task with a high degree of confidence (p < 0.05).

---

## Visualizations

The following figures illustrate the training dynamics and final distributions:

1. **Final Loss Distribution**: A boxplot comparing the stability and performance of SGD vs Adam across seeds.
2. **Loss Curves**: Mean loss with standard deviation shading, showing Adam's faster and more consistent convergence.

*Note: Visualizations are saved as PNG files in the results/ folder.*
