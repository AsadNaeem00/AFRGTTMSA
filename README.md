# Translation Noise in Multilingual Sentiment Features: Simulation Study

Code accompanying the paper **"[Your paper title here]"** ([Your Name], [Year]).

This repository contains a fully **synthetic** simulation study of how machine-translation (MT) errors in multilingual news-sentiment features degrade a graph neural network (GNN) that predicts sanction impact on a trade network. No real data are used.

## Repository contents

| File | Purpose |
|------|---------|
| `sim2.py` | Generates synthetic trade-network worlds, trains the models, applies translation-error processes, and saves raw results to `raw_results2.json`. |
| `analyze2.py` | Reads the raw results, computes bootstrap confidence intervals and Wilcoxon tests (with Holm correction for H1–H5), and produces `summary2.json` and `fig3.png`. |

## What the simulation does

- **World:** 500 countries (nodes) with a gravity-style weighted trade network (mean degree about 10), exposure values, a latent "true" sentiment, and a binary sanction-impact label. Each country has a share of coverage in low-resource languages, which drives its translation-error rate.
- **Models:**
  - `B0`: logistic regression without sentiment
  - `B1`: GCN without sentiment
  - `B2`: GCN with sentiment features (main model)
  - `B3`: direct multilingual scoring, i.e. no translation (scoring noise only)
- **Translation-error types:** lexical drift, polarity reversal, a mixed 75/25 process, a uniform-rate variant, a Gaussian noise baseline matched to the mixed process's energy, and a "reasoning MT" mitigation that halves drift and removes most polarity errors.
- **Experiments:**
  - degradation curves over error rates ρ ∈ {0 … 0.6}
  - cost-weighted financial loss (false negatives cost 5×, false positives 1×)
  - equity gap between high- and low-resource-language countries (tercile AUROC)
  - mitigation ablation at ρ = 0.2 and 0.4
  - single-node "blast radius" analysis
  - concentration of errors on high-exposure countries
  - a GCN-depth ablation (simplified GCN, K = 0–3)
- **Statistics:** 40 random seeds, 5 corruption repetitions each, bootstrap 95% CIs, Wilcoxon signed-rank tests, and Holm correction over hypotheses H1–H5.

Several parameters are **assumptions**, not estimates from data. They are set at the top of `sim2.py`: `SIG_L`, `PHI`, `SIG_D`, `C_FN`, `C_FP`, `DRIFT_SCALE_MIT` and `POL_KEEP_MIT`.

## Requirements

- Python 3.9+
- `numpy`, `scipy`, `scikit-learn`, `matplotlib`

```bash
pip install numpy scipy scikit-learn matplotlib
```

## Usage

```bash
# 1. Run the simulation (writes raw_results2.json)
python sim2.py

# 2. Analyse results (writes summary2.json and fig3.png)
python analyze2.py
```

The simulation is CPU-only and uses a small NumPy GCN implementation (no PyTorch needed). Runtime depends on your machine; the single-node blast-radius loop is the slowest part.

> **Note on paths:** the scripts currently read and write under `/home/claude/`. Edit the paths in `sim2.py` (the final `json.dump`) and in `analyze2.py` (the `json.load`, `json.dump` and `plt.savefig` calls) to match your own directory, e.g. `./raw_results2.json`.

## Outputs

- `raw_results2.json`: per-seed raw metrics (AUROC, macro-F1, cost-weighted loss)
- `summary2.json`: aggregated means, confidence intervals, p-values and Holm-adjusted p-values
- `fig3.png`: predictive degradation and financial loss increase vs. translation-error rate

## Reproducibility

All randomness is seeded (world seeds 0–39; corruption RNG seeded from the world seed; bootstrap RNG seed 7), so re-running gives identical results given the same library versions.

## Citation

```bibtex
@misc{yourkey2026,
  title  = {[Your paper title here]},
  author = {[Your Name]},
  year   = {2026}
}
```

## License

[Choose a license, e.g. MIT]
