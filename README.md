# Renormalization Group Flow for ETFs

Applies Wilsonian Renormalization Group (RG) to ETF return time series. Coarse‑grains returns at different scales, derives flow equations for couplings (volatility, skewness, memory parameters), and identifies fixed points (scale‑invariant regimes). The per‑ETF score measures stability: high = near fixed point (stable), low = far from fixed point (regime change).

## Features
- Three ETF universes (FI/Commodities, Equity Sectors, Combined)
- Seven rolling windows (63–4536 days)
- Couplings: mean, volatility, skewness, kurtosis
- Coarse‑graining at multiple block sizes
- Fixed point identification via flow convergence
- Score = 1 / (1 + distance to fixed point)
- Two‑tab Streamlit dashboard (auto best, manual)
- Results stored on Hugging Face: `P2SAMAPA/p2-etf-renormalization-group-results`

## Usage

1. Set `HF_TOKEN` environment variable.
2. Install dependencies: `pip install -r requirements.txt`
3. Run training: `python train.py` (fast)
4. Launch dashboard: `streamlit run streamlit_app.py`

## Interpretation

- High RG stability → ETF is in a scale‑invariant (stable) regime.
- Low RG stability → ETF is far from fixed point → likely regime change.

## Requirements

See `requirements.txt`.
