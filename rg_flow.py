import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

def compute_couplings(series):
    """
    Compute the couplings (parameters) of the time series:
    mean, volatility, skewness, kurtosis.
    """
    if len(series) < 5:
        return np.zeros(4)
    mean = np.mean(series)
    vol = np.std(series)
    skew = np.mean(((series - mean) / vol)**3) if vol > 0 else 0.0
    kurt = np.mean(((series - mean) / vol)**4) - 3 if vol > 0 else 0.0
    return np.array([mean, vol, skew, kurt])

def coarse_grain(series, block_size):
    """
    Coarse-grain a time series by averaging over blocks of size `block_size`.
    """
    n = len(series)
    n_blocks = n // block_size
    if n_blocks == 0:
        return np.array([])
    coarse = np.array([np.mean(series[i*block_size:(i+1)*block_size]) for i in range(n_blocks)])
    return coarse

def rg_flow(series, block_sizes=[2, 4, 8, 16, 32]):
    """
    Compute the RG flow: couplings as a function of block size.
    Returns: flow trajectories for each coupling.
    """
    couplings = []
    for b in block_sizes:
        coarse = coarse_grain(series, b)
        if len(coarse) < 5:
            break
        c = compute_couplings(coarse)
        couplings.append(c)
    if len(couplings) == 0:
        return np.array([])
    return np.array(couplings)

def compute_composite_macro_factor(macro_df):
    """Compute composite macro factor from all macro variables."""
    if len(macro_df) < 2:
        return np.ones(len(macro_df)) * 0.5
    scaler = StandardScaler()
    macro_scaled = scaler.fit_transform(macro_df)
    pca = PCA(n_components=1)
    factor = pca.fit_transform(macro_scaled).flatten()
    factor = (factor - factor.min()) / (factor.max() - factor.min() + 1e-8)
    return factor

def find_fixed_point(flow_trajectory):
    """
    Find the fixed point of the RG flow (where couplings no longer change).
    """
    if len(flow_trajectory) < 2:
        return np.zeros(len(flow_trajectory[0]))
    # Compute differences between consecutive coupling vectors
    diffs = np.diff(flow_trajectory, axis=0)
    # Find where the differences are smallest (closest to fixed point)
    diff_norms = np.linalg.norm(diffs, axis=1)
    if len(diff_norms) == 0:
        return flow_trajectory[-1]
    min_idx = np.argmin(diff_norms)
    fixed_point = flow_trajectory[min_idx + 1] if min_idx + 1 < len(flow_trajectory) else flow_trajectory[-1]
    return fixed_point

def rg_score(returns, macro_df, block_sizes=[2, 4, 8, 16, 32]):
    """
    Compute per-ETF RG score = distance from the current coupling to the fixed point.
    Small distance = stable regime; large distance = regime change.
    """
    if len(returns) < 5 or macro_df is None or len(macro_df) < 5:
        return 0.0
    # Align lengths
    min_len = min(len(returns), len(macro_df))
    returns = returns[:min_len]
    macro_df = macro_df.iloc[:min_len]
    # Remove NaN
    mask = ~(np.isnan(returns) | np.isnan(macro_df).any(axis=1))
    returns = returns[mask]
    macro_df = macro_df[mask]
    if len(returns) < 10:
        return 0.0
    # Compute RG flow
    flow = rg_flow(returns, block_sizes)
    if len(flow) < 2:
        return 0.0
    # Find fixed point
    fixed_point = find_fixed_point(flow)
    # Current couplings (at original scale)
    current = compute_couplings(returns)
    # Distance from fixed point (normalised)
    distance = np.linalg.norm(current - fixed_point) / (np.linalg.norm(fixed_point) + 1e-8)
    # Macro factor to modulate score
    macro_factor = compute_composite_macro_factor(macro_df)[-1]
    # Higher macro factor increases sensitivity (larger distance)
    adjusted_distance = distance * (1 + macro_factor)
    # Score = 1 / (1 + distance) so that small distance = high score (stable)
    score = 1.0 / (1.0 + adjusted_distance)
    return float(score)
