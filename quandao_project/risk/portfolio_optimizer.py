import numpy as np
import pandas as pd
import cvxpy as cp
import warnings

def mean_variance_optimization(expected_returns: np.ndarray, cov_matrix: np.ndarray, 
                               risk_aversion: float = 2.0, max_weight: float = 0.1) -> np.ndarray:
    """
    Computes optimal portfolio weights using Mean-Variance Optimization.
    
    Parameters
    ----------
    expected_returns : 1D array of alpha scores or expected returns
    cov_matrix : 2D covariance matrix of asset returns
    risk_aversion : float controlling the trade-off between return and risk
    max_weight : float maximum allowed weight per asset
    """
    n = len(expected_returns)
    w = cp.Variable(n)
    
    # Ensure cov_matrix is positive semi-definite (PSD)
    cov_matrix = (cov_matrix + cov_matrix.T) / 2
    min_eig = np.min(np.real(np.linalg.eigvals(cov_matrix)))
    if min_eig < 0:
        cov_matrix -= 10 * min_eig * np.eye(*cov_matrix.shape)
        
    portfolio_return = expected_returns @ w
    portfolio_variance = cp.quad_form(w, cov_matrix)
    objective = cp.Maximize(portfolio_return - risk_aversion * portfolio_variance)
    
    # Constraints: Fully invested, long/short bounded
    constraints = [
        cp.sum(w) == 1,
        w >= -max_weight,
        w <= max_weight
    ]
    
    prob = cp.Problem(objective, constraints)
    try:
        prob.solve(solver=cp.SCS)
    except Exception as e:
        warnings.warn(f"Solver failed: {e}")
        
    if w.value is None:
        # Fallback to equal weight
        return np.ones(n) / n
        
    return w.value

def risk_parity_allocation(cov_matrix: np.ndarray) -> np.ndarray:
    """
    Computes Risk Parity (Inverse Volatility) allocation.
    Assumes zero correlation (diagonal dominance) for simplicity, which matches
    a pure inverse-volatility weighting scheme.
    """
    variances = np.diag(cov_matrix)
    variances = np.where(variances < 1e-8, 1e-8, variances) # Prevent division by zero
    inv_vol = 1.0 / np.sqrt(variances)
    return inv_vol / np.sum(inv_vol)

def generate_target_weights(returns_df: pd.DataFrame, alpha_scores: pd.Series, 
                            method: str = "mvo", max_weight: float = 0.1) -> pd.Series:
    """
    Main entry point for Portfolio Optimization layer.
    
    Parameters
    ----------
    returns_df : TxN DataFrame of historical returns for covariance calculation
    alpha_scores : Series of length N containing latest multi-factor composite scores
    method : 'mvo' or 'risk_parity'
    """
    assets = alpha_scores.index
    returns_matrix = returns_df[assets].fillna(0.0).values
    cov_matrix = np.cov(returns_matrix, rowvar=False)
    
    if method.lower() == "mvo":
        weights = mean_variance_optimization(
            expected_returns=alpha_scores.values, 
            cov_matrix=cov_matrix, 
            max_weight=max_weight
        )
    elif method.lower() == "risk_parity":
        weights = risk_parity_allocation(cov_matrix)
    else:
        raise ValueError(f"Unknown allocation method: {method}")
        
    return pd.Series(weights, index=assets, name="target_weight")
