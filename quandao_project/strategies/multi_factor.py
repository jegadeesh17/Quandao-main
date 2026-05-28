import pandas as pd
import numpy as np

def neutralize_factor(df: pd.DataFrame, factor_col: str, group_col: str = None) -> pd.Series:
    """
    Neutralizes a factor by cross-sectional de-meaning within a group.
    Assumes df represents a cross-section or we are neutralizing temporally.
    """
    if group_col and group_col in df.columns:
        mean = df.groupby(group_col)[factor_col].transform('mean')
        return df[factor_col] - mean
    else:
        return df[factor_col] - df[factor_col].mean()

def compute_information_coefficient(df: pd.DataFrame, factor_col: str, fwd_return_col: str) -> float:
    """
    Computes Spearman Rank IC for a single factor against forward returns.
    """
    if df[factor_col].std() == 0 or df[fwd_return_col].std() == 0:
        return 0.0
    return df[factor_col].corr(df[fwd_return_col], method='spearman')

def generate_multi_factor_composite(df: pd.DataFrame, factor_weights: dict) -> pd.Series:
    """
    Constructs a composite alpha signal using weighted normalized factors.
    """
    composite = pd.Series(0.0, index=df.index)
    
    for factor, weight in factor_weights.items():
        if factor in df.columns:
            # Z-score normalization
            mean = df[factor].mean()
            std = df[factor].std()
            if std > 0:
                normalized = (df[factor] - mean) / std
                composite += (normalized * weight)
                
    return composite

def generate_directional_signal_from_composite(df: pd.DataFrame, composite_col: str, threshold: float = 1.0) -> pd.DataFrame:
    """
    Converts a continuous composite score into the strict [-1, 0, 1] signal contract required by Quandao.
    """
    out_df = df.copy()
    out_df["signal"] = 0
    
    out_df.loc[out_df[composite_col] > threshold, "signal"] = 1
    out_df.loc[out_df[composite_col] < -threshold, "signal"] = -1
    
    return out_df
