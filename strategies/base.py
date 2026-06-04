"""
backend/strategies/base.py
Abstract base class that all strategies must implement.
This ensures a consistent interface: every strategy accepts a DataFrame
and returns a standardised result dict.
"""

from abc import ABC, abstractmethod
import pandas as pd


class BaseStrategy(ABC):
    """
    All Quandao strategies must subclass this and implement :meth:`run`.

    The :meth:`run` method receives OHLCV data as a DataFrame and any
    strategy-specific keyword arguments, and must return a dict with at
    least the following keys:

    .. code-block:: python

        {
            "trades":  list[dict],   # one dict per executed trade
            "metrics": dict          # summary statistics (win_rate, total_pnl, …)
        }

    Strategies may also return optional keys such as ``"metadata"``,
    ``"yearly_performance"``, ``"monthly_performance"``, etc.
    """

    def prepare_data(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        Prepare raw OHLCV data by timezone normalizing, sorting, and adding indicators.

        Parameters
        ----------
        df : pd.DataFrame
            Raw OHLCV data.
        **kwargs :
            Parameters for indicators or preprocessing.

        Returns
        -------
        pd.DataFrame
            Prepared DataFrame.
        """
        if df is None or df.empty:
            return pd.DataFrame()

        df = df.copy()

        # Ensure time column is set
        if 'time' not in df.columns:
            if df.index.name == 'time' or (isinstance(df.index, pd.DatetimeIndex) and df.index.name is None):
                df = df.reset_index()
            else:
                df['time'] = df.index

        df = df.sort_values('time').reset_index(drop=True)
        if not pd.api.types.is_datetime64_any_dtype(df['time']):
            df['time'] = pd.to_datetime(df['time'], utc=True)

        return df

    @abstractmethod
    def run(self, df: pd.DataFrame, **kwargs) -> dict:
        """
        Execute the strategy on *df* and return results.

        Parameters
        ----------
        df : pd.DataFrame
            OHLCV data with columns: time, open, high, low, close, volume.
            ``time`` must be a timezone-aware datetime.
        **kwargs :
            Strategy-specific parameters (e.g. stop_loss_pts, orb_minutes).

        Returns
        -------
        dict
            Must contain keys ``trades`` (list of trade dicts) and
            ``metrics`` (summary statistics dict).
        """
        ...

    @property
    def name(self) -> str:
        """Human-readable strategy name. Defaults to the class name."""
        return type(self).__name__
