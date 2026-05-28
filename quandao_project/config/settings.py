"""
quandao_project/config/settings.py
===================================
Single source of truth for ALL configuration.

Every value is pulled from the environment / .env file.
No file in data/, strategies/, risk/, or execution/ should
contain a hardcoded key, URL, path, or limit.

Usage
-----
    from quandao_project.config import Settings

    db_url  = Settings.DATABASE_URL
    app_id  = Settings.FYERS_APP_ID
"""

from __future__ import annotations

import os
from pathlib import Path
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Locate and load .env
# ---------------------------------------------------------------------------
# This file lives at:  quandao_project/config/settings.py
# The project root is three levels up:  config/ → quandao_project/ → root
_PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]
_ENV_PATH: Path = _PROJECT_ROOT / ".env"

load_dotenv(_ENV_PATH)


def _require(key: str) -> str:
    """Read an env var and raise a clear error if it is missing."""
    value = os.getenv(key)
    if not value:
        raise EnvironmentError(
            f"[Settings] Required environment variable '{key}' is not set. "
            f"Please add it to {_ENV_PATH}"
        )
    return value


def _optional(key: str, default: str = "") -> str:
    """Read an optional env var, returning *default* when absent."""
    return os.getenv(key, default)


# ===========================================================================
# Settings — all attributes are plain class-level values (loaded once at
# import time).  Import and use them directly; do NOT re-read os.getenv
# anywhere else in the project.
# ===========================================================================

class Settings:
    # -----------------------------------------------------------------------
    # Project paths (derived – never hardcoded)
    # -----------------------------------------------------------------------
    PROJECT_ROOT: Path = _PROJECT_ROOT
    ENV_PATH: Path = _ENV_PATH
    LOG_DIR: Path = _PROJECT_ROOT / "quandao_project" / "logs"

    # -----------------------------------------------------------------------
    # Database
    # -----------------------------------------------------------------------
    DATABASE_URL: str = _require("DATABASE_URL")

    # -----------------------------------------------------------------------
    # Fyers API (Indian equities broker)
    # -----------------------------------------------------------------------
    FYERS_APP_ID: str = _require("APP_ID")
    FYERS_APP_SECRET: str = _require("APP_SECRET_ID")
    FYERS_APP_ID_HASH: str = _optional("APP_ID_HASH")
    FYERS_PIN: str = _optional("PIN")
    FYERS_ACCESS_TOKEN: str = _optional("ACCESS_TOKEN")
    FYERS_REFRESH_TOKEN: str = _optional("REFRESH_TOKEN")
    FYERS_REDIRECT_URI: str = _optional("REDIRECT_URI", "https://sridamul.in/")

    # -----------------------------------------------------------------------
    # Data pipeline — backfill start dates
    # -----------------------------------------------------------------------
    DAILY_START_DATE: str = _optional("DAILY_START_DATE", "1990-01-01")
    ONE_MIN_START_DATE: str = _optional("ONE_MIN_START_DATE", "2017-01-01")
    FOREX_DAILY_START_DATE: str = _optional("FOREX_DAILY_START_DATE", "2000-01-01")
    FOREX_ONE_MIN_START_DATE: str = _optional("FOREX_ONE_MIN_START_DATE", "2015-01-01")

    # -----------------------------------------------------------------------
    # Risk / position sizing limits
    # -----------------------------------------------------------------------
    # Maximum capital at risk per trade as a fraction of total equity (0–1).
    MAX_RISK_PER_TRADE: float = float(_optional("MAX_RISK_PER_TRADE", "0.01"))

    # Maximum leverage multiplier applied to any position.
    MAX_LEVERAGE: float = float(_optional("MAX_LEVERAGE", "10.0"))

    # Maximum % of portfolio allocated to a single position.
    MAX_POSITION_SIZE_PCT: float = float(_optional("MAX_POSITION_SIZE_PCT", "0.05"))

    # -----------------------------------------------------------------------
    # Flask / API server
    # -----------------------------------------------------------------------
    FLASK_HOST: str = _optional("FLASK_HOST", "0.0.0.0")
    FLASK_PORT: int = int(_optional("FLASK_PORT", "5000"))
    FLASK_DEBUG: bool = _optional("FLASK_DEBUG", "false").lower() == "true"

    # -----------------------------------------------------------------------
    # Supported instruments (comma-separated in .env, or hard defaults here)
    # -----------------------------------------------------------------------
    # Example .env entry:
    #   INSTRUMENTS=NIFTY 50:NSE:NIFTY50-INDEX,RELIANCE:NSE:RELIANCE-EQ
    #
    # Returns a dict {display_name: db_symbol}.
    @classmethod
    def get_instruments(cls) -> dict[str, str]:
        raw = _optional("INSTRUMENTS", "")
        if raw:
            result = {}
            for pair in raw.split(","):
                pair = pair.strip()
                if ":" in pair:
                    name, symbol = pair.split(":", 1)
                    result[name.strip()] = symbol.strip()
            return result
        # Sensible defaults if the env var is absent
        return {
            "NIFTY 50": "NSE:NIFTY50-INDEX",
            "RELIANCE":  "NSE:RELIANCE-EQ",
            "HDFCBANK":  "NSE:HDFCBANK-EQ",
        }

    # -----------------------------------------------------------------------
    # Convenience helpers
    # -----------------------------------------------------------------------
    @classmethod
    def reload(cls) -> None:
        """Force a re-read of .env (useful in long-running processes)."""
        load_dotenv(_ENV_PATH, override=True)

    @classmethod
    def summary(cls) -> str:
        """Return a safe, redacted summary of active settings for logging."""
        def mask(value: str) -> str:
            return value[:6] + "***" if len(value) > 6 else "***"

        lines = [
            "=== Quandao Settings ===",
            f"  PROJECT_ROOT       : {cls.PROJECT_ROOT}",
            f"  ENV_PATH           : {cls.ENV_PATH}",
            f"  DATABASE_URL       : {mask(cls.DATABASE_URL)}",
            f"  FYERS_APP_ID       : {mask(cls.FYERS_APP_ID)}",
            f"  FYERS_ACCESS_TOKEN : {'SET' if cls.FYERS_ACCESS_TOKEN else 'NOT SET'}",
            f"  MAX_RISK_PER_TRADE : {cls.MAX_RISK_PER_TRADE}",
            f"  MAX_LEVERAGE       : {cls.MAX_LEVERAGE}",
            f"  FLASK_PORT         : {cls.FLASK_PORT}",
        ]
        return "\n".join(lines)
