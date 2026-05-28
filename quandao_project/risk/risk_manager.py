"""
quandao_project/risk/risk_manager.py
======================================
Strict safety filter between strategy signals and broker execution.

This module handles position sizing and risk verification using
flat, pure functions with no external dependencies on execution or data.
"""

def calculate_position_size(
    equity: float, 
    risk_percentage: float, 
    atr_value: float, 
    stop_multiplier: float
) -> float:
    """
    Calculate trade size based on a fixed fractional risk model.
    
    Parameters
    ----------
    equity          : Total capital available.
    risk_percentage : Fraction of equity to risk (e.g., 0.01 for 1%).
    atr_value       : Current ATR value for the asset.
    stop_multiplier : Multiplier applied to ATR to determine stop distance.
    
    Returns
    -------
    float: Position size rounded to 2 decimal places.
    """
    if atr_value <= 0 or stop_multiplier <= 0:
        return 0.0
        
    stop_distance = atr_value * stop_multiplier
    risk_amount = equity * risk_percentage
    
    if stop_distance == 0:
        return 0.0
        
    position_size = risk_amount / stop_distance
    return round(max(position_size, 0.0), 2)


def evaluate_trade_allowance(
    current_open_positions: int, 
    max_allowed_positions: int, 
    daily_loss: float, 
    max_daily_loss_limit: float
) -> bool:
    """
    Risk gatekeeper determining if a trade is allowed to proceed.
    
    Parameters
    ----------
    current_open_positions : Number of currently open trades.
    max_allowed_positions  : Maximum concurrent trades allowed.
    daily_loss             : Current realized/unrealized daily loss (positive number).
    max_daily_loss_limit   : Maximum allowed daily loss before trading is halted.
    
    Returns
    -------
    bool: True if trade is allowed, False if risk limits are breached.
    """
    if current_open_positions >= max_allowed_positions:
        return False
        
    if daily_loss >= max_daily_loss_limit:
        return False
        
    return True


def generate_risk_adjusted_order(
    signal: int, 
    price: float, 
    atr_value: float, 
    equity: float, 
    settings: dict
) -> dict:
    """
    Calculate sizing and targets, returning a standardized order payload.
    
    Parameters
    ----------
    signal   : 1 (Buy), -1 (Sell), or 0 (Hold).
    price    : Current entry price.
    atr_value: Current ATR value.
    equity   : Current account equity.
    settings : Dictionary containing risk parameters. Expected keys:
               - 'risk_percentage' (e.g., 0.01)
               - 'stop_multiplier' (e.g., 1.5)
               - 'tp_multiplier' (e.g., 3.0)
               
    Returns
    -------
    dict: Standardized dictionary package or empty dict if signal is 0.
    """
    if signal == 0:
        return {}
        
    risk_pct = settings.get("risk_percentage", 0.01)
    sl_mult = settings.get("stop_multiplier", 1.5)
    tp_mult = settings.get("tp_multiplier", 3.0)
    
    qty = calculate_position_size(equity, risk_pct, atr_value, sl_mult)
    
    # If risk parameters resulted in a zero quantity, reject the order.
    if qty <= 0:
        return {}
        
    sl_distance = atr_value * sl_mult
    
    if tp_mult:
        tp_distance = atr_value * tp_mult
        tp_price = price + tp_distance if signal == 1 else price - tp_distance
    else:
        tp_price = 0.0
        
    if signal == 1:
        sl_price = price - sl_distance
    elif signal == -1:
        sl_price = price + sl_distance
    else:
        return {}
        
    return {
        "signal": signal,
        "target_qty": qty,
        "entry_price": round(price, 5),
        "stop_loss": round(sl_price, 5),
        "take_profit": round(tp_price, 5) if tp_price else 0.0,
        "status": "APPROVED"
    }
