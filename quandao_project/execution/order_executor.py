"""
quandao_project/execution/order_executor.py
=============================================
Strict execution layer routing risk-approved payloads to brokers.

This module handles actual market execution via Fyers or MetaTrader 5 adapters,
as well as providing a robust Paper Trade Journaling mechanism.
It assumes all mathematical sizing and risk checks have already occurred.
"""

import os
import json
import time
import uuid
from datetime import datetime

# ===========================================================================
# 1. Local Trade Journaling
# ===========================================================================

class TradeJournal:
    """Logs approved order payloads into a local JSON ledger."""
    
    def __init__(self, filepath: str):
        self.filepath = filepath
        
    def log_trade(self, order_payload: dict, symbol: str, status: str = "FILLED") -> dict:
        trade = order_payload.copy()
        trade["ticket_id"] = trade.get("ticket_id", str(uuid.uuid4()))
        trade["symbol"] = symbol
        trade["timestamp"] = datetime.utcnow().isoformat()
        trade["status"] = status
        
        trades = []
        if os.path.exists(self.filepath):
            with open(self.filepath, "r") as f:
                try:
                    trades = json.load(f)
                except json.JSONDecodeError:
                    pass
                    
        trades.append(trade)
        
        with open(self.filepath, "w") as f:
            json.dump(trades, f, indent=4)
            
        return trade


# ===========================================================================
# 2. Adapter Pattern for Brokers (Backtest & Live)
# ===========================================================================

class FyersBacktestAdapter:
    """Simulates Fyers execution for backtesting."""
    
    def place_order(self, order_payload: dict, symbol: str) -> dict:
        journal = TradeJournal("fyers_backtest_trades.json")
        return journal.log_trade(order_payload, symbol, "FILLED_BACKTEST")


class FyersLiveAdapter:
    """Translates generic order payloads into Fyers API v3 live orders and logs them."""
    
    def place_order(self, order_payload: dict, symbol: str, fyers_client) -> dict:
        try:
            side = 1 if order_payload["signal"] == 1 else -1 
            
            fyers_order = {
                "symbol": symbol,
                "qty": int(order_payload["target_qty"]),
                "type": 2, # Market order
                "side": side,
                "productType": "INTRADAY",
                "limitPrice": 0,
                "stopPrice": 0,
                "validity": "DAY",
                "disclosedQty": 0,
                "offlineOrder": False,
                "stopLoss": round(abs(order_payload["entry_price"] - order_payload["stop_loss"]), 2),
                "takeProfit": round(abs(order_payload["entry_price"] - order_payload["take_profit"]), 2)
            }
            
            if not fyers_client:
                return {"status": "ERROR", "error": "Fyers client not provided."}

            response = fyers_client.place_order(data=fyers_order)
            
            if response.get("s") == "ok":
                ticket_id = response.get("id")
                order_payload["ticket_id"] = ticket_id
                
                journal = TradeJournal("fyers_live_trades.json")
                journal.log_trade(order_payload, symbol, "FILLED_LIVE")
                
                return {"status": "FILLED_LIVE", "ticket_id": ticket_id, "broker": "FYERS"}
            else:
                return {"status": "REJECTED_LIVE", "error": response.get("message")}
                
        except Exception as e:
            return {"status": "ERROR", "error": str(e)}

    def place_limit_order(self, fyers_client, symbol: str, side: int, qty: int, price: float) -> dict:
        try:
            fyers_order = {
                "symbol": symbol,
                "qty": int(qty),
                "type": 1, # Limit Order
                "side": side, # 1 = BUY, -1 = SELL
                "productType": "INTRADAY",
                "limitPrice": round(price, 2),
                "stopPrice": 0,
                "validity": "DAY",
                "disclosedQty": 0,
                "offlineOrder": False
            }
            if not fyers_client:
                return {"status": "ERROR", "error": "Fyers client not provided."}
            response = fyers_client.place_order(data=fyers_order)
            if response.get("s") == "ok":
                return {"status": "OK", "order_id": response.get("id")}
            else:
                return {"status": "REJECTED", "error": response.get("message")}
        except Exception as e:
            return {"status": "ERROR", "error": str(e)}

    def place_sl_market_order(self, fyers_client, symbol: str, side: int, qty: int, trigger_price: float) -> dict:
        try:
            fyers_order = {
                "symbol": symbol,
                "qty": int(qty),
                "type": 4, # Stop Market (SL-M)
                "side": side, # 1 = BUY, -1 = SELL
                "productType": "INTRADAY",
                "limitPrice": 0,
                "stopPrice": round(trigger_price, 2),
                "validity": "DAY",
                "disclosedQty": 0,
                "offlineOrder": False
            }
            if not fyers_client:
                return {"status": "ERROR", "error": "Fyers client not provided."}
            response = fyers_client.place_order(data=fyers_order)
            if response.get("s") == "ok":
                return {"status": "OK", "order_id": response.get("id")}
            else:
                return {"status": "REJECTED", "error": response.get("message")}
        except Exception as e:
            return {"status": "ERROR", "error": str(e)}

    def cancel_order(self, fyers_client, order_id: str) -> dict:
        try:
            if not fyers_client:
                return {"status": "ERROR", "error": "Fyers client not provided."}
            response = fyers_client.cancel_order(data={"id": order_id})
            if response.get("s") == "ok":
                return {"status": "OK", "order_id": response.get("id")}
            else:
                return {"status": "REJECTED", "error": response.get("message")}
        except Exception as e:
            return {"status": "ERROR", "error": str(e)}

    def get_positions(self, fyers_client) -> dict:
        try:
            if not fyers_client:
                return {"status": "ERROR", "error": "Fyers client not provided."}
            response = fyers_client.positions()
            if response.get("s") == "ok":
                return {"status": "OK", "netPositions": response.get("netPositions", [])}
            else:
                return {"status": "ERROR", "error": response.get("message")}
        except Exception as e:
            return {"status": "ERROR", "error": str(e)}

    def get_orders(self, fyers_client) -> dict:
        try:
            if not fyers_client:
                return {"status": "ERROR", "error": "Fyers client not provided."}
            response = fyers_client.orderbook()
            if response.get("s") == "ok":
                return {"status": "OK", "orderBook": response.get("orderBook", [])}
            else:
                return {"status": "ERROR", "error": response.get("message")}
        except Exception as e:
            return {"status": "ERROR", "error": str(e)}


class MT5BacktestAdapter:
    """Simulates MT5 execution for Forex backtesting."""
    
    def place_order(self, order_payload: dict, symbol: str) -> dict:
        journal = TradeJournal("mt5_backtest_trades.json")
        return journal.log_trade(order_payload, symbol, "FILLED_BACKTEST")


class MT5LiveAdapter:
    """Translates generic order payloads into MetaTrader 5 live trade requests and logs them."""
    
    def place_order(self, order_payload: dict, symbol: str) -> dict:
        try:
            import MetaTrader5 as mt5
            
            if not mt5.initialize():
                return {"status": "ERROR", "error": "MT5 initialization failed."}
                
            tick = mt5.symbol_info_tick(symbol)
            if tick is None:
                return {"status": "ERROR", "error": f"Failed to get tick for {symbol}. Market may be closed."}
                
            terminal_info = mt5.terminal_info()
            if terminal_info is None or not terminal_info.connected:
                return {"status": "ERROR", "error": "MT5 Terminal is not connected to the trade server."}
                
            action = mt5.ORDER_TYPE_BUY if order_payload["signal"] == 1 else mt5.ORDER_TYPE_SELL
            price = tick.ask if order_payload["signal"] == 1 else tick.bid
            
            # Check if broker requires market execution (no stops in initial request)
            symbol_info = mt5.symbol_info(symbol)
            is_market_exec = symbol_info is not None and symbol_info.trade_exemode == mt5.SYMBOL_TRADE_EXECUTION_MARKET
            
            sl = 0.0 if is_market_exec else float(order_payload.get("stop_loss", 0.0))
            tp = 0.0 if is_market_exec else float(order_payload.get("take_profit", 0.0))
            
            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": symbol,
                "volume": float(order_payload["target_qty"]),
                "type": action,
                "price": price,
                "sl": sl,
                "tp": tp,
                "deviation": 20,
                "magic": 100100,
                "comment": "Quandao System",
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_IOC,
            }
            
            result = mt5.order_send(request)
            
            if result.retcode != mt5.TRADE_RETCODE_DONE:
                return {"status": "REJECTED_LIVE", "error": result.comment, "retcode": result.retcode}
                
            ticket_id = result.order
            order_payload["ticket_id"] = ticket_id
            
            # Attach stops for market execution order afterward
            if is_market_exec and (order_payload.get("stop_loss", 0.0) > 0 or order_payload.get("take_profit", 0.0) > 0):
                time.sleep(0.1) # brief synchronization pause
                modify_request = {
                    "action": mt5.TRADE_ACTION_SLTP,
                    "symbol": symbol,
                    "position": ticket_id,
                    "sl": float(order_payload.get("stop_loss", 0.0)),
                    "tp": float(order_payload.get("take_profit", 0.0))
                }
                modify_res = mt5.order_send(modify_request)
                if modify_res.retcode != mt5.TRADE_RETCODE_DONE:
                    order_payload["modify_stops_error"] = modify_res.comment
            
            journal = TradeJournal("mt5_live_trades.json")
            journal.log_trade(order_payload, symbol, "FILLED_LIVE")
            
            return {"status": "FILLED_LIVE", "ticket_id": ticket_id, "broker": "MT5"}
            
        except Exception as e:
            return {"status": "ERROR", "error": str(e)}


# ===========================================================================
# 3. Main Router Engine
# ===========================================================================

def execute_order(
    order_payload: dict, 
    broker: str, 
    symbol: str, 
    is_live: bool = False, 
    fyers_client=None
) -> dict:
    """
    Main execution router. Parses the risk-approved payload and delegates it 
    to one of the 4 specific execution adapters based on broker and mode.
    
    Parameters
    ----------
    order_payload : The output dictionary from the Risk Manager.
    broker        : 'FYERS' or 'MT5'.
    symbol        : Target instrument.
    is_live       : If True, executes live API and logs. If False, runs historical backtest logic.
    fyers_client  : Authenticated Fyers instance (required if broker is FYERS and is_live is True).
    
    Returns
    -------
    dict: Final execution status and ticket information.
    """
    if not order_payload or order_payload.get("status") != "APPROVED":
        return {"status": "IGNORED", "message": "Payload is empty or not risk-approved."}
        
    broker_upper = broker.upper()
    
    if broker_upper == "FYERS":
        if is_live:
            adapter = FyersLiveAdapter()
            return adapter.place_order(order_payload, symbol, fyers_client)
        else:
            adapter = FyersBacktestAdapter()
            return adapter.place_order(order_payload, symbol)
            
    elif broker_upper == "MT5":
        if is_live:
            adapter = MT5LiveAdapter()
            return adapter.place_order(order_payload, symbol)
        else:
            adapter = MT5BacktestAdapter()
            return adapter.place_order(order_payload, symbol)
            
    else:
        return {"status": "ERROR", "error": f"Unknown broker specified: {broker}"}
