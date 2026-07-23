from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional

import ccxt
import numpy as np
import pandas as pd
import requests

BASE_DIR = Path(__file__).resolve().parent
STATE_FILE = BASE_DIR / "state.json"

SYMBOLS = [
    "BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT",
    "XRP/USDT", "DOGE/USDT", "LINK/USDT", "SUI/USDT",
]
BTC_SYMBOL = "BTC/USDT"
HTF, MTF, LTF = "4h", "1h", "15m"
CANDLE_LIMIT = 300
EMA_FAST, EMA_SLOW = 20, 50
RSI_PERIOD = ATR_PERIOD = ADX_PERIOD = 14
VOLUME_PERIOD = 20
MIN_ADX = 20.0
MIN_ATR_PERCENT = 0.25
MIN_VOLUME_RATIO = 0.85
MIN_SCORE = 9
MAX_SCORE = 12
STOP_ATR = 1.25
TP1_RR, TP2_RR, TP3_RR = 1.5, 2.0, 3.0
BOS_LOOKBACK, RETEST_LOOKBACK, SWEEP_LOOKBACK = 20, 4, 12
SEND_EMPTY_SUMMARY = False

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("v3-pro")

@dataclass(frozen=True)
class Signal:
    symbol: str
    side: Literal["BUY", "SELL"]
    grade: str
    score: int
    entry: float
    stop_loss: float
    target_1: float
    target_2: float
    target_3: float
    rsi: float
    adx: float
    volume_ratio: float
    atr_percent: float
    candle_timestamp: int
    reasons: tuple[str, ...]

def create_exchange():
    exchange_id = os.getenv("EXCHANGE_ID", "kucoin").strip().lower()
    exchange_class = getattr(ccxt, exchange_id, None)
    if exchange_class is None:
        raise ValueError(f"Unsupported exchange: {exchange_id}")
    exchange = exchange_class({"enableRateLimit": True, "timeout": 30000})
    exchange.load_markets()
    log.info("Connected to %s", exchange_id)
    return exchange

def fetch_candles(exchange, symbol: str, timeframe: str) -> pd.DataFrame:
    rows = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=CANDLE_LIMIT)
    if len(rows) < 100:
        raise RuntimeError(f"Insufficient candles: {symbol} {timeframe}")
    df = pd.DataFrame(rows, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df[["open", "high", "low", "close", "volume"]] = df[["open", "high", "low", "close", "volume"]].astype(float)
    df["timestamp"] = df["timestamp"].astype("int64")
    return df.iloc[:-1].copy()

def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["ema20"] = out["close"].ewm(span=EMA_FAST, adjust=False).mean()
    out["ema50"] = out["close"].ewm(span=EMA_SLOW, adjust=False).mean()

    delta = out["close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / RSI_PERIOD, adjust=False, min_periods=RSI_PERIOD).mean()
    avg_loss = loss.ewm(alpha=1 / RSI_PERIOD, adjust=False, min_periods=RSI_PERIOD).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    out["rsi"] = 100 - (100 / (1 + rs))

    prev_close = out["close"].shift(1)
    tr = pd.concat([
        out["high"] - out["low"],
        (out["high"] - prev_close).abs(),
        (out["low"] - prev_close).abs(),
    ], axis=1).max(axis=1)
    out["atr"] = tr.ewm(alpha=1 / ATR_PERIOD, adjust=False, min_periods=ATR_PERIOD).mean()

    up_move = out["high"].diff()
    down_move = -out["low"].diff()
    plus_dm = pd.Series(np.where((up_move > down_move) & (up_move > 0), up_move, 0.0), index=out.index)
    minus_dm = pd.Series(np.where((down_move > up_move) & (down_move > 0), down_move, 0.0), index=out.index)
    atr_safe = out["atr"].replace(0, np.nan)
    plus_di = 100 * plus_dm.ewm(alpha=1 / ADX_PERIOD, adjust=False, min_periods=ADX_PERIOD).mean() / atr_safe
    minus_di = 100 * minus_dm.ewm(alpha=1 / ADX_PERIOD, adjust=False, min_periods=ADX_PERIOD).mean() / atr_safe
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    out["adx"] = dx.ewm(alpha=1 / ADX_PERIOD, adjust=False, min_periods=ADX_PERIOD).mean()

    out["volume_average"] = out["volume"].rolling(VOLUME_PERIOD).mean()
    out["prior_high"] = out["high"].shift(1).rolling(BOS_LOOKBACK).max()
    out["prior_low"] = out["low"].shift(1).rolling(BOS_LOOKBACK).min()
    return out

def trend_direction(df: pd.DataFrame) -> Literal["bullish", "bearish", "neutral"]:
    row = df.iloc[-1]
    close, ema20, ema50 = float(row["close"]), float(row["ema20"]), float(row["ema50"])
    if close > ema20 > ema50:
        return "bullish"
    if close < ema20 < ema50:
        return "bearish"
    return "neutral"

def bullish_engulfing(previous: pd.Series, current: pd.Series) -> bool:
    return previous["close"] < previous["open"] and current["close"] > current["open"] and current["open"] <= previous["close"] and current["close"] >= previous["open"]

def bearish_engulfing(previous: pd.Series, current: pd.Series) -> bool:
    return previous["close"] > previous["open"] and current["close"] < current["open"] and current["open"] >= previous["close"] and current["close"] <= previous["open"]

def detect_bos_and_retest(df: pd.DataFrame, side: str) -> tuple[bool, bool, float]:
    recent = df.tail(RETEST_LOOKBACK + 1)
    current = df.iloc[-1]
    if side == "BUY":
        levels = recent["prior_high"].dropna()
        if levels.empty:
            return False, False, float("nan")
        level = float(levels.iloc[-1])
        bos = bool((recent["close"] > recent["prior_high"]).fillna(False).any())
        retest = bool(current["low"] <= level and current["close"] > level)
        return bos, retest, level
    levels = recent["prior_low"].dropna()
    if levels.empty:
        return False, False, float("nan")
    level = float(levels.iloc[-1])
    bos = bool((recent["close"] < recent["prior_low"]).fillna(False).any())
    retest = bool(current["high"] >= level and current["close"] < level)
    return bos, retest, level

def detect_liquidity_sweep(df: pd.DataFrame, side: str) -> bool:
    current = df.iloc[-1]
    history = df.iloc[-(SWEEP_LOOKBACK + 1):-1]
    if history.empty:
        return False
    if side == "BUY":
        level = float(history["low"].min())
        return bool(current["low"] < level and current["close"] > level)
    level = float(history["high"].max())
    return bool(current["high"] > level and current["close"] < level)

def grade_from_score(score: int) -> str:
    return "A+" if score >= 11 else "A" if score >= 9 else "B"

def load_state() -> dict:
    if not STATE_FILE.exists():
        return {"sent": {}}
    try:
        return json.loads(STATE_FILE.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {"sent": {}}

def save_state(state: dict) -> None:
    STATE_FILE.write_text(json.dumps(state, indent=2), encoding="utf-8")

def signal_key(signal: Signal) -> str:
    return f"{signal.symbol}|{signal.side}|{signal.candle_timestamp}"

def already_sent(signal: Signal, state: dict) -> bool:
    return signal_key(signal) in state.get("sent", {})

def remember_signal(signal: Signal, state: dict) -> None:
    state.setdefault("sent", {})[signal_key(signal)] = {
        "symbol": signal.symbol,
        "side": signal.side,
        "timestamp": signal.candle_timestamp,
        "score": signal.score,
    }
    items = list(state["sent"].items())
    if len(items) > 300:
        state["sent"] = dict(items[-300:])

def generate_signal(exchange, symbol: str, btc_4h_trend: str) -> Optional[Signal]:
    h4 = add_indicators(fetch_candles(exchange, symbol, HTF))
    h1 = add_indicators(fetch_candles(exchange, symbol, MTF))
    m15 = add_indicators(fetch_candles(exchange, symbol, LTF))
    h4_trend, h1_trend = trend_direction(h4), trend_direction(h1)
    current, previous = m15.iloc[-1], m15.iloc[-2]

    required = [current["ema20"], current["ema50"], current["rsi"], current["atr"], current["adx"], current["volume_average"]]
    if not all(np.isfinite(float(v)) for v in required):
        return None

    close, atr = float(current["close"]), float(current["atr"])
    rsi, adx = float(current["rsi"]), float(current["adx"])
    volume_average = float(current["volume_average"])
    if close <= 0 or atr <= 0 or volume_average <= 0:
        return None

    volume_ratio = float(current["volume"]) / volume_average
    atr_percent = (atr / close) * 100
    long_score = short_score = 0
    long_reasons, short_reasons = [], []

    if symbol == BTC_SYMBOL or btc_4h_trend == "bullish":
        long_score += 2; long_reasons.append("BTC 4H regime bullish")
    if symbol == BTC_SYMBOL or btc_4h_trend == "bearish":
        short_score += 2; short_reasons.append("BTC 4H regime bearish")

    if h4_trend == "bullish":
        long_score += 2; long_reasons.append("Coin 4H bullish")
    elif h4_trend == "bearish":
        short_score += 2; short_reasons.append("Coin 4H bearish")

    if h1_trend == "bullish":
        long_score += 2; long_reasons.append("Coin 1H bullish")
    elif h1_trend == "bearish":
        short_score += 2; short_reasons.append("Coin 1H bearish")

    if adx >= MIN_ADX:
        long_score += 1; short_score += 1
        long_reasons.append(f"ADX {adx:.1f}"); short_reasons.append(f"ADX {adx:.1f}")
    if atr_percent >= MIN_ATR_PERCENT:
        long_score += 1; short_score += 1
        long_reasons.append(f"ATR {atr_percent:.2f}%"); short_reasons.append(f"ATR {atr_percent:.2f}%")
    if volume_ratio >= MIN_VOLUME_RATIO:
        long_score += 1; short_score += 1
        long_reasons.append(f"Volume {volume_ratio:.2f}x"); short_reasons.append(f"Volume {volume_ratio:.2f}x")
    if 50 <= rsi <= 70:
        long_score += 1; long_reasons.append(f"RSI {rsi:.1f}")
    if 30 <= rsi <= 50:
        short_score += 1; short_reasons.append(f"RSI {rsi:.1f}")

    long_bos, long_retest, long_level = detect_bos_and_retest(m15, "BUY")
    short_bos, short_retest, short_level = detect_bos_and_retest(m15, "SELL")
    if long_bos:
        long_score += 1; long_reasons.append("15m BOS")
    if short_bos:
        short_score += 1; short_reasons.append("15m BOS")
    if long_retest:
        long_score += 1; long_reasons.append("15m retest")
    if short_retest:
        short_score += 1; short_reasons.append("15m retest")

    long_sweep = detect_liquidity_sweep(m15, "BUY")
    short_sweep = detect_liquidity_sweep(m15, "SELL")
    if long_sweep or bullish_engulfing(previous, current):
        long_score += 1; long_reasons.append("Sweep/engulfing confirmation")
    if short_sweep or bearish_engulfing(previous, current):
        short_score += 1; short_reasons.append("Sweep/engulfing confirmation")

    long_allowed = (symbol == BTC_SYMBOL or btc_4h_trend == "bullish") and h4_trend == "bullish" and h1_trend == "bullish" and adx >= MIN_ADX and atr_percent >= MIN_ATR_PERCENT and (long_bos or long_retest or long_sweep)
    short_allowed = (symbol == BTC_SYMBOL or btc_4h_trend == "bearish") and h4_trend == "bearish" and h1_trend == "bearish" and adx >= MIN_ADX and atr_percent >= MIN_ATR_PERCENT and (short_bos or short_retest or short_sweep)

    log.info("%s | BTC=%s 4H=%s 1H=%s | BUY=%s SELL=%s | ADX=%.1f RSI=%.1f VOL=%.2fx", symbol, btc_4h_trend, h4_trend, h1_trend, long_score, short_score, adx, rsi, volume_ratio)

    if long_allowed and long_score >= MIN_SCORE and long_score > short_score:
        structure_stop = float(m15.tail(SWEEP_LOOKBACK)["low"].min()) if np.isfinite(long_level) else close - atr * STOP_ATR
        stop_loss = min(structure_stop, close - atr * STOP_ATR)
        risk = close - stop_loss
        if risk <= 0:
            return None
        return Signal(symbol, "BUY", grade_from_score(long_score), long_score, close, stop_loss, close + risk * TP1_RR, close + risk * TP2_RR, close + risk * TP3_RR, rsi, adx, volume_ratio, atr_percent, int(current["timestamp"]), tuple(long_reasons))

    if short_allowed and short_score >= MIN_SCORE and short_score > long_score:
        structure_stop = float(m15.tail(SWEEP_LOOKBACK)["high"].max()) if np.isfinite(short_level) else close + atr * STOP_ATR
        stop_loss = max(structure_stop, close + atr * STOP_ATR)
        risk = stop_loss - close
        if risk <= 0:
            return None
        return Signal(symbol, "SELL", grade_from_score(short_score), short_score, close, stop_loss, close - risk * TP1_RR, close - risk * TP2_RR, close - risk * TP3_RR, rsi, adx, volume_ratio, atr_percent, int(current["timestamp"]), tuple(short_reasons))
    return None

def format_price(value: float) -> str:
    if value >= 1000:
        return f"{value:,.2f}"
    if value >= 1:
        return f"{value:.4f}"
    return f"{value:.7f}"

def format_signal(signal: Signal) -> str:
    reasons = "\n".join(f"✅ {r}" for r in signal.reasons)
    return (
        f"🚨 V3 PRO {signal.grade} SIGNAL\n\n"
        f"Pair: {signal.symbol}\nDirection: {signal.side}\nQuality: {signal.score}/{MAX_SCORE}\n\n"
        f"Entry: {format_price(signal.entry)}\nStop Loss: {format_price(signal.stop_loss)}\n"
        f"TP1 (1.5R): {format_price(signal.target_1)}\nTP2 (2R): {format_price(signal.target_2)}\nTP3 (3R): {format_price(signal.target_3)}\n\n"
        f"RSI: {signal.rsi:.1f}\nADX: {signal.adx:.1f}\nVolume: {signal.volume_ratio:.2f}x\nATR: {signal.atr_percent:.2f}%\n\n"
        f"{reasons}\n\n⚠️ Alert only—not guaranteed. Check the chart and limit risk."
    )

def send_telegram(message: str) -> None:
    token = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
    chat_id = os.getenv("TELEGRAM_CHAT_ID", "").strip()
    if not token or not chat_id:
        raise RuntimeError("Telegram secrets are missing")
    response = requests.post(
        f"https://api.telegram.org/bot{token}/sendMessage",
        json={"chat_id": chat_id, "text": message, "disable_web_page_preview": True},
        timeout=20,
    )
    response.raise_for_status()

def main() -> None:
    exchange = create_exchange()
    state = load_state()
    available = [s for s in SYMBOLS if s in exchange.markets]
    btc_4h_trend = trend_direction(add_indicators(fetch_candles(exchange, BTC_SYMBOL, HTF)))
    log.info("BTC 4H regime: %s", btc_4h_trend)
    signals, errors = [], []

    for symbol in available:
        try:
            signal = generate_signal(exchange, symbol, btc_4h_trend)
            if signal is None or already_sent(signal, state):
                continue
            send_telegram(format_signal(signal))
            remember_signal(signal, state)
            signals.append(signal)
        except (ccxt.NetworkError, ccxt.ExchangeError, requests.RequestException) as exc:
            errors.append(f"{symbol}: API error")
            log.warning("%s API error: %s", symbol, exc)
        except Exception as exc:
            errors.append(f"{symbol}: {type(exc).__name__}")
            log.exception("%s scan failed", symbol)

    save_state(state)
    summary = f"✅ V3 Pro scan completed\n\nBTC 4H regime: {btc_4h_trend.upper()}\nPairs checked: {len(available)}\nNew signals: {len(signals)}\nErrors: {len(errors)}"
    if signals:
        summary += "\n\n" + "\n".join(f"• {s.symbol}: {s.side} {s.grade} ({s.score}/{MAX_SCORE})" for s in signals)
        send_telegram(summary)
    elif SEND_EMPTY_SUMMARY:
        send_telegram(summary)
    log.info(summary.replace("\n", " | "))

if __name__ == "__main__":
    main()
