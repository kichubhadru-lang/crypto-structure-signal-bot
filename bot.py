
from __future__ import annotations

import json
import logging
import os
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional

import ccxt
import numpy as np
import pandas as pd
import requests
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("structure-bot")


@dataclass(frozen=True)
class Structure:
    direction: Literal["bullish", "bearish", "range", "unknown"]
    last_swing_high: Optional[float]
    previous_swing_high: Optional[float]
    last_swing_low: Optional[float]
    previous_swing_low: Optional[float]


@dataclass(frozen=True)
class Signal:
    symbol: str
    side: Literal["BUY", "SELL"]
    entry: float
    stop: float
    target_1: float
    target_2: float
    reasons: tuple[str, ...]
    candle_timestamp: int


def load_config() -> dict:
    with open(BASE_DIR / "config.json", "r", encoding="utf-8") as f:
        return json.load(f)


def make_exchange(exchange_id: str):
    exchange_class = getattr(ccxt, exchange_id, None)
    if exchange_class is None:
        raise ValueError(f"Unsupported exchange: {exchange_id}")
    exchange = exchange_class({"enableRateLimit": True})
    exchange.load_markets()
    return exchange


def fetch_ohlcv(exchange, symbol: str, timeframe: str, limit: int) -> pd.DataFrame:
    rows = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
    if len(rows) < 100:
        raise RuntimeError(f"Insufficient candles for {symbol} {timeframe}")
    df = pd.DataFrame(rows, columns=["timestamp", "open", "high", "low", "close", "volume"])
    return df.astype({
        "timestamp": "int64", "open": "float64", "high": "float64",
        "low": "float64", "close": "float64", "volume": "float64"
    })


def add_atr(df: pd.DataFrame, period: int) -> pd.DataFrame:
    out = df.copy()
    prev_close = out["close"].shift(1)
    tr = pd.concat([
        out["high"] - out["low"],
        (out["high"] - prev_close).abs(),
        (out["low"] - prev_close).abs()
    ], axis=1).max(axis=1)
    out["atr"] = tr.rolling(period).mean()
    return out


def mark_swings(df: pd.DataFrame, left: int, right: int) -> pd.DataFrame:
    out = df.copy()
    window = left + right + 1
    out["swing_high"] = out["high"].eq(
        out["high"].rolling(window, center=True).max()
    )
    out["swing_low"] = out["low"].eq(
        out["low"].rolling(window, center=True).min()
    )
    return out


def detect_structure(df: pd.DataFrame, left: int, right: int) -> Structure:
    marked = mark_swings(df, left, right)
    highs = marked.loc[marked["swing_high"], "high"].dropna()
    lows = marked.loc[marked["swing_low"], "low"].dropna()

    if len(highs) < 2 or len(lows) < 2:
        return Structure("unknown", None, None, None, None)

    prev_high, last_high = float(highs.iloc[-2]), float(highs.iloc[-1])
    prev_low, last_low = float(lows.iloc[-2]), float(lows.iloc[-1])

    if last_high > prev_high and last_low > prev_low:
        direction = "bullish"
    elif last_high < prev_high and last_low < prev_low:
        direction = "bearish"
    else:
        direction = "range"

    return Structure(direction, last_high, prev_high, last_low, prev_low)


def crossed_above(series: pd.Series, level: float) -> bool:
    return bool(series.iloc[-2] <= level < series.iloc[-1])


def crossed_below(series: pd.Series, level: float) -> bool:
    return bool(series.iloc[-2] >= level > series.iloc[-1])


def generate_signal(exchange, symbol: str, cfg: dict) -> Optional[Signal]:
    limit = int(cfg["candle_limit"])
    htf = fetch_ohlcv(exchange, symbol, cfg["higher_timeframe"], limit)
    setup = fetch_ohlcv(exchange, symbol, cfg["setup_timeframe"], limit)
    entry = fetch_ohlcv(exchange, symbol, cfg["entry_timeframe"], limit)

    left, right = int(cfg["swing_left"]), int(cfg["swing_right"])
    htf_structure = detect_structure(htf, left, right)
    setup_structure = detect_structure(setup, left, right)

    if htf_structure.direction not in ("bullish", "bearish"):
        return None
    if setup_structure.direction != htf_structure.direction:
        return None

    entry = add_atr(entry, int(cfg["atr_period"]))
    entry["volume_sma"] = entry["volume"].rolling(int(cfg["volume_sma_period"])).mean()

    last = entry.iloc[-1]
    atr = float(last["atr"]) if pd.notna(last["atr"]) else np.nan
    volume_sma = float(last["volume_sma"]) if pd.notna(last["volume_sma"]) else np.nan
    if not np.isfinite(atr) or not np.isfinite(volume_sma) or atr <= 0:
        return None

    volume_ratio = float(last["volume"] / volume_sma) if volume_sma > 0 else 0.0
    if volume_ratio < float(cfg["minimum_volume_ratio"]):
        return None

    marked_entry = mark_swings(entry, left, right)
    swing_highs = marked_entry.loc[marked_entry["swing_high"], "high"].dropna()
    swing_lows = marked_entry.loc[marked_entry["swing_low"], "low"].dropna()
    if len(swing_highs) < 1 or len(swing_lows) < 1:
        return None

    recent_high = float(swing_highs.iloc[-1])
    recent_low = float(swing_lows.iloc[-1])
    close = float(last["close"])
    tolerance = atr * float(cfg["pullback_atr_tolerance"])
    stop_buffer = atr * float(cfg["stop_atr_buffer"])
    min_rr = float(cfg["minimum_rr"])

    if htf_structure.direction == "bullish":
        # BOS must have occurred recently; current candle should hold/reclaim the broken level.
        recent_bos = (entry["close"].tail(8) > recent_high).any()
        retest = abs(close - recent_high) <= tolerance or close > recent_high
        bullish_candle = close > float(last["open"])
        if not (recent_bos and retest and bullish_candle):
            return None

        stop = min(recent_low, recent_high - atr) - stop_buffer
        risk = close - stop
        if risk <= 0:
            return None
        return Signal(
            symbol=symbol, side="BUY", entry=close, stop=stop,
            target_1=close + risk * min_rr,
            target_2=close + risk * (min_rr + 1.0),
            reasons=(
                f"4H structure: {htf_structure.direction}",
                f"1H structure: {setup_structure.direction}",
                "15m bullish BOS/retest",
                f"Volume: {volume_ratio:.2f}x average",
            ),
            candle_timestamp=int(last["timestamp"]),
        )

    recent_bos = (entry["close"].tail(8) < recent_low).any()
    retest = abs(close - recent_low) <= tolerance or close < recent_low
    bearish_candle = close < float(last["open"])
    if not (recent_bos and retest and bearish_candle):
        return None

    stop = max(recent_high, recent_low + atr) + stop_buffer
    risk = stop - close
    if risk <= 0:
        return None
    return Signal(
        symbol=symbol, side="SELL", entry=close, stop=stop,
        target_1=close - risk * min_rr,
        target_2=close - risk * (min_rr + 1.0),
        reasons=(
            f"4H structure: {htf_structure.direction}",
            f"1H structure: {setup_structure.direction}",
            "15m bearish BOS/retest",
            f"Volume: {volume_ratio:.2f}x average",
        ),
        candle_timestamp=int(last["timestamp"]),
    )


def init_db() -> sqlite3.Connection:
    conn = sqlite3.connect(BASE_DIR / "signals.db")
    conn.execute("""
        CREATE TABLE IF NOT EXISTS signals (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol TEXT NOT NULL,
            side TEXT NOT NULL,
            candle_timestamp INTEGER NOT NULL,
            created_at INTEGER NOT NULL,
            UNIQUE(symbol, side, candle_timestamp)
        )
    """)
    conn.commit()
    return conn


def signal_is_new(conn: sqlite3.Connection, signal: Signal, cooldown_minutes: int) -> bool:
    row = conn.execute(
        "SELECT created_at FROM signals WHERE symbol=? AND side=? ORDER BY created_at DESC LIMIT 1",
        (signal.symbol, signal.side),
    ).fetchone()
    now = int(time.time())
    if row and now - int(row[0]) < cooldown_minutes * 60:
        return False
    try:
        conn.execute(
            "INSERT INTO signals(symbol, side, candle_timestamp, created_at) VALUES(?,?,?,?)",
            (signal.symbol, signal.side, signal.candle_timestamp, now),
        )
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        return False


def format_signal(signal: Signal) -> str:
    reasons = "\n".join(f"• {reason}" for reason in signal.reasons)
    risk = abs(signal.entry - signal.stop)
    return (
        f"📊 {signal.symbol} — {signal.side}\n\n"
        f"Entry: {signal.entry:.8g}\n"
        f"Stop: {signal.stop:.8g}\n"
        f"TP1: {signal.target_1:.8g}\n"
        f"TP2: {signal.target_2:.8g}\n"
        f"Risk distance: {risk:.8g}\n\n"
        f"{reasons}\n\n"
        "Alert-only signal. Validate liquidity, spread, and news risk before trading."
    )


def send_telegram(text: str) -> None:
    token = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
    chat_id = os.getenv("TELEGRAM_CHAT_ID", "").strip()
    if not token or not chat_id:
        log.info("\n%s", text)
        return

    response = requests.post(
        f"https://api.telegram.org/bot{token}/sendMessage",
        json={"chat_id": chat_id, "text": text},
        timeout=20,
    )
    response.raise_for_status()


def main() -> None:
    cfg = load_config()
    exchange_id = os.getenv("EXCHANGE_ID", "binance")
    interval = max(30, int(os.getenv("SCAN_INTERVAL_SECONDS", "60")))
    exchange = make_exchange(exchange_id)
    conn = init_db()

    available = [s for s in cfg["symbols"] if s in exchange.markets]
    missing = sorted(set(cfg["symbols"]) - set(available))
    if missing:
        log.warning("Unavailable on %s: %s", exchange_id, ", ".join(missing))

    log.info("Scanning %s on %s", ", ".join(available), exchange_id)

    while True:
        for symbol in available:
            try:
                signal = generate_signal(exchange, symbol, cfg)
                if signal and signal_is_new(
                    conn, signal, int(cfg["signal_cooldown_minutes"])
                ):
                    send_telegram(format_signal(signal))
                else:
                    log.info("%s: no new qualified setup", symbol)
            except (ccxt.NetworkError, ccxt.ExchangeError, requests.RequestException) as exc:
                log.warning("%s temporary API error: %s", symbol, exc)
            except Exception:
                log.exception("%s scan failed", symbol)
        time.sleep(interval)


if __name__ == "__main__":
    main()
