from __future__ import annotations

import json
import logging
import math
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional

import ccxt
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests

BASE_DIR = Path(__file__).resolve().parent
STATE_FILE = BASE_DIR / "state.json"
PERFORMANCE_FILE = BASE_DIR / "performance.json"

SYMBOLS = [
    "BTC/USDT",
    "ETH/USDT",
    "SOL/USDT",
    "BNB/USDT",
    "XRP/USDT",
    "DOGE/USDT",
    "LINK/USDT",
    "SUI/USDT",
]

BTC_SYMBOL = "BTC/USDT"

HTF = "4h"
MTF = "1h"
LTF = "15m"
CANDLE_LIMIT = 320

EMA_FAST = 20
EMA_SLOW = 50
RSI_PERIOD = 14
ATR_PERIOD = 14
ADX_PERIOD = 14
VOLUME_PERIOD = 20

MIN_ADX = 20.0
MIN_ATR_PERCENT = 0.25
MIN_VOLUME_RATIO = 0.90
MIN_CONFIDENCE = 78

STOP_ATR = 1.25
TP1_RR = 1.5
TP2_RR = 2.0
TP3_RR = 3.0

BOS_LOOKBACK = 20
SWING_WINDOW = 3
SWEEP_LOOKBACK = 14
SR_LOOKBACK = 80
FVG_LOOKBACK = 30
OB_LOOKBACK = 25
MAX_SR_DISTANCE_ATR = 1.25

SEND_EMPTY_SUMMARY = False

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
log = logging.getLogger("v35-institutional")


@dataclass(frozen=True)
class Signal:
    symbol: str
    side: Literal["BUY", "SELL"]
    grade: str
    confidence: int
    entry: float
    stop_loss: float
    target_1: float
    target_2: float
    target_3: float
    rsi: float
    adx: float
    atr_percent: float
    volume_ratio: float
    candle_timestamp: int
    bos: bool
    choch: bool
    sweep: bool
    fvg: bool
    order_block: bool
    reasons: tuple[str, ...]


def create_exchange():
    exchange_id = os.getenv("EXCHANGE_ID", "kucoin").strip().lower()
    exchange_class = getattr(ccxt, exchange_id, None)
    if exchange_class is None:
        raise ValueError(f"Unsupported exchange: {exchange_id}")

    exchange = exchange_class({
        "enableRateLimit": True,
        "timeout": 30000,
    })
    exchange.load_markets()
    log.info("Connected to %s", exchange_id)
    return exchange


def fetch_candles(exchange, symbol: str, timeframe: str) -> pd.DataFrame:
    rows = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=CANDLE_LIMIT)
    if len(rows) < 120:
        raise RuntimeError(f"Insufficient candles: {symbol} {timeframe}")

    df = pd.DataFrame(
        rows,
        columns=["timestamp", "open", "high", "low", "close", "volume"],
    )
    numeric_cols = ["open", "high", "low", "close", "volume"]
    df[numeric_cols] = df[numeric_cols].astype(float)
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

    previous_close = out["close"].shift(1)
    tr = pd.concat(
        [
            out["high"] - out["low"],
            (out["high"] - previous_close).abs(),
            (out["low"] - previous_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    out["atr"] = tr.ewm(alpha=1 / ATR_PERIOD, adjust=False, min_periods=ATR_PERIOD).mean()

    up_move = out["high"].diff()
    down_move = -out["low"].diff()

    plus_dm = pd.Series(
        np.where((up_move > down_move) & (up_move > 0), up_move, 0.0),
        index=out.index,
    )
    minus_dm = pd.Series(
        np.where((down_move > up_move) & (down_move > 0), down_move, 0.0),
        index=out.index,
    )

    atr_safe = out["atr"].replace(0, np.nan)
    plus_di = 100 * plus_dm.ewm(
        alpha=1 / ADX_PERIOD, adjust=False, min_periods=ADX_PERIOD
    ).mean() / atr_safe
    minus_di = 100 * minus_dm.ewm(
        alpha=1 / ADX_PERIOD, adjust=False, min_periods=ADX_PERIOD
    ).mean() / atr_safe

    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    out["adx"] = dx.ewm(
        alpha=1 / ADX_PERIOD, adjust=False, min_periods=ADX_PERIOD
    ).mean()

    out["volume_average"] = out["volume"].rolling(VOLUME_PERIOD).mean()
    out["prior_high"] = out["high"].shift(1).rolling(BOS_LOOKBACK).max()
    out["prior_low"] = out["low"].shift(1).rolling(BOS_LOOKBACK).min()

    return out


def trend_direction(df: pd.DataFrame) -> Literal["bullish", "bearish", "neutral"]:
    row = df.iloc[-1]
    close = float(row["close"])
    ema20 = float(row["ema20"])
    ema50 = float(row["ema50"])

    if close > ema20 > ema50:
        return "bullish"
    if close < ema20 < ema50:
        return "bearish"
    return "neutral"


def swing_points(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    highs = out["high"]
    lows = out["low"]

    out["swing_high"] = highs.where(
        (highs > highs.shift(1))
        & (highs > highs.shift(2))
        & (highs > highs.shift(3))
        & (highs >= highs.shift(-1))
        & (highs >= highs.shift(-2))
        & (highs >= highs.shift(-3))
    )

    out["swing_low"] = lows.where(
        (lows < lows.shift(1))
        & (lows < lows.shift(2))
        & (lows < lows.shift(3))
        & (lows <= lows.shift(-1))
        & (lows <= lows.shift(-2))
        & (lows <= lows.shift(-3))
    )
    return out


def detect_bos_choch(df: pd.DataFrame, side: str) -> tuple[bool, bool, float]:
    structured = swing_points(df)
    recent = structured.tail(80)
    swing_highs = recent["swing_high"].dropna()
    swing_lows = recent["swing_low"].dropna()

    if len(swing_highs) < 2 or len(swing_lows) < 2:
        return False, False, float("nan")

    close = float(recent.iloc[-1]["close"])
    previous_close = float(recent.iloc[-2]["close"])

    latest_high = float(swing_highs.iloc[-1])
    latest_low = float(swing_lows.iloc[-1])

    previous_high = float(swing_highs.iloc[-2])
    previous_low = float(swing_lows.iloc[-2])

    prior_structure_bullish = latest_high > previous_high and latest_low > previous_low
    prior_structure_bearish = latest_high < previous_high and latest_low < previous_low

    if side == "BUY":
        bos = previous_close <= latest_high and close > latest_high
        choch = prior_structure_bearish and close > latest_high
        return bos, choch, latest_high

    bos = previous_close >= latest_low and close < latest_low
    choch = prior_structure_bullish and close < latest_low
    return bos, choch, latest_low


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


def detect_fvg(df: pd.DataFrame, side: str) -> tuple[bool, tuple[float, float] | None]:
    recent = df.tail(FVG_LOOKBACK).reset_index(drop=True)

    for i in range(len(recent) - 3, 1, -1):
        candle_1 = recent.iloc[i - 2]
        candle_3 = recent.iloc[i]

        if side == "BUY" and candle_3["low"] > candle_1["high"]:
            zone = (float(candle_1["high"]), float(candle_3["low"]))
            close = float(recent.iloc[-1]["close"])
            touched = float(recent.iloc[-1]["low"]) <= zone[1]
            respected = close >= zone[0]
            return bool(touched and respected), zone

        if side == "SELL" and candle_3["high"] < candle_1["low"]:
            zone = (float(candle_3["high"]), float(candle_1["low"]))
            close = float(recent.iloc[-1]["close"])
            touched = float(recent.iloc[-1]["high"]) >= zone[0]
            respected = close <= zone[1]
            return bool(touched and respected), zone

    return False, None


def detect_order_block(df: pd.DataFrame, side: str) -> tuple[bool, tuple[float, float] | None]:
    recent = df.tail(OB_LOOKBACK).reset_index(drop=True)
    current = recent.iloc[-1]

    for i in range(len(recent) - 2, 1, -1):
        candle = recent.iloc[i - 1]
        impulse = recent.iloc[i]

        if side == "BUY":
            bearish_candle = candle["close"] < candle["open"]
            bullish_impulse = impulse["close"] > impulse["open"]
            strong_move = (impulse["close"] - impulse["open"]) > 0.8 * float(current["atr"])
            if bearish_candle and bullish_impulse and strong_move:
                zone = (float(candle["low"]), float(candle["open"]))
                touched = float(current["low"]) <= zone[1]
                respected = float(current["close"]) > zone[0]
                return bool(touched and respected), zone

        if side == "SELL":
            bullish_candle = candle["close"] > candle["open"]
            bearish_impulse = impulse["close"] < impulse["open"]
            strong_move = (impulse["open"] - impulse["close"]) > 0.8 * float(current["atr"])
            if bullish_candle and bearish_impulse and strong_move:
                zone = (float(candle["open"]), float(candle["high"]))
                touched = float(current["high"]) >= zone[0]
                respected = float(current["close"]) < zone[1]
                return bool(touched and respected), zone

    return False, None


def nearby_support_resistance(df: pd.DataFrame, side: str, atr: float) -> bool:
    recent = df.tail(SR_LOOKBACK)
    close = float(recent.iloc[-1]["close"])

    if side == "BUY":
        resistance = float(recent.iloc[:-1]["high"].quantile(0.95))
        return 0 < resistance - close <= MAX_SR_DISTANCE_ATR * atr

    support = float(recent.iloc[:-1]["low"].quantile(0.05))
    return 0 < close - support <= MAX_SR_DISTANCE_ATR * atr


def grade(confidence: int) -> str:
    if confidence >= 90:
        return "A+"
    if confidence >= 82:
        return "A"
    return "B+"


def calculate_confidence(
    *,
    market_regime_match: bool,
    h4_match: bool,
    h1_match: bool,
    adx_ok: bool,
    atr_ok: bool,
    volume_ok: bool,
    rsi_ok: bool,
    bos: bool,
    choch: bool,
    sweep: bool,
    fvg: bool,
    order_block: bool,
    nearby_sr: bool,
) -> int:
    score = 0
    score += 12 if market_regime_match else 0
    score += 13 if h4_match else 0
    score += 12 if h1_match else 0
    score += 7 if adx_ok else 0
    score += 5 if atr_ok else 0
    score += 7 if volume_ok else 0
    score += 6 if rsi_ok else 0
    score += 10 if bos else 0
    score += 10 if choch else 0
    score += 7 if sweep else 0
    score += 6 if fvg else 0
    score += 7 if order_block else 0
    score -= 12 if nearby_sr else 0
    return max(0, min(100, score))


def build_signal(exchange, symbol: str, btc_trend: str) -> Optional[tuple[Signal, pd.DataFrame]]:
    h4 = add_indicators(fetch_candles(exchange, symbol, HTF))
    h1 = add_indicators(fetch_candles(exchange, symbol, MTF))
    m15 = add_indicators(fetch_candles(exchange, symbol, LTF))

    h4_trend = trend_direction(h4)
    h1_trend = trend_direction(h1)

    row = m15.iloc[-1]
    close = float(row["close"])
    atr = float(row["atr"])
    rsi = float(row["rsi"])
    adx = float(row["adx"])
    vol_avg = float(row["volume_average"])

    if not all(math.isfinite(v) for v in [close, atr, rsi, adx, vol_avg]):
        return None
    if close <= 0 or atr <= 0 or vol_avg <= 0:
        return None

    volume_ratio = float(row["volume"]) / vol_avg
    atr_percent = (atr / close) * 100

    candidates = []

    for side in ("BUY", "SELL"):
        bullish = side == "BUY"
        direction = "bullish" if bullish else "bearish"

        market_regime_match = symbol == BTC_SYMBOL or btc_trend == direction
        h4_match = h4_trend == direction
        h1_match = h1_trend == direction
        adx_ok = adx >= MIN_ADX
        atr_ok = atr_percent >= MIN_ATR_PERCENT
        volume_ok = volume_ratio >= MIN_VOLUME_RATIO
        rsi_ok = (50 <= rsi <= 70) if bullish else (30 <= rsi <= 50)

        bos, choch, structure_level = detect_bos_choch(m15, side)
        sweep = detect_liquidity_sweep(m15, side)
        fvg, fvg_zone = detect_fvg(m15, side)
        order_block, ob_zone = detect_order_block(m15, side)
        nearby_sr = nearby_support_resistance(m15, side, atr)

        confidence = calculate_confidence(
            market_regime_match=market_regime_match,
            h4_match=h4_match,
            h1_match=h1_match,
            adx_ok=adx_ok,
            atr_ok=atr_ok,
            volume_ok=volume_ok,
            rsi_ok=rsi_ok,
            bos=bos,
            choch=choch,
            sweep=sweep,
            fvg=fvg,
            order_block=order_block,
            nearby_sr=nearby_sr,
        )

        mandatory = (
            market_regime_match
            and h4_match
            and h1_match
            and adx_ok
            and atr_ok
            and (bos or choch or sweep)
            and (fvg or order_block or volume_ok)
            and not nearby_sr
        )

        reasons = []
        if market_regime_match:
            reasons.append(f"BTC 4H regime supports {side}")
        if h4_match:
            reasons.append(f"Coin 4H trend supports {side}")
        if h1_match:
            reasons.append(f"Coin 1H trend supports {side}")
        if bos:
            reasons.append("15m Break of Structure")
        if choch:
            reasons.append("15m Change of Character")
        if sweep:
            reasons.append("Liquidity sweep confirmed")
        if fvg:
            reasons.append("Fair Value Gap reaction")
        if order_block:
            reasons.append("Order Block reaction")
        if volume_ok:
            reasons.append(f"Volume {volume_ratio:.2f}x average")
        if rsi_ok:
            reasons.append(f"RSI momentum {rsi:.1f}")
        if adx_ok:
            reasons.append(f"ADX trend strength {adx:.1f}")
        if atr_ok:
            reasons.append(f"ATR volatility {atr_percent:.2f}%")

        log.info(
            "%s %s | confidence=%s | BTC=%s 4H=%s 1H=%s BOS=%s CHOCH=%s SWEEP=%s FVG=%s OB=%s SR=%s",
            symbol,
            side,
            confidence,
            btc_trend,
            h4_trend,
            h1_trend,
            bos,
            choch,
            sweep,
            fvg,
            order_block,
            nearby_sr,
        )

        if mandatory and confidence >= MIN_CONFIDENCE:
            candidates.append(
                (
                    confidence,
                    side,
                    bos,
                    choch,
                    sweep,
                    fvg,
                    order_block,
                    tuple(reasons),
                    structure_level,
                    fvg_zone,
                    ob_zone,
                )
            )

    if not candidates:
        return None

    candidates.sort(key=lambda item: item[0], reverse=True)
    (
        confidence,
        side,
        bos,
        choch,
        sweep,
        fvg,
        order_block,
        reasons,
        structure_level,
        fvg_zone,
        ob_zone,
    ) = candidates[0]

    if side == "BUY":
        structural_stop = float(m15.tail(SWEEP_LOOKBACK)["low"].min())
        stop_loss = min(structural_stop, close - atr * STOP_ATR)
        risk = close - stop_loss
        if risk <= 0:
            return None
        tp1 = close + risk * TP1_RR
        tp2 = close + risk * TP2_RR
        tp3 = close + risk * TP3_RR
    else:
        structural_stop = float(m15.tail(SWEEP_LOOKBACK)["high"].max())
        stop_loss = max(structural_stop, close + atr * STOP_ATR)
        risk = stop_loss - close
        if risk <= 0:
            return None
        tp1 = close - risk * TP1_RR
        tp2 = close - risk * TP2_RR
        tp3 = close - risk * TP3_RR

    signal = Signal(
        symbol=symbol,
        side=side,
        grade=grade(confidence),
        confidence=confidence,
        entry=close,
        stop_loss=stop_loss,
        target_1=tp1,
        target_2=tp2,
        target_3=tp3,
        rsi=rsi,
        adx=adx,
        atr_percent=atr_percent,
        volume_ratio=volume_ratio,
        candle_timestamp=int(row["timestamp"]),
        bos=bos,
        choch=choch,
        sweep=sweep,
        fvg=fvg,
        order_block=order_block,
        reasons=reasons,
    )
    return signal, m15


def load_json(path: Path, fallback: dict) -> dict:
    if not path.exists():
        return fallback
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return fallback


def save_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def signal_key(signal: Signal) -> str:
    return f"{signal.symbol}|{signal.side}|{signal.candle_timestamp}"


def already_sent(signal: Signal, state: dict) -> bool:
    return signal_key(signal) in state.get("sent", {})


def remember_signal(signal: Signal, state: dict) -> None:
    state.setdefault("sent", {})[signal_key(signal)] = {
        "symbol": signal.symbol,
        "side": signal.side,
        "timestamp": signal.candle_timestamp,
        "confidence": signal.confidence,
    }
    entries = list(state["sent"].items())
    if len(entries) > 400:
        state["sent"] = dict(entries[-400:])


def record_open_signal(signal: Signal, performance: dict) -> None:
    performance.setdefault("open_signals", {})[signal_key(signal)] = {
        "symbol": signal.symbol,
        "side": signal.side,
        "entry": signal.entry,
        "stop_loss": signal.stop_loss,
        "target_1": signal.target_1,
        "target_2": signal.target_2,
        "target_3": signal.target_3,
        "timestamp": signal.candle_timestamp,
        "confidence": signal.confidence,
    }


def format_price(value: float) -> str:
    if value >= 1000:
        return f"{value:,.2f}"
    if value >= 1:
        return f"{value:.4f}"
    return f"{value:.7f}"


def format_signal(signal: Signal) -> str:
    reasons = "\n".join(f"✅ {reason}" for reason in signal.reasons)
    return (
        f"🏛️ V3.5 INSTITUTIONAL {signal.grade}\n\n"
        f"Pair: {signal.symbol}\n"
        f"Direction: {signal.side}\n"
        f"Confidence: {signal.confidence}%\n\n"
        f"Entry: {format_price(signal.entry)}\n"
        f"Stop Loss: {format_price(signal.stop_loss)}\n"
        f"TP1 (1.5R): {format_price(signal.target_1)}\n"
        f"TP2 (2R): {format_price(signal.target_2)}\n"
        f"TP3 (3R): {format_price(signal.target_3)}\n\n"
        f"RSI: {signal.rsi:.1f}\n"
        f"ADX: {signal.adx:.1f}\n"
        f"Volume: {signal.volume_ratio:.2f}x\n"
        f"ATR: {signal.atr_percent:.2f}%\n\n"
        f"{reasons}\n\n"
        "⚠️ Alert only. This is not guaranteed and is not financial advice."
    )


def telegram_credentials() -> tuple[str, str]:
    token = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
    chat_id = os.getenv("TELEGRAM_CHAT_ID", "").strip()
    if not token or not chat_id:
        raise RuntimeError("Telegram secrets are missing")
    return token, chat_id


def send_telegram_text(message: str) -> None:
    token, chat_id = telegram_credentials()
    response = requests.post(
        f"https://api.telegram.org/bot{token}/sendMessage",
        json={
            "chat_id": chat_id,
            "text": message,
            "disable_web_page_preview": True,
        },
        timeout=25,
    )
    response.raise_for_status()


def create_chart(df: pd.DataFrame, signal: Signal) -> str:
    chart_df = df.tail(100).copy()
    x = np.arange(len(chart_df))

    fig, ax = plt.subplots(figsize=(12, 7))

    for i, row in enumerate(chart_df.itertuples()):
        open_price = row.open
        close_price = row.close
        high = row.high
        low = row.low

        ax.vlines(i, low, high, linewidth=1)
        bottom = min(open_price, close_price)
        height = max(abs(close_price - open_price), signal.entry * 0.00005)
        ax.add_patch(
            plt.Rectangle(
                (i - 0.3, bottom),
                0.6,
                height,
                fill=False,
                linewidth=1,
            )
        )

    ax.plot(x, chart_df["ema20"], linewidth=1.2, label="EMA20")
    ax.plot(x, chart_df["ema50"], linewidth=1.2, label="EMA50")

    ax.axhline(signal.entry, linestyle="--", linewidth=1.2, label="Entry")
    ax.axhline(signal.stop_loss, linestyle="--", linewidth=1.2, label="Stop")
    ax.axhline(signal.target_1, linestyle=":", linewidth=1.0, label="TP1")
    ax.axhline(signal.target_2, linestyle=":", linewidth=1.0, label="TP2")
    ax.axhline(signal.target_3, linestyle=":", linewidth=1.0, label="TP3")

    ax.set_title(
        f"{signal.symbol} {signal.side} | Confidence {signal.confidence}% | 15m"
    )
    ax.set_xlabel("Closed candles")
    ax.set_ylabel("Price")
    ax.legend()
    ax.grid(True, alpha=0.25)
    fig.tight_layout()

    path = Path(tempfile.gettempdir()) / f"{signal.symbol.replace('/', '_')}_{signal.side}.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return str(path)


def send_telegram_photo(photo_path: str, caption: str) -> None:
    token, chat_id = telegram_credentials()
    with open(photo_path, "rb") as photo:
        response = requests.post(
            f"https://api.telegram.org/bot{token}/sendPhoto",
            data={
                "chat_id": chat_id,
                "caption": caption[:1024],
            },
            files={"photo": photo},
            timeout=40,
        )
    response.raise_for_status()


def main() -> None:
    exchange = create_exchange()
    state = load_json(STATE_FILE, {"sent": {}})
    performance = load_json(
        PERFORMANCE_FILE,
        {
            "open_signals": {},
            "closed_signals": [],
        },
    )

    available = [symbol for symbol in SYMBOLS if symbol in exchange.markets]
    missing = [symbol for symbol in SYMBOLS if symbol not in exchange.markets]
    if missing:
        log.warning("Unavailable symbols: %s", ", ".join(missing))

    btc_4h = add_indicators(fetch_candles(exchange, BTC_SYMBOL, HTF))
    btc_trend = trend_direction(btc_4h)
    log.info("BTC 4H regime: %s", btc_trend)

    new_signals: list[Signal] = []
    errors: list[str] = []

    for symbol in available:
        try:
            result = build_signal(exchange, symbol, btc_trend)
            if result is None:
                continue

            signal, chart_df = result
            if already_sent(signal, state):
                log.info("%s duplicate signal suppressed", symbol)
                continue

            message = format_signal(signal)
            send_telegram_text(message)

            chart_path = create_chart(chart_df, signal)
            send_telegram_photo(
                chart_path,
                f"{signal.symbol} {signal.side} | {signal.grade} | {signal.confidence}%",
            )

            remember_signal(signal, state)
            record_open_signal(signal, performance)
            new_signals.append(signal)
            log.info("%s %s alert sent", signal.symbol, signal.side)

        except (
            ccxt.NetworkError,
            ccxt.ExchangeError,
            requests.RequestException,
        ) as exc:
            errors.append(f"{symbol}: API error")
            log.warning("%s API error: %s", symbol, exc)
        except Exception as exc:
            errors.append(f"{symbol}: {type(exc).__name__}")
            log.exception("%s scan failed", symbol)

    save_json(STATE_FILE, state)
    save_json(PERFORMANCE_FILE, performance)

    summary = (
        "✅ V3.5 Institutional scan completed\n\n"
        f"BTC 4H regime: {btc_trend.upper()}\n"
        f"Pairs checked: {len(available)}\n"
        f"New signals: {len(new_signals)}\n"
        f"Errors: {len(errors)}"
    )

    if new_signals:
        summary += "\n\n" + "\n".join(
            f"• {signal.symbol}: {signal.side} {signal.grade} ({signal.confidence}%)"
            for signal in new_signals
        )
        send_telegram_text(summary)
    elif SEND_EMPTY_SUMMARY:
        send_telegram_text(summary)

    log.info(summary.replace("\n", " | "))


if __name__ == "__main__":
    main()
