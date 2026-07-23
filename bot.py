from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Literal, Optional

import ccxt
import numpy as np
import pandas as pd
import requests


# =========================================================
# SETTINGS
# =========================================================

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

TREND_TIMEFRAME = "1h"
ENTRY_TIMEFRAME = "15m"

CANDLE_LIMIT = 250

EMA_FAST = 20
EMA_SLOW = 50
RSI_PERIOD = 14
ATR_PERIOD = 14
VOLUME_PERIOD = 20

# Lower number = more signals.
# Recommended range: 5 to 7.
MINIMUM_SCORE = 6

STOP_ATR_MULTIPLIER = 1.2
TARGET_1_RR = 1.5
TARGET_2_RR = 2.5

SEND_SCAN_SUMMARY = True


# =========================================================
# LOGGING
# =========================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

log = logging.getLogger("crypto-signal-bot")


# =========================================================
# DATA CLASSES
# =========================================================

@dataclass(frozen=True)
class Signal:
    symbol: str
    side: Literal["BUY", "SELL"]
    entry: float
    stop_loss: float
    target_1: float
    target_2: float
    score: int
    maximum_score: int
    rsi: float
    volume_ratio: float
    reasons: tuple[str, ...]


# =========================================================
# EXCHANGE
# =========================================================

def create_exchange():
    exchange_id = os.getenv(
        "EXCHANGE_ID",
        "kucoin",
    ).strip().lower()

    exchange_class = getattr(
        ccxt,
        exchange_id,
        None,
    )

    if exchange_class is None:
        raise ValueError(
            f"Unsupported exchange: {exchange_id}"
        )

    exchange = exchange_class(
        {
            "enableRateLimit": True,
            "timeout": 30000,
        }
    )

    exchange.load_markets()

    log.info(
        "Connected to exchange: %s",
        exchange_id,
    )

    return exchange


# =========================================================
# TELEGRAM
# =========================================================

def send_telegram(message: str) -> None:
    token = os.getenv(
        "TELEGRAM_BOT_TOKEN",
        "",
    ).strip()

    chat_id = os.getenv(
        "TELEGRAM_CHAT_ID",
        "",
    ).strip()

    if not token or not chat_id:
        log.warning(
            "Telegram secrets are missing."
        )
        log.info("\n%s", message)
        return

    url = (
        f"https://api.telegram.org/"
        f"bot{token}/sendMessage"
    )

    response = requests.post(
        url,
        json={
            "chat_id": chat_id,
            "text": message,
            "disable_web_page_preview": True,
        },
        timeout=20,
    )

    response.raise_for_status()


# =========================================================
# MARKET DATA
# =========================================================

def fetch_candles(
    exchange,
    symbol: str,
    timeframe: str,
) -> pd.DataFrame:
    candles = exchange.fetch_ohlcv(
        symbol=symbol,
        timeframe=timeframe,
        limit=CANDLE_LIMIT,
    )

    if len(candles) < 100:
        raise RuntimeError(
            f"Not enough candles for {symbol} {timeframe}"
        )

    dataframe = pd.DataFrame(
        candles,
        columns=[
            "timestamp",
            "open",
            "high",
            "low",
            "close",
            "volume",
        ],
    )

    numeric_columns = [
        "open",
        "high",
        "low",
        "close",
        "volume",
    ]

    dataframe[numeric_columns] = (
        dataframe[numeric_columns]
        .astype(float)
    )

    dataframe["timestamp"] = (
        dataframe["timestamp"]
        .astype("int64")
    )

    # Ignore the currently forming candle.
    return dataframe.iloc[:-1].copy()


# =========================================================
# INDICATORS
# =========================================================

def add_indicators(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    output = dataframe.copy()

    output["ema_fast"] = (
        output["close"]
        .ewm(
            span=EMA_FAST,
            adjust=False,
        )
        .mean()
    )

    output["ema_slow"] = (
        output["close"]
        .ewm(
            span=EMA_SLOW,
            adjust=False,
        )
        .mean()
    )

    price_change = output["close"].diff()

    gain = price_change.clip(lower=0)
    loss = -price_change.clip(upper=0)

    average_gain = gain.ewm(
        alpha=1 / RSI_PERIOD,
        adjust=False,
        min_periods=RSI_PERIOD,
    ).mean()

    average_loss = loss.ewm(
        alpha=1 / RSI_PERIOD,
        adjust=False,
        min_periods=RSI_PERIOD,
    ).mean()

    relative_strength = (
        average_gain
        / average_loss.replace(0, np.nan)
    )

    output["rsi"] = (
        100
        - (
            100
            / (1 + relative_strength)
        )
    )

    previous_close = output["close"].shift(1)

    true_range = pd.concat(
        [
            output["high"] - output["low"],
            (
                output["high"]
                - previous_close
            ).abs(),
            (
                output["low"]
                - previous_close
            ).abs(),
        ],
        axis=1,
    ).max(axis=1)

    output["atr"] = (
        true_range
        .rolling(ATR_PERIOD)
        .mean()
    )

    output["volume_average"] = (
        output["volume"]
        .rolling(VOLUME_PERIOD)
        .mean()
    )

    output["previous_20_high"] = (
        output["high"]
        .shift(1)
        .rolling(20)
        .max()
    )

    output["previous_20_low"] = (
        output["low"]
        .shift(1)
        .rolling(20)
        .min()
    )

    return output


# =========================================================
# SIGNAL GENERATION
# =========================================================

def generate_signal(
    exchange,
    symbol: str,
) -> Optional[Signal]:
    trend_data = add_indicators(
        fetch_candles(
            exchange,
            symbol,
            TREND_TIMEFRAME,
        )
    )

    entry_data = add_indicators(
        fetch_candles(
            exchange,
            symbol,
            ENTRY_TIMEFRAME,
        )
    )

    trend_candle = trend_data.iloc[-1]
    entry_candle = entry_data.iloc[-1]
    previous_entry = entry_data.iloc[-2]

    required_values = [
        trend_candle["ema_fast"],
        trend_candle["ema_slow"],
        trend_candle["rsi"],
        entry_candle["ema_fast"],
        entry_candle["ema_slow"],
        entry_candle["rsi"],
        entry_candle["atr"],
        entry_candle["volume_average"],
    ]

    if not all(
        np.isfinite(value)
        for value in required_values
    ):
        return None

    entry_price = float(
        entry_candle["close"]
    )

    atr = float(
        entry_candle["atr"]
    )

    rsi = float(
        entry_candle["rsi"]
    )

    average_volume = float(
        entry_candle["volume_average"]
    )

    if atr <= 0 or average_volume <= 0:
        return None

    volume_ratio = (
        float(entry_candle["volume"])
        / average_volume
    )

    long_score = 0
    short_score = 0

    long_reasons: list[str] = []
    short_reasons: list[str] = []

    maximum_score = 8

    # -----------------------------------------------------
    # 1-HOUR TREND
    # -----------------------------------------------------

    if (
        trend_candle["ema_fast"]
        > trend_candle["ema_slow"]
    ):
        long_score += 2
        long_reasons.append(
            "1H EMA20 is above EMA50"
        )

    if (
        trend_candle["ema_fast"]
        < trend_candle["ema_slow"]
    ):
        short_score += 2
        short_reasons.append(
            "1H EMA20 is below EMA50"
        )

    if (
        trend_candle["close"]
        > trend_candle["ema_fast"]
    ):
        long_score += 1
        long_reasons.append(
            "1H price is above EMA20"
        )

    if (
        trend_candle["close"]
        < trend_candle["ema_fast"]
    ):
        short_score += 1
        short_reasons.append(
            "1H price is below EMA20"
        )

    # -----------------------------------------------------
    # 15-MINUTE TREND
    # -----------------------------------------------------

    if (
        entry_candle["ema_fast"]
        > entry_candle["ema_slow"]
    ):
        long_score += 1
        long_reasons.append(
            "15m EMA trend is bullish"
        )

    if (
        entry_candle["ema_fast"]
        < entry_candle["ema_slow"]
    ):
        short_score += 1
        short_reasons.append(
            "15m EMA trend is bearish"
        )

    # -----------------------------------------------------
    # RSI
    # -----------------------------------------------------

    if 48 <= rsi <= 72:
        long_score += 1
        long_reasons.append(
            f"15m RSI supports buying: {rsi:.1f}"
        )

    if 28 <= rsi <= 52:
        short_score += 1
        short_reasons.append(
            f"15m RSI supports selling: {rsi:.1f}"
        )

    # -----------------------------------------------------
    # CANDLE DIRECTION
    # -----------------------------------------------------

    bullish_candle = (
        entry_candle["close"]
        > entry_candle["open"]
    )

    bearish_candle = (
        entry_candle["close"]
        < entry_candle["open"]
    )

    if bullish_candle:
        long_score += 1
        long_reasons.append(
            "15m candle closed bullish"
        )

    if bearish_candle:
        short_score += 1
        short_reasons.append(
            "15m candle closed bearish"
        )

    # -----------------------------------------------------
    # VOLUME
    # -----------------------------------------------------

    if volume_ratio >= 0.80:
        long_score += 1
        short_score += 1

        volume_reason = (
            f"Volume is {volume_ratio:.2f}x average"
        )

        long_reasons.append(volume_reason)
        short_reasons.append(volume_reason)

    # -----------------------------------------------------
    # PULLBACK / RECLAIM
    # -----------------------------------------------------

    bullish_reclaim = (
        previous_entry["low"]
        <= previous_entry["ema_fast"]
        and entry_candle["close"]
        > entry_candle["ema_fast"]
    )

    bearish_rejection = (
        previous_entry["high"]
        >= previous_entry["ema_fast"]
        and entry_candle["close"]
        < entry_candle["ema_fast"]
    )

    if bullish_reclaim:
        long_score += 1
        long_reasons.append(
            "Price reclaimed 15m EMA20"
        )

    if bearish_rejection:
        short_score += 1
        short_reasons.append(
            "Price rejected 15m EMA20"
        )

    # -----------------------------------------------------
    # BREAKOUT
    # -----------------------------------------------------

    previous_high = float(
        entry_candle["previous_20_high"]
    )

    previous_low = float(
        entry_candle["previous_20_low"]
    )

    if (
        np.isfinite(previous_high)
        and entry_price > previous_high
    ):
        long_score += 1
        long_reasons.append(
            "Price broke the previous 20-candle high"
        )

    if (
        np.isfinite(previous_low)
        and entry_price < previous_low
    ):
        short_score += 1
        short_reasons.append(
            "Price broke the previous 20-candle low"
        )

    log.info(
        "%s | BUY score %s | SELL score %s | RSI %.1f | Volume %.2fx",
        symbol,
        long_score,
        short_score,
        rsi,
        volume_ratio,
    )

    # -----------------------------------------------------
    # CHOOSE DIRECTION
    # -----------------------------------------------------

    if (
        long_score >= MINIMUM_SCORE
        and long_score > short_score
    ):
        stop_loss = (
            entry_price
            - atr * STOP_ATR_MULTIPLIER
        )

        risk = entry_price - stop_loss

        target_1 = (
            entry_price
            + risk * TARGET_1_RR
        )

        target_2 = (
            entry_price
            + risk * TARGET_2_RR
        )

        return Signal(
            symbol=symbol,
            side="BUY",
            entry=entry_price,
            stop_loss=stop_loss,
            target_1=target_1,
            target_2=target_2,
            score=long_score,
            maximum_score=maximum_score,
            rsi=rsi,
            volume_ratio=volume_ratio,
            reasons=tuple(long_reasons),
        )

    if (
        short_score >= MINIMUM_SCORE
        and short_score > long_score
    ):
        stop_loss = (
            entry_price
            + atr * STOP_ATR_MULTIPLIER
        )

        risk = stop_loss - entry_price

        target_1 = (
            entry_price
            - risk * TARGET_1_RR
        )

        target_2 = (
            entry_price
            - risk * TARGET_2_RR
        )

        return Signal(
            symbol=symbol,
            side="SELL",
            entry=entry_price,
            stop_loss=stop_loss,
            target_1=target_1,
            target_2=target_2,
            score=short_score,
            maximum_score=maximum_score,
            rsi=rsi,
            volume_ratio=volume_ratio,
            reasons=tuple(short_reasons),
        )

    return None


# =========================================================
# MESSAGE FORMAT
# =========================================================

def format_price(price: float) -> str:
    if price >= 1000:
        return f"{price:,.2f}"

    if price >= 1:
        return f"{price:.4f}"

    return f"{price:.7f}"


def format_signal(signal: Signal) -> str:
    reasons = "\n".join(
        f"✅ {reason}"
        for reason in signal.reasons
    )

    confidence = round(
        (
            signal.score
            / signal.maximum_score
        )
        * 100
    )

    return (
        f"🚨 CRYPTO SIGNAL\n\n"
        f"Pair: {signal.symbol}\n"
        f"Direction: {signal.side}\n"
        f"Score: {signal.score}/{signal.maximum_score}\n"
        f"Confidence score: {confidence}%\n\n"
        f"Entry: {format_price(signal.entry)}\n"
        f"Stop Loss: {format_price(signal.stop_loss)}\n"
        f"Target 1: {format_price(signal.target_1)}\n"
        f"Target 2: {format_price(signal.target_2)}\n\n"
        f"RSI: {signal.rsi:.1f}\n"
        f"Volume: {signal.volume_ratio:.2f}x average\n\n"
        f"{reasons}\n\n"
        "⚠️ Check the chart before entering. "
        "Signals are not guaranteed."
    )


# =========================================================
# MAIN
# =========================================================

def main() -> None:
    exchange = create_exchange()

    available_symbols = [
        symbol
        for symbol in SYMBOLS
        if symbol in exchange.markets
    ]

    unavailable_symbols = [
        symbol
        for symbol in SYMBOLS
        if symbol not in exchange.markets
    ]

    if unavailable_symbols:
        log.warning(
            "Unavailable symbols: %s",
            ", ".join(unavailable_symbols),
        )

    log.info(
        "Starting scan for %s symbols",
        len(available_symbols),
    )

    signals: list[Signal] = []
    errors: list[str] = []

    for symbol in available_symbols:
        try:
            log.info(
                "Scanning %s",
                symbol,
            )

            signal = generate_signal(
                exchange,
                symbol,
            )

            if signal is not None:
                signals.append(signal)

                send_telegram(
                    format_signal(signal)
                )

                log.info(
                    "%s signal sent for %s",
                    signal.side,
                    symbol,
                )

            else:
                log.info(
                    "%s | No qualified signal",
                    symbol,
                )

        except (
            ccxt.NetworkError,
            ccxt.ExchangeError,
            requests.RequestException,
        ) as error:
            error_message = (
                f"{symbol}: API error"
            )

            errors.append(error_message)

            log.warning(
                "%s | %s",
                symbol,
                error,
            )

        except Exception as error:
            error_message = (
                f"{symbol}: {type(error).__name__}"
            )

            errors.append(error_message)

            log.exception(
                "%s scan failed",
                symbol,
            )

    log.info(
        "Scan complete. Signals found: %s",
        len(signals),
    )

    if SEND_SCAN_SUMMARY:
        summary = (
            "✅ Crypto scan completed\n\n"
            f"Pairs checked: {len(available_symbols)}\n"
            f"Signals found: {len(signals)}\n"
            f"Errors: {len(errors)}\n\n"
        )

        if signals:
            signal_names = "\n".join(
                f"• {signal.symbol}: {signal.side} "
                f"({signal.score}/{signal.maximum_score})"
                for signal in signals
            )

            summary += (
                "Signals:\n"
                f"{signal_names}"
            )

        else:
            summary += (
                "No qualified setup during this scan. "
                "The bot will check again automatically."
            )

        send_telegram(summary)


if __name__ == "__main__":
    main()
