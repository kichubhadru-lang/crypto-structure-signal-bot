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

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

log = logging.getLogger("crypto-structure-bot")


@dataclass(frozen=True)
class MarketStructure:
    direction: Literal["bullish", "bearish", "range", "unknown"]
    latest_high: Optional[float]
    previous_high: Optional[float]
    latest_low: Optional[float]
    previous_low: Optional[float]


@dataclass(frozen=True)
class Signal:
    symbol: str
    side: Literal["BUY", "SELL"]
    entry: float
    stop_loss: float
    target_1: float
    target_2: float
    candle_timestamp: int
    reasons: tuple[str, ...]


def load_config() -> dict:
    config_path = BASE_DIR / "config.json"

    if not config_path.exists():
        raise FileNotFoundError("config.json was not found")

    with config_path.open("r", encoding="utf-8") as file:
        return json.load(file)


def create_exchange(exchange_id: str):
    exchange_class = getattr(ccxt, exchange_id, None)

    if exchange_class is None:
        raise ValueError(f"Unsupported exchange: {exchange_id}")

    exchange = exchange_class(
        {
            "enableRateLimit": True,
            "timeout": 30000,
        }
    )

    exchange.load_markets()
    return exchange


def fetch_candles(
    exchange,
    symbol: str,
    timeframe: str,
    limit: int,
) -> pd.DataFrame:
    candles = exchange.fetch_ohlcv(
        symbol,
        timeframe=timeframe,
        limit=limit,
    )

    if len(candles) < 100:
        raise RuntimeError(
            f"Not enough candle data for {symbol} on {timeframe}"
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

    dataframe[numeric_columns] = dataframe[numeric_columns].astype(float)
    dataframe["timestamp"] = dataframe["timestamp"].astype("int64")

    # Remove the currently forming candle.
    return dataframe.iloc[:-1].copy()


def add_atr(
    dataframe: pd.DataFrame,
    period: int,
) -> pd.DataFrame:
    output = dataframe.copy()

    previous_close = output["close"].shift(1)

    true_range = pd.concat(
        [
            output["high"] - output["low"],
            (output["high"] - previous_close).abs(),
            (output["low"] - previous_close).abs(),
        ],
        axis=1,
    ).max(axis=1)

    output["atr"] = true_range.rolling(period).mean()

    return output


def mark_swings(
    dataframe: pd.DataFrame,
    left_bars: int,
    right_bars: int,
) -> pd.DataFrame:
    output = dataframe.copy()

    window = left_bars + right_bars + 1

    highest_value = output["high"].rolling(
        window=window,
        center=True,
    ).max()

    lowest_value = output["low"].rolling(
        window=window,
        center=True,
    ).min()

    output["swing_high"] = output["high"].eq(highest_value)
    output["swing_low"] = output["low"].eq(lowest_value)

    return output


def detect_market_structure(
    dataframe: pd.DataFrame,
    left_bars: int,
    right_bars: int,
) -> MarketStructure:
    marked = mark_swings(
        dataframe,
        left_bars,
        right_bars,
    )

    swing_highs = marked.loc[
        marked["swing_high"],
        "high",
    ].dropna()

    swing_lows = marked.loc[
        marked["swing_low"],
        "low",
    ].dropna()

    if len(swing_highs) < 2 or len(swing_lows) < 2:
        return MarketStructure(
            direction="unknown",
            latest_high=None,
            previous_high=None,
            latest_low=None,
            previous_low=None,
        )

    previous_high = float(swing_highs.iloc[-2])
    latest_high = float(swing_highs.iloc[-1])

    previous_low = float(swing_lows.iloc[-2])
    latest_low = float(swing_lows.iloc[-1])

    if latest_high > previous_high and latest_low > previous_low:
        direction = "bullish"

    elif latest_high < previous_high and latest_low < previous_low:
        direction = "bearish"

    else:
        direction = "range"

    return MarketStructure(
        direction=direction,
        latest_high=latest_high,
        previous_high=previous_high,
        latest_low=latest_low,
        previous_low=previous_low,
    )


def get_entry_levels(
    dataframe: pd.DataFrame,
    left_bars: int,
    right_bars: int,
    recent_window: int,
) -> tuple[Optional[float], Optional[float]]:
    if len(dataframe) <= recent_window + 20:
        return None, None

    historical_data = dataframe.iloc[:-recent_window].copy()

    marked = mark_swings(
        historical_data,
        left_bars,
        right_bars,
    )

    swing_highs = marked.loc[
        marked["swing_high"],
        "high",
    ].dropna()

    swing_lows = marked.loc[
        marked["swing_low"],
        "low",
    ].dropna()

    if swing_highs.empty or swing_lows.empty:
        return None, None

    return (
        float(swing_highs.iloc[-1]),
        float(swing_lows.iloc[-1]),
    )


def generate_signal(
    exchange,
    symbol: str,
    config: dict,
) -> Optional[Signal]:
    candle_limit = int(config["candle_limit"])

    higher_timeframe_data = fetch_candles(
        exchange,
        symbol,
        config["higher_timeframe"],
        candle_limit,
    )

    setup_timeframe_data = fetch_candles(
        exchange,
        symbol,
        config["setup_timeframe"],
        candle_limit,
    )

    entry_timeframe_data = fetch_candles(
        exchange,
        symbol,
        config["entry_timeframe"],
        candle_limit,
    )

    left_bars = int(config["swing_left"])
    right_bars = int(config["swing_right"])

    higher_structure = detect_market_structure(
        higher_timeframe_data,
        left_bars,
        right_bars,
    )

    setup_structure = detect_market_structure(
        setup_timeframe_data,
        left_bars,
        right_bars,
    )

    if higher_structure.direction not in ("bullish", "bearish"):
        log.info(
            "%s | 4H structure is %s",
            symbol,
            higher_structure.direction,
        )
        return None

    if setup_structure.direction != higher_structure.direction:
        log.info(
            "%s | Structure conflict: 4H=%s, 1H=%s",
            symbol,
            higher_structure.direction,
            setup_structure.direction,
        )
        return None

    entry_timeframe_data = add_atr(
        entry_timeframe_data,
        int(config["atr_period"]),
    )

    volume_period = int(config["volume_sma_period"])

    entry_timeframe_data["volume_average"] = (
        entry_timeframe_data["volume"]
        .rolling(volume_period)
        .mean()
    )

    latest_candle = entry_timeframe_data.iloc[-1]

    atr = float(latest_candle["atr"])
    average_volume = float(latest_candle["volume_average"])

    if not np.isfinite(atr) or atr <= 0:
        return None

    if not np.isfinite(average_volume) or average_volume <= 0:
        return None

    volume_ratio = (
        float(latest_candle["volume"]) / average_volume
    )

    minimum_volume_ratio = float(
        config["minimum_volume_ratio"]
    )

    if volume_ratio < minimum_volume_ratio:
        log.info(
            "%s | Volume too low: %.2fx",
            symbol,
            volume_ratio,
        )
        return None

    recent_window = int(config["recent_bos_window"])

    structure_high, structure_low = get_entry_levels(
        entry_timeframe_data,
        left_bars,
        right_bars,
        recent_window,
    )

    if structure_high is None or structure_low is None:
        return None

    recent_candles = entry_timeframe_data.tail(
        recent_window
    )

    current_open = float(latest_candle["open"])
    current_high = float(latest_candle["high"])
    current_low = float(latest_candle["low"])
    current_close = float(latest_candle["close"])

    pullback_tolerance = (
        atr * float(config["pullback_atr_tolerance"])
    )

    stop_buffer = (
        atr * float(config["stop_atr_buffer"])
    )

    minimum_risk_reward = float(config["minimum_rr"])

    if higher_structure.direction == "bullish":
        bos_occurred = bool(
            (recent_candles["close"] > structure_high).any()
        )

        retest_occurred = (
            current_low
            <= structure_high + pullback_tolerance
            and current_close > structure_high
        )

        bullish_confirmation = current_close > current_open

        if not (
            bos_occurred
            and retest_occurred
            and bullish_confirmation
        ):
            log.info(
                "%s | No qualified bullish BOS and retest",
                symbol,
            )
            return None

        stop_loss = min(
            structure_low,
            structure_high - atr,
        ) - stop_buffer

        risk = current_close - stop_loss

        if risk <= 0:
            return None

        target_1 = (
            current_close
            + risk * minimum_risk_reward
        )

        target_2 = (
            current_close
            + risk * (minimum_risk_reward + 1)
        )

        return Signal(
            symbol=symbol,
            side="BUY",
            entry=current_close,
            stop_loss=stop_loss,
            target_1=target_1,
            target_2=target_2,
            candle_timestamp=int(
                latest_candle["timestamp"]
            ),
            reasons=(
                "4H bullish HH/HL structure",
                "1H bullish structure confirmation",
                "15m bullish break of structure",
                "15m retest held above structure",
                f"Volume {volume_ratio:.2f}x average",
            ),
        )

    bos_occurred = bool(
        (recent_candles["close"] < structure_low).any()
    )

    retest_occurred = (
        current_high
        >= structure_low - pullback_tolerance
        and current_close < structure_low
    )

    bearish_confirmation = current_close < current_open

    if not (
        bos_occurred
        and retest_occurred
        and bearish_confirmation
    ):
        log.info(
            "%s | No qualified bearish BOS and retest",
            symbol,
        )
        return None

    stop_loss = max(
        structure_high,
        structure_low + atr,
    ) + stop_buffer

    risk = stop_loss - current_close

    if risk <= 0:
        return None

    target_1 = (
        current_close
        - risk * minimum_risk_reward
    )

    target_2 = (
        current_close
        - risk * (minimum_risk_reward + 1)
    )

    return Signal(
        symbol=symbol,
        side="SELL",
        entry=current_close,
        stop_loss=stop_loss,
        target_1=target_1,
        target_2=target_2,
        candle_timestamp=int(
            latest_candle["timestamp"]
        ),
        reasons=(
            "4H bearish LH/LL structure",
            "1H bearish structure confirmation",
            "15m bearish break of structure",
            "15m retest rejected below structure",
            f"Volume {volume_ratio:.2f}x average",
        ),
    )


def format_price(price: float) -> str:
    if price >= 1000:
        return f"{price:,.2f}"

    if price >= 1:
        return f"{price:.4f}"

    return f"{price:.6f}"


def format_signal(signal: Signal) -> str:
    reason_text = "\n".join(
        f"✅ {reason}"
        for reason in signal.reasons
    )

    risk_distance = abs(
        signal.entry - signal.stop_loss
    )

    return (
        f"📊 {signal.symbol}\n"
        f"Signal: {signal.side}\n\n"
        f"Entry: {format_price(signal.entry)}\n"
        f"Stop Loss: {format_price(signal.stop_loss)}\n"
        f"Target 1: {format_price(signal.target_1)}\n"
        f"Target 2: {format_price(signal.target_2)}\n"
        f"Risk distance: {format_price(risk_distance)}\n\n"
        f"{reason_text}\n\n"
        "⚠️ Alert only. This is not guaranteed financial advice."
    )


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
            "Telegram secrets are missing. Printing signal instead."
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
        },
        timeout=20,
    )

    response.raise_for_status()


def main() -> None:
    config = load_config()

    exchange_id = os.getenv(
        "EXCHANGE_ID",
        "binance",
    )

    exchange = create_exchange(exchange_id)

    configured_symbols = config["symbols"]

    available_symbols = [
        symbol
        for symbol in configured_symbols
        if symbol in exchange.markets
    ]

    missing_symbols = sorted(
        set(configured_symbols)
        - set(available_symbols)
    )

    if missing_symbols:
        log.warning(
            "Unavailable symbols on %s: %s",
            exchange_id,
            ", ".join(missing_symbols),
        )

    log.info(
        "Scanning %s symbols on %s",
        len(available_symbols),
        exchange_id,
    )

    signals_found = 0

    for symbol in available_symbols:
        try:
            log.info("Checking %s", symbol)

            signal = generate_signal(
                exchange,
                symbol,
                config,
            )

            if signal is None:
                log.info(
                    "%s | No qualified setup",
                    symbol,
                )
                continue

            signals_found += 1

            message = format_signal(signal)

            send_telegram(message)

            log.info(
                "%s | %s signal sent",
                symbol,
                signal.side,
            )

        except (
            ccxt.NetworkError,
            ccxt.ExchangeError,
            requests.RequestException,
        ) as error:
            log.warning(
                "%s | API error: %s",
                symbol,
                error,
            )

        except Exception:
            log.exception(
                "%s | Unexpected scan failure",
                symbol,
            )

    log.info(
        "Scan complete. Signals found: %s",
        signals_found,
    )


if __name__ == "__main__":
    main()
