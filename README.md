# Crypto V3.6 Adaptive Scanner

This upgrade adds measured performance adaptation to the V3.5 Institutional scanner.

## New features

- Automatically checks open signals against later 15-minute candles
- Records TP1 wins and stop-loss losses
- Maintains win/loss statistics by setup pattern
- Adjusts future confidence scores only after enough samples
- Skips altcoins when BTC 4H is neutral
- Sends Telegram scan and performance summaries
- Preserves duplicate and performance state through GitHub Actions cache

## Important methodology note

When TP1 and stop loss appear inside the same 15-minute candle, the bot records a loss because candle data cannot prove which price was reached first. This conservative rule avoids overstating performance.

Historical performance can improve calibration but cannot guarantee future results.
