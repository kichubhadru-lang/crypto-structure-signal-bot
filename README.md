# Crypto Market Structure Signal Bot

Alert-only MVP for:

**Tier 1:** BTC, ETH, SOL, BNB  
**Tier 2:** XRP, DOGE, LINK, SUI

## Logic

- 4H market structure determines directional bias.
- 1H structure must agree with 4H.
- 15m requires a recent break of structure and retest/hold.
- Current volume must exceed its 20-candle average.
- ATR creates the stop buffer.
- Targets use a minimum 1:2 risk-to-reward.
- Ranging or conflicting structures produce no signal.
- Duplicate/cooldown protection is stored in SQLite.

## Install

```bash
python -m venv .venv
source .venv/bin/activate
# Windows: .venv\Scripts\activate

pip install -r requirements.txt
cp .env.example .env
python bot.py
```

Without Telegram credentials, signals print in the terminal.

## Telegram setup

Put these values in `.env`:

```env
TELEGRAM_BOT_TOKEN=your_token
TELEGRAM_CHAT_ID=your_chat_id
```

## Important

This bot does not guarantee profitable or exact signals. It is an alert engine that applies explicit rules. Paper-test and backtest it before considering live execution.
