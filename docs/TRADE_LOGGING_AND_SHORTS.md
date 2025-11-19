# Trade Logging & Short Position Fixes

**Date:** November 14, 2025  
**Version:** 2.1.1

## Summary

Added comprehensive trade decision logging and fixed short position order submission to comply with Alpaca's API requirements.

---

## 1. Trade Decision Logging

### New Feature: Dedicated Trade Log

**Log File:** `data/logs/trade_decisions.log`

All trade decisions (opens, closes, holds, rejections) are now logged with timestamps and full context.

### Log Format

```
YYYY-MM-DD HH:MM:SS | ACTION | TICKER | AGENT_NAME | key=value | key=value ...
```

### Logged Actions

| Action | Description | Details Logged |
|--------|-------------|----------------|
| `OPEN_LONG` | Opened long position | price, notional, stop_loss, take_profit |
| `OPEN_SHORT` | Opened short position | price, qty, notional, stop_loss, take_profit |
| `CLOSE` | Closed position | reason (stop_loss, take_profit, agent_signal), entry_price, quantity |
| `HOLD` | Position held (no action) | current_state |
| `REJECTED` | Trade rejected | reason, additional context |

### Rejection Reasons

- `notional_too_small` - Notional value < $1
- `insufficient_notional_for_short` - Not enough $ to buy 1 whole share for short
- `already_owned` - Ticker already in portfolio
- `no_price_data` - Could not fetch latest price
- `api_error` - Alpaca API error
- `close_failed` - Failed to close position
- `asset_not_active` - Asset delisted/not tradeable

### Example Log Entries

```
2025-11-14 08:45:23 | OPEN_LONG    | AAPL   | HMM_Causal_Agent     | price=$175.23 | notional=$3125.00 | stop_loss=$172.50 | take_profit=$180.25
2025-11-14 08:45:25 | OPEN_SHORT   | AVGO   | HMM_Causal_Agent     | price=$337.80 | qty=4 | notional=$1351.20 | stop_loss=$346.21 | take_profit=$320.60
2025-11-14 11:45:12 | CLOSE        | NVDA   | HMM_Causal_Agent     | reason=take_profit | entry_price=$485.20 | quantity=6.2453
2025-11-14 11:45:15 | REJECTED     | TSLA   | HMM_Causal_Agent     | reason=already_owned | state=OPEN
2025-11-14 16:45:03 | REJECTED     | SPLK   | HMM_Causal_Agent     | reason=api_error | error=asset SPLK is not active | attempted_action=OPEN_SHORT
```

### How to View Trade Log

```bash
# View all trades today
grep "$(date +%Y-%m-%d)" data/logs/trade_decisions.log

# View only successful opens
grep "OPEN_" data/logs/trade_decisions.log

# View only shorts
grep "SHORT" data/logs/trade_decisions.log

# View all rejections
grep "REJECTED" data/logs/trade_decisions.log

# View trades for specific ticker
grep "AAPL" data/logs/trade_decisions.log

# Last 20 trades
tail -20 data/logs/trade_decisions.log
```

---

## 2. Short Position Fixes

### Problem Identified

When attempting to open short positions, two issues were encountered:

1. **Fractional Shorts Not Supported**
   ```
   Error: "fractional orders cannot be sold short"
   ```
   Alpaca does not allow fractional shares for short positions.

2. **Inactive Assets**
   ```
   Error: "asset SPLK is not active"
   ```
   Some tickers (like SPLK - Splunk, acquired by Cisco) are no longer tradeable.

### Solution Implemented

**For Long Positions:**
- Continue using `notional` (dollar amount)
- Allows fractional shares
- Example: Buy $1,500 of AAPL → 8.57 shares @ $175

**For Short Positions:**
- Use `qty` (whole shares) instead of `notional`
- Calculate: `qty = int(notional / price)`
- Example: Short $1,500 of AVGO @ $337.80 → 4 shares
- Reject if qty < 1 share

### Code Changes

**Before (Broken):**
```python
# Used notional for all orders
market_order_data = MarketOrderRequest(
    symbol=ticker,
    notional=abs_notional,  # ❌ Fails for shorts
    side=OrderSide.SELL,
    time_in_force=TimeInForce.DAY
)
```

**After (Fixed):**
```python
if is_short:
    # Calculate whole shares for short
    qty = int(abs_notional / entry_price)
    if qty < 1:
        # Reject if not enough for 1 share
        return False
    market_order_data = MarketOrderRequest(
        symbol=ticker,
        qty=qty,  # ✅ Whole shares
        side=OrderSide.SELL,
        time_in_force=TimeInForce.DAY
    )
else:
    # Use notional for longs (supports fractional)
    market_order_data = MarketOrderRequest(
        symbol=ticker,
        notional=abs_notional,
        side=OrderSide.BUY,
        time_in_force=TimeInForce.DAY
    )
```

### Minimum Short Position Size

Since shorts require whole shares, the minimum notional for a short is:
```
min_notional = stock_price * 1 share
```

Examples:
- AVGO @ $337.80 → Minimum $337.80 to short
- NVDA @ $145.25 → Minimum $145.25 to short  
- AAPL @ $175.20 → Minimum $175.20 to short

If allocated notional < stock price, the short is automatically rejected and logged.

---

## 3. Short Position Verification

### How to Check if Shorts Are Working

1. **Check Trade Log for Short Orders:**
   ```bash
   grep "OPEN_SHORT" data/logs/trade_decisions.log
   ```

2. **Check for Short Rejections:**
   ```bash
   grep "SHORT" data/logs/trade_decisions.log | grep "REJECTED"
   ```

3. **Check Alpaca Account:**
   - Log in to [Alpaca Paper Trading](https://app.alpaca.markets/paper/dashboard/overview)
   - Go to "Positions" tab
   - Short positions show as **negative quantities**

4. **Check Service Logs:**
   ```bash
   grep "SHORT" data/logs/trader_service.log | tail -20
   ```

### Current Short Position Settings

From `config/trader_config.yaml`:
```yaml
allow_shorts: true                 # Shorts enabled ✅
short_confidence_threshold: 0.75   # 75% confidence required
```

For shorts to be placed:
- HMM must predict **negative outlook** (bearish state)
- Confidence must be >= **75%**
- Available position slots (max 20)
- Notional must be >= stock price (for 1 share minimum)

---

## 4. Testing & Validation

### Test Short Order Submission

Run a dry-run to see what would be traded:
```bash
.venv/bin/python scripts/trader_with_config.py --dry-run
```

Look for:
- `Action=SHORT` in the signals
- `Submitting SHORT order for [TICKER]`
- Success or rejection messages

### Monitor Next Scheduled Run

Next runs: **8:45 AM, 11:45 AM, 4:45 PM EST**

Watch logs in real-time:
```bash
# Watch main service log
tail -f data/logs/trader_service.log

# Watch trade decisions (separate terminal)
tail -f data/logs/trade_decisions.log
```

### Manual Trade Decision Review

Check what the bot decided at last run:
```bash
# Last 50 lines of service log
tail -50 data/logs/trader_service.log

# All decisions from last 2 hours
grep "$(date -d '2 hours ago' +%Y-%m-%d)" data/logs/trade_decisions.log
```

---

## 5. Configuration Reference

### Enable/Disable Shorts

**File:** `config/trader_config.yaml`

```yaml
agents:
  - name: "HMM_Causal_Agent"
    # ...
    strategy:
      hmm:
        allow_shorts: true              # Set to false to disable
        short_confidence_threshold: 0.75 # 0.6-0.95 recommended
```

After changing config:
```bash
sudo systemctl restart alpaca-trader
```

### Adjust Short Confidence Threshold

Higher threshold = fewer shorts, higher quality signals:
- `0.60` - Aggressive (trade more shorts)
- `0.75` - **Default** (balanced)
- `0.90` - Conservative (only very confident shorts)

---

## 6. Troubleshooting

### No Shorts Being Placed

**Check 1: Are shorts enabled?**
```bash
grep "allow_shorts" config/trader_config.yaml
```
Should show `allow_shorts: true`

**Check 2: Are signals being generated?**
```bash
grep "Action=SHORT" data/logs/trader_service.log | tail -10
```

**Check 3: Are they being rejected?**
```bash
grep "SHORT" data/logs/trade_decisions.log | grep "REJECTED"
```

**Check 4: Check confidence threshold:**
```bash
# See if any shorts had confidence >= threshold
grep "SHORT" data/logs/trader_service.log | grep "Confidence="
```

### Shorts Rejected with "asset not active"

Some stocks have been delisted or acquired. These are filtered out automatically.

**Solution:** The DAG universe is updated quarterly. Force refresh:
```bash
# Rebuild DAG with latest S&P 500 constituents
.venv/bin/python scripts/build_large_dag.py
```

### Shorts Rejected with "insufficient_notional_for_short"

Stock price is higher than allocated notional.

**Example:**
- Allocated: $500 for position
- Stock price: $675
- Shares: 0 (need at least 1)
- Result: Rejected

**Solution:** Increase allocation per position:
```yaml
agents:
  - max_positions: 15  # Fewer positions = more $ per position
```

Or use a different position sizing strategy:
```yaml
hmm:
  max_position: 0.75  # Max 75% of allocation per position
```

---

## 7. Files Modified

### Core Changes

1. **`src/trading/orchestrator.py`**
   - Added `_setup_trade_logger()` method
   - Added `_log_trade_decision()` method
   - Modified `_execute_buy()` to use qty for shorts
   - Modified `_execute_sell()` to log reason
   - Added logging calls throughout trade flow

2. **`config/trader_config.yaml`**
   - Changed `allow_shorts: false` → `allow_shorts: true`

### New Files

- `data/logs/trade_decisions.log` - Dedicated trade decision log

---

## 8. Next Steps

1. **Monitor Trade Log** - Watch next scheduled run (4:45 PM EST)
2. **Verify Shorts** - Check Alpaca account for negative quantities
3. **Analyze Performance** - Compare long vs short P&L after 1 week
4. **Adjust Thresholds** - Tune confidence threshold based on results

---

## Version History

- **v2.1.1** (Nov 14, 2025) - Added trade logging and fixed short order submission
- **v2.1.0** (Nov 13, 2025) - Short position support with inverted stop-loss/take-profit
- **v2.0.0** (Nov 12, 2025) - Causal DAG features and confidence-based sizing

---

**Documentation Author:** AI Assistant  
**Last Updated:** November 14, 2025
