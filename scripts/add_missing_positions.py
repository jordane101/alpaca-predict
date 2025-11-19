#!/usr/bin/env python3
"""
Manually add missing positions (INTC, AVGO, TSLA, ABNB) to position_ownership.json
"""
import os
import sys
import json
from dotenv import load_dotenv

sys.path.insert(0, '/home/eli/alpaca-predict')

from alpaca.trading.client import TradingClient

load_dotenv('.env')
KEY = os.getenv('PAPER_KEY')
SECRET = os.getenv('PAPER_SEC')

client = TradingClient(KEY, SECRET, paper=True)
positions = client.get_all_positions()

# Load current managed positions
with open('position_ownership.json', 'r') as f:
    managed_data = json.load(f)

# Get managed tickers
managed_tickers = set()
agent_name = None
for agent, agent_positions in managed_data.items():
    agent_name = agent  # Get the agent name
    for pos in agent_positions:
        managed_tickers.add(pos['symbol'])

if not agent_name:
    print("ERROR: No agent found in position_ownership.json")
    sys.exit(1)

print(f"Agent: {agent_name}")
print(f"Currently managed: {len(managed_tickers)} tickers\n")

# Find missing positions
missing_tickers = []
for pos in positions:
    if pos.symbol not in managed_tickers:
        missing_tickers.append(pos.symbol)

print(f"Found {len(missing_tickers)} positions NOT in managed list: {missing_tickers}\n")

if not missing_tickers:
    print("✅ All positions are already managed!")
    sys.exit(0)

# Add missing positions
stop_loss_pct = 0.03  # 3%
take_profit_pct = 0.06  # 6%

new_positions = []
for pos in positions:
    if pos.symbol in missing_tickers:
        qty = float(pos.qty)
        entry_price = float(pos.avg_entry_price)
        is_short = qty < 0
        
        # Calculate SL/TP
        if is_short:
            stop_loss_price = entry_price * (1 + stop_loss_pct)
            take_profit_price = entry_price * (1 - take_profit_pct * 2)
        else:
            stop_loss_price = entry_price * (1 - stop_loss_pct)
            take_profit_price = entry_price * (1 + take_profit_pct * 2)
        
        new_pos = {
            "symbol": pos.symbol,
            "entry_price": entry_price,
            "quantity": qty,
            "stop_loss_price": stop_loss_price,
            "take_profit_price": take_profit_price,
            "state": "OPEN",
            "cooldown_reason": None
        }
        
        new_positions.append(new_pos)
        
        side = "SHORT" if is_short else "LONG "
        print(f"Adding {side} {pos.symbol}:")
        print(f"  Entry: ${entry_price:.2f}")
        print(f"  Qty: {qty:.4f}")
        print(f"  Stop Loss: ${stop_loss_price:.2f}")
        print(f"  Take Profit: ${take_profit_price:.2f}")

# Add to managed_data
managed_data[agent_name].extend(new_positions)

# Save back
with open('position_ownership.json', 'w') as f:
    json.dump(managed_data, f, indent=4)

print(f"\n✅ Added {len(new_positions)} positions to {agent_name}")
print(f"📁 Updated position_ownership.json")
print(f"\nNew total: {len(managed_data[agent_name])} managed positions")
