#!/usr/bin/env python3
"""
Test script to manually sync positions and verify INTC short is added.
"""
import os
import sys
import json
from dotenv import load_dotenv

# Add project root to path
sys.path.insert(0, '/home/eli/alpaca-predict')

from alpaca.trading.client import TradingClient
from src.trading.trading_agent import TradingAgent
from src.trading.position import ManagedPosition, PositionState

load_dotenv('.env')
KEY = os.getenv('PAPER_KEY')
SECRET = os.getenv('PAPER_SEC')

client = TradingClient(KEY, SECRET, paper=True)

# Get live positions
positions = client.get_all_positions()

print(f"\n{'='*60}")
print(f"TESTING POSITION SYNC LOGIC")
print(f"{'='*60}\n")

print(f"Found {len(positions)} positions in Alpaca account:")
for pos in positions:
    qty = float(pos.qty)
    side = "SHORT" if qty < 0 else "LONG "
    print(f"  {side} {pos.symbol:6} qty={qty:10.4f} @ ${float(pos.avg_entry_price):.2f}")

# Load current managed positions
with open('position_ownership.json', 'r') as f:
    managed_data = json.load(f)

managed_tickers = set()
for agent_positions in managed_data.values():
    for pos in agent_positions:
        managed_tickers.add(pos['symbol'])

print(f"\nCurrently managed: {len(managed_tickers)} tickers")

# Find missing positions
live_tickers = {p.symbol for p in positions}
missing = live_tickers - managed_tickers

print(f"\n{len(missing)} positions NOT in managed list:")
for ticker in sorted(missing):
    pos = next((p for p in positions if p.symbol == ticker), None)
    if pos:
        qty = float(pos.qty)
        side = "SHORT" if qty < 0 else "LONG "
        print(f"  {side} {ticker:6} qty={qty:10.4f} @ ${float(pos.avg_entry_price):.2f}")

# Simulate the sync logic for missing positions
print(f"\n{'='*60}")
print(f"SIMULATING SYNC - ADDING MISSING POSITIONS")
print(f"{'='*60}\n")

# Create a mock agent
class MockAgent:
    name = "HMM_Causal_Agent"
    asset_class = "us_equity"
    stop_loss_pct = 0.03  # 3%
    take_profit_pct = 0.06  # 6%

agent = MockAgent()

new_positions = {}
for live_pos in positions:
    ticker = live_pos.symbol
    if ticker not in managed_tickers:
        qty = float(live_pos.qty)
        entry_price = float(live_pos.avg_entry_price)
        is_short = qty < 0
        
        # Calculate SL/TP
        if is_short:
            stop_loss_price = entry_price * (1 + agent.stop_loss_pct)
            take_profit_price = entry_price * (1 - agent.take_profit_pct * 2)
        else:
            stop_loss_price = entry_price * (1 - agent.stop_loss_pct)
            take_profit_price = entry_price * (1 + agent.take_profit_pct * 2)
        
        new_positions[ticker] = {
            'symbol': ticker,
            'quantity': qty,
            'entry_price': entry_price,
            'is_short': is_short,
            'stop_loss_price': stop_loss_price,
            'take_profit_price': take_profit_price,
            'agent_name': agent.name,
            'state': 'OPEN'
        }
        
        side = "SHORT" if is_short else "LONG "
        print(f"{side} {ticker:6}")
        print(f"  Entry: ${entry_price:.2f}")
        print(f"  Qty: {qty:.4f}")
        print(f"  Stop Loss: ${stop_loss_price:.2f} ({'above' if is_short else 'below'} entry)")
        print(f"  Take Profit: ${take_profit_price:.2f} ({'below' if is_short else 'above'} entry)")
        print()

print(f"{'='*60}")
print(f"RESULT: {len(new_positions)} positions would be added to management")
print(f"{'='*60}\n")

# Verify INTC specifically
if 'INTC' in new_positions:
    intc = new_positions['INTC']
    print("✅ SUCCESS: INTC short position will be added!")
    print(f"   Entry: ${intc['entry_price']:.2f}")
    print(f"   Qty: {intc['quantity']:.4f} (negative = short)")
    print(f"   Stop Loss: ${intc['stop_loss_price']:.2f} (above entry - limits loss if price rises)")
    print(f"   Take Profit: ${intc['take_profit_price']:.2f} (below entry - captures profit if price falls)")
else:
    print("❌ PROBLEM: INTC was not added!")

print()
