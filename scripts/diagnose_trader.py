#!/usr/bin/env python3
"""
Quick diagnostic to check why trader isn't placing trades.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import asyncio
from datetime import datetime, time
from src.trading.orchestrator import Orchestrator
from src.trading.strategies import HMMStrategy

# Check what the current schedule should be
print("="*70)
print("TRADER DIAGNOSTICS")
print("="*70)

# Current time
now = datetime.now()
print(f"\nCurrent Time: {now.strftime('%Y-%m-%d %H:%M:%S %Z')}")
print(f"Time Zone: {now.astimezone().tzname()}")

# Schedule from trader.py
schedule = {
    'hour': '8,16',
    'minute': '45',
    'timezone': 'America/New_York'
}
print(f"\nConfigured Schedule:")
print(f"  Times: {schedule['hour']}:{schedule['minute']} {schedule['timezone']}")
print(f"  Next run: 8:45 AM or 4:45 PM EST")

# Check if 8:45 has passed today
if now.hour >= 8 and now.minute >= 45:
    print(f"  ✓ 8:45 AM schedule HAS passed today")
else:
    print(f"  ✗ 8:45 AM schedule has NOT passed yet today")

if now.hour >= 16 and now.minute >= 45:
    print(f"  ✓ 4:45 PM schedule HAS passed today")
else:
    print(f"  ✗ 4:45 PM schedule has NOT passed yet today")

# Check strategy configuration
print(f"\nStrategy Configuration:")
print(f"  Model: 2-state HMM")
print(f"  Causal Features: NOT ENABLED ❌")
print(f"  Confidence Sizing: NOT ENABLED ❌")
print(f"  Max Positions: 30")
print(f"  Allocation: 50%")

print(f"\n⚠️  Note: This is the OLD configuration (v1.0)")
print(f"   The NEW configuration (v2.0) with causal features is available via:")
print(f"   python scripts/trader_with_config.py")

# Check account status
print(f"\n" + "="*70)
print("Checking Alpaca Account...")
print("="*70)

try:
    from alpaca.trading.client import TradingClient
    from dotenv import load_dotenv
    import os
    
    load_dotenv()
    api_key = os.getenv('PAPER_KEY') or os.getenv('APCA_API_KEY_ID')
    api_secret = os.getenv('PAPER_SEC') or os.getenv('APCA_API_SECRET_KEY')
    
    if not api_key or not api_secret:
        print("❌ API credentials not found in .env")
    else:
        client = TradingClient(api_key, api_secret, paper=True)
        account = client.get_account()
        
        print(f"\nAccount Status:")
        print(f"  Equity: ${float(account.equity):,.2f}")
        print(f"  Cash: ${float(account.cash):,.2f}")
        print(f"  Buying Power: ${float(account.buying_power):,.2f}")
        print(f"  Portfolio Value: ${float(account.portfolio_value):,.2f}")
        
        # Check positions
        positions = client.get_all_positions()
        print(f"\n  Current Positions: {len(positions)}")
        
        if positions:
            print(f"\n  Active Positions:")
            for pos in positions[:10]:  # Show first 10
                print(f"    {pos.symbol}: {pos.qty} shares @ ${float(pos.current_price):.2f} "
                      f"(P&L: ${float(pos.unrealized_pl):,.2f})")
        else:
            print(f"    No positions held")
        
        # Check recent orders
        print(f"\nChecking Recent Orders (last 24 hours)...")
        from alpaca.trading.requests import GetOrdersRequest
        from alpaca.trading.enums import QueryOrderStatus
        
        request = GetOrdersRequest(
            status=QueryOrderStatus.ALL,
            limit=10
        )
        orders = client.get_orders(request)
        
        if orders:
            print(f"  Found {len(orders)} recent orders:")
            for order in orders[:5]:
                print(f"    {order.created_at.strftime('%Y-%m-%d %H:%M')} - "
                      f"{order.side} {order.symbol} {order.qty} @ {order.type} - "
                      f"Status: {order.status}")
        else:
            print(f"  ❌ No orders found in last 24 hours")
            print(f"\n  Possible reasons:")
            print(f"    1. No buy signals generated (all stocks negative outlook)")
            print(f"    2. Insufficient confidence/quality signals")
            print(f"    3. Schedule hasn't run yet today")
            print(f"    4. Analysis cycle encountered an error")
        
except Exception as e:
    print(f"❌ Error checking account: {e}")

# Check if service is logging
print(f"\n" + "="*70)
print("Logging Status")
print("="*70)

log_dir = Path("data/logs")
if log_dir.exists():
    service_log = log_dir / "trader_service.log"
    service_err = log_dir / "trader_service_error.log"
    
    if service_log.exists():
        size = service_log.stat().st_size
        print(f"  trader_service.log: {size} bytes")
        if size == 0:
            print(f"    ⚠️  Log file is EMPTY - service may not be logging properly")
    
    if service_err.exists():
        print(f"  trader_service_error.log: {service_err.stat().st_size} bytes")
        
    # Look for today's trader log
    today = datetime.now().strftime("%Y-%m-%d")
    today_log = log_dir / f"trader_log_{today}.log"
    if today_log.exists():
        print(f"  Today's log: {today_log} ({today_log.stat().st_size} bytes)")
    else:
        print(f"  ⚠️  No log file for today: {today_log}")
else:
    print(f"  ❌ Log directory not found: {log_dir}")

print(f"\n" + "="*70)
print("RECOMMENDATIONS")
print("="*70)

print(f"""
1. Check if the scheduled analysis ran:
   journalctl -u alpaca-trader --since "2025-11-13 08:00" -n 100

2. If no signals, it might be because:
   - Using 2-state model (less sensitive than 4-state)
   - No causal features (less predictive power)
   - All stocks showing negative/uncertain outlook
   
3. To use the NEW v2.0 configuration with better features:
   # Stop current service
   sudo systemctl stop alpaca-trader
   
   # Edit config
   nano config/trader_config.yaml
   
   # Update service to use new script
   sudo nano /etc/systemd/system/alpaca-trader.service
   # Change ExecStart to: /path/to/trader_with_config.py
   
   # Reload and restart
   sudo systemctl daemon-reload
   sudo systemctl start alpaca-trader

4. Or manually test analysis right now:
   python scripts/trader_with_config.py --dry-run
   
5. Check if market is open:
   The bot only trades during market hours (9:30 AM - 4:00 PM EST)
""")

print("="*70)
