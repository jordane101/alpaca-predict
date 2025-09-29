"""
A simple trading bot that uses the HMM analysis prediction
to make trade decisions with the Alpaca API. This script
configures and runs a TradingAgent.

Author - Eli Jordan
Date - 07/29/2025
"""

import asyncio
from orchestrator import Orchestrator
from strategies import HMMStrategy, DonchianBreakoutStrategy

if __name__ == "__main__":
    # --- Define Agent Configurations ---
    # Create a list of configurations, where each dictionary defines one agent.
    # The orchestrator will manage the total capital, so 'total_allocation_pct'
    # for each agent should sum up to your desired total allocation (or less).
    agent_configs = [
        {
            "name": "HMM_Momentum_Agent",
            "strategy": HMMStrategy(
                n_components=3,
                model_order=1,
                optimize_order=False, # Set to True to find best model order per stock (slower)
                max_order_to_test=5,
                ranking_metric='sharpe' # 'sharpe' or 'return'
            ),
            "max_positions": 30,
            "total_allocation_pct": 0.50, # Use 50% of total equity for this agent
            "stop_loss_pct": 0.05,        # Sell if a position drops 5%
            "take_profit_pct": 0.10,      # Sell if a position gains 10%
        }
        ,{
            "name": "Donchian_Breakout_Agent",
            "strategy": DonchianBreakoutStrategy(period=20), # 20-day breakout
            "max_positions": 10,
            "total_allocation_pct": 0.10, # Use 20% of total equity for this agent
            "stop_loss_pct": 0.07,
            "take_profit_pct": 0.15,
            # Using default waterfall allocation for this agent
            "waterfall_allocation_pcts": None,
        }
    ]

    # --- Schedule Configuration ---
    # Define when the main analysis cycle should run using cron-style syntax.
    # This example runs at 9:45 AM and 3:45 PM Eastern Time.
    # The websocket connection will handle stop-loss/take-profit in real-time.
    SCHEDULE_CONFIG = {
        'hour': '9,15',
        'minute': '45',
        'timezone': 'America/New_York'
    }

    # --- Orchestrator Initialization and Execution ---
    # The orchestrator takes the list of agent configurations and manages them.
    print("--- Initializing Orchestrator ---")
    orchestrator = Orchestrator(agent_configs=agent_configs)

    # Start the orchestrator's continuous monitoring and trading loop.
    try:
        print("--- Starting Orchestrator Event Loop ---")
        print("Press Ctrl+C to stop.")
        asyncio.run(orchestrator.start(schedule_config=SCHEDULE_CONFIG))
    except KeyboardInterrupt:
        print("\nOrchestrator stopped by user.")
    finally:
        print("\nTrader script finished.")