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
                ranking_metric='sharpe', # 'sharpe' or 'return'
                retrain_max_age_days=1,  # Retrain models every morning.
                walk_forward_window=252, # ~1 year of training data for backtests
                retrain_period=63        # ~1 quarter before retraining in backtests
            ),
            "max_positions": 30,
            "total_allocation_pct": 0.50, # Use 50% of total equity for this agent
            # NOTE: HMMStrategy now uses dynamic stop-loss and take-profit targets.
            # These static values are used as fallbacks by the orchestrator.
            # The orchestrator also uses these to define a re-entry "middle ground".
            "stop_loss_pct": 0.05,
            # Take-profit is set to 2x the stop-loss for a 2:1 risk/reward ratio.
            "take_profit_pct": 0.10, # 2 * stop_loss_pct
            "max_analysis_workers": 4,    # Limit CPU usage for analysis. Default is os.cpu_count() - 1.
        }
        # ,{
        #     "name": "Donchian_Breakout_Agent",
        #     "strategy": DonchianBreakoutStrategy(period=20), # 20-day breakout
        #     "max_positions": 10,
        #     "total_allocation_pct": 0.10, # Use 20% of total equity for this agent
        #     "stop_loss_pct": 0.07,
        #     "take_profit_pct": 0.15,
        #     # Using default waterfall allocation for this agent
        #     "waterfall_allocation_pcts": None,
        # }
    ]

    # --- Schedule Configuration ---
    # Define when the main analysis cycle should run using cron-style syntax.
    # This example runs at 9:45 AM and 3:45 PM Eastern Time.
    # The websocket connection will handle stop-loss/take-profit in real-time.
    SCHEDULE_CONFIG = {
        'hour': '8,15',
        'minute': '45',
        'timezone': 'America/New_York'
    }

    async def main(orchestrator, schedule_config):
        """The main async entry point for the trader."""
        try:
            print("--- Starting Orchestrator Event Loop ---")
            print("Press Ctrl+C to stop.")
            # Shield the orchestrator's start method from cancellation. When Ctrl+C
            # is pressed, asyncio.run cancels this 'main' task. The 'await shield'
            # raises a CancelledError, but the orchestrator.start() task is NOT
            # cancelled. This allows our 'finally' block to perform a clean shutdown.
            await asyncio.shield(orchestrator.start(schedule_config=schedule_config))
        except asyncio.CancelledError:
            # This is the expected exception when Ctrl+C is pressed.
            print("\nShutdown signal received. Proceeding with graceful shutdown.")
            pass
        finally:
            # This ensures graceful shutdown of the orchestrator's components.
            await orchestrator.shutdown()

    # --- Orchestrator Initialization and Execution ---
    # The orchestrator takes the list of agent configurations and manages them.
    print("--- Initializing Orchestrator ---")
    orchestrator = Orchestrator(agent_configs=agent_configs)

    # Start the orchestrator's continuous monitoring and trading loop.
    try:
        asyncio.run(main(orchestrator, SCHEDULE_CONFIG))
    except KeyboardInterrupt:
        # This handles Ctrl+C if it's pressed before the event loop even starts.
        print("\nOrchestrator stopped by user before starting.")
    finally:
        print("\nTrader script finished.")