"""
Manages multiple TradingAgents, resolving trade conflicts and executing orders.

Author - Eli Jordan
Date - 07/29/2025
"""

import os
import json
import asyncio
from dotenv import load_dotenv
from alpaca.trading.client import TradingClient
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.live.stock import StockDataStream
from alpaca.trading.requests import MarketOrderRequest, ClosePositionRequest
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.common.exceptions import APIError
from alpaca.data.enums import DataFeed
from apscheduler.schedulers.asyncio import AsyncIOScheduler

from trading_agent import TradingAgent
from strategies import HMMStrategy, DonchianBreakoutStrategy

class Orchestrator:
    """
    Manages multiple TradingAgents, resolving trade conflicts and executing orders.
    """
    load_dotenv(".env")
    KEY = os.getenv("PAPER_KEY")
    SECRET = os.getenv("PAPER_SEC")
    OWNERSHIP_FILE = "position_ownership.json"

    def __init__(self, agent_configs: list):
        """
        Initializes the Orchestrator.

        Args:
            agent_configs (list): A list of dictionaries, each defining an agent's configuration.
        """
        self.trading_client = TradingClient(self.KEY, self.SECRET, paper=True)
        self.data_client = StockHistoricalDataClient(self.KEY, self.SECRET)
        self.stream_client = StockDataStream(self.KEY, self.SECRET, feed=DataFeed.IEX)

        self.agents = [
            TradingAgent(
                trading_client=self.trading_client,
                data_client=self.data_client,
                **config
            ) for config in agent_configs
        ]

        self.position_ownership = self._load_ownership()
        self.live_positions = {} # A cache of symbol -> position object
        self.triggered_sells = set() # A set to prevent duplicate real-time triggers
        print(f"Initialized {len(self.agents)} agents: {[agent.name for agent in self.agents]}")

    def _load_ownership(self):
        if os.path.exists(self.OWNERSHIP_FILE):
            try:
                with open(self.OWNERSHIP_FILE, 'r') as f:
                    print(f"Loading position ownership from {self.OWNERSHIP_FILE}")
                    return json.load(f)
            except (json.JSONDecodeError, IOError) as e:
                print(f"Warning: Could not load ownership file. Starting fresh. Error: {e}")
        return {}

    def _save_ownership(self):
        try:
            with open(self.OWNERSHIP_FILE, 'w') as f:
                print(f"Saving position ownership to {self.OWNERSHIP_FILE}")
                json.dump(self.position_ownership, f, indent=4)
        except IOError as e:
            print(f"Error: Could not save ownership file. Error: {e}")

    def _sync_ownership_with_account(self, account_positions):
        """
        Aligns the ownership map with actual account positions.
        - Removes ownership for tickers that are no longer held.
        """
        print("\nSyncing ownership map with live account positions...")
        live_tickers = {p.symbol for p in account_positions}

        owned_tickers = list(self.position_ownership.keys())
        for ticker in owned_tickers:
            if ticker not in live_tickers:
                print(f"  -> Position {ticker} no longer held. Removing from ownership map.")
                del self.position_ownership[ticker]

        print(f"Synced ownership map: {self.position_ownership}")

    async def start(self, schedule_config: dict):
        """
        Starts the orchestrator's main event loop.
        - Runs an initial analysis cycle.
        - Schedules recurring analysis cycles.
        - Connects to the WebSocket for real-time price monitoring.
        """
        # 1. Run an initial analysis cycle to get positions and decisions
        await self.run_analysis_cycle()

        # 2. Set up the scheduler for recurring analysis
        scheduler = AsyncIOScheduler()
        scheduler.add_job(self.run_analysis_cycle, 'cron', **schedule_config)
        scheduler.start()
        print(f"\nAnalysis cycle scheduled. First run will be at the next configured time: {schedule_config}")

        # 3. Start the WebSocket stream for real-time monitoring
        print("Connecting to WebSocket for real-time trade monitoring...")
        await self.stream_client.run()

    async def run_analysis_cycle(self):
        """
        Runs a full analysis and trading cycle.
        This is intended to be run on a schedule.
        """
        print("\n" + "="*20 + " Starting Scheduled Analysis Cycle " + "="*20)
        try:
            loop = asyncio.get_running_loop()
            account = await loop.run_in_executor(None, self.trading_client.get_account)
            positions = await loop.run_in_executor(None, self.trading_client.get_all_positions)

            # Update our live position cache and clear previous real-time triggers
            self.live_positions = {p.symbol: p for p in positions}
            self.triggered_sells.clear()

            self._sync_ownership_with_account(positions)

            all_decisions = {'buys': [], 'sells': []}

            # Gather decisions from all agents (run blocking analysis in executor)
            for agent in self.agents:
                print(f"\n--- Getting decisions from agent: {agent.name} ---")
                owned_tickers = {
                    ticker for ticker, data in self.position_ownership.items()
                    if isinstance(data, dict) and data.get('owner') == agent.name
                }
                decisions = await loop.run_in_executor(
                    None, agent.generate_trade_decisions, account, positions, owned_tickers
                )

                for ticker in decisions.get('sells', []):
                    all_decisions['sells'].append((agent, ticker))
                for pick in decisions.get('buys', []):
                    all_decisions['buys'].append((agent, pick))

            # Resolve and execute trades
            await self._resolve_and_execute_trades(all_decisions)

            # Update WebSocket subscriptions based on new portfolio
            await self._update_subscriptions()

        except Exception as e:
            print(f"An error occurred during the analysis cycle: {e}")
        finally:
            self._save_ownership()
            print("\n" + "="*20 + " Scheduled Analysis Cycle Complete " + "="*20)

    async def on_trade(self, trade):
        """
        Callback for real-time trade updates from the WebSocket.
        Handles real-time stop-loss and take-profit checks.
        """
        symbol = trade.symbol
        price = trade.price

        # Ignore if not a position we own or if already triggered for selling
        if symbol not in self.live_positions or symbol in self.triggered_sells:
            return

        position = self.live_positions[symbol]
        # The ownership map now contains all the info we need.
        # It can be a dict (new format) or string (old format, which we ignore).
        ownership_data = self.position_ownership.get(symbol)
        if not ownership_data:
            return

        if not isinstance(ownership_data, dict):
            return # Ignore positions with old ownership format

        stop_loss_price = ownership_data.get('stop_loss_price')
        take_profit_price = ownership_data.get('take_profit_price')

        triggered = False
        reason = ""

        # Check for stop-loss
        if stop_loss_price and price <= stop_loss_price:
            triggered = True
            reason = f"STOP-LOSS (Price ${price:.2f} <= Trigger ${stop_loss_price:.2f})"

        # Check for take-profit (if not already triggered by stop-loss)
        if not triggered and take_profit_price and price >= take_profit_price:
            triggered = True
            reason = f"TAKE-PROFIT (Price ${price:.2f} >= Trigger ${take_profit_price:.2f})"

        if triggered:
            print(f"\n! REAL-TIME TRIGGER: {reason} for {symbol} (owned by {ownership_data.get('owner')})")
            self.triggered_sells.add(symbol) # Prevent re-triggering
            if await self._execute_sell(symbol):
                # Clean up state after successful sell
                if symbol in self.position_ownership:
                    del self.position_ownership[symbol]
                if symbol in self.live_positions:
                    del self.live_positions[symbol]
                await self.stream_client.unsubscribe_trades(symbol)
                self._save_ownership()

    async def _resolve_and_execute_trades(self, all_decisions):
        """Processes sell and buy decisions, handling ownership and conflicts."""
        
        # --- Process Sells First ---
        print("\n--- Resolving and Executing Sells ---")
        for agent, ticker_to_sell in all_decisions['sells']:
            if ticker_to_sell in self.triggered_sells:
                print(f"Decision: {agent.name} to SELL {ticker_to_sell} -> SKIPPED (Already triggered by real-time monitor).")
                continue

            ownership_data = self.position_ownership.get(ticker_to_sell)
            # Check if owned by the agent making the decision
            if ownership_data and isinstance(ownership_data, dict) and ownership_data.get('owner') == agent.name:
                print(f"Decision: {agent.name} to SELL {ticker_to_sell} -> APPROVED")
                if await self._execute_sell(ticker_to_sell):
                    del self.position_ownership[ticker_to_sell]
            elif ownership_data: # It's owned, but not by this agent
                print(f"Decision: {agent.name} to SELL {ticker_to_sell} -> DENIED (Not owned by agent).")

        # --- Process Buys ---
        print("\n--- Resolving and Executing Buys ---")
        sorted_buys = sorted(all_decisions['buys'], key=lambda item: item[1]['ranking_strength'], reverse=True)

        for agent, pick in sorted_buys:
            ticker = pick['ticker']
            notional = pick['notional_value']

            # Check if already owned
            ownership_info = self.position_ownership.get(ticker)
            if ownership_info:
                owner = ownership_info.get('owner', 'another agent') if isinstance(ownership_info, dict) else ownership_info
                print(f"Decision: {agent.name} to BUY {ticker} -> DENIED (Already owned by {owner}).")
                continue

            # --- Calculate Price Targets BEFORE buying ---
            latest_price = await self._get_latest_price(ticker)
            if not latest_price:
                print(f"  -> Could not get latest price for {ticker}. Skipping buy.")
                continue

            stop_loss_price = None
            take_profit_price = None

            # Dynamic Target Logic for HMM Strategy
            if isinstance(agent.strategy, HMMStrategy) and pick.get('predicted_state_mean_return', 0) > 0:
                # DYNAMIC TAKE-PROFIT: Based on the state's mean return (expected reward)
                tp_pct = pick['predicted_state_mean_return']
                take_profit_price = latest_price * (1 + tp_pct)
                print(f"  -> Using DYNAMIC take-profit for {ticker} based on HMM state. Target: ${take_profit_price:.2f} ({tp_pct:.2%})")

                # DYNAMIC STOP-LOSS: Based on the state's std dev (expected volatility/risk)
                sl_pct = pick.get('predicted_state_std_return', agent.stop_loss_pct or 0)
                if sl_pct > 0:
                    stop_loss_price = latest_price * (1 - sl_pct)
                    print(f"  -> Using DYNAMIC stop-loss for {ticker} based on HMM state. Target: ${stop_loss_price:.2f} ({sl_pct:.2%})")

            # Static Target Logic for all other strategies (or as a fallback for HMM)
            else:
                if agent.take_profit_pct:
                    take_profit_price = latest_price * (1 + agent.take_profit_pct)
                    print(f"  -> Using STATIC take-profit for {ticker}. Target: ${take_profit_price:.2f} ({agent.take_profit_pct:.2%})")
                if agent.stop_loss_pct:
                    stop_loss_price = latest_price * (1 - agent.stop_loss_pct)
                    print(f"  -> Using STATIC stop-loss for {ticker}. Target: ${stop_loss_price:.2f} ({agent.stop_loss_pct:.2%})")

            print(f"Decision: {agent.name} to BUY {ticker} for ${notional:.2f} -> APPROVED")
            await self._execute_buy(ticker, notional, stop_loss_price, take_profit_price, agent.name)

    async def _execute_sell(self, ticker: str) -> bool:
        """Executes a sell order for the given ticker."""
        print(f"  -> Submitting SELL order for {ticker}.")
        try:
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(None, self.trading_client.close_position, ticker)
            print(f"  -> Successfully submitted SELL order for {ticker}.")
            return True
        except APIError as e:
            print(f"  -> Failed to close position for {ticker}. Reason: {e}")
            return False

    async def _execute_buy(self, ticker: str, notional_value: float, stop_loss_price: float | None, take_profit_price: float | None, agent_name: str) -> bool:
        """Executes a buy order for the given ticker and notional value."""
        if notional_value < 1:
            print(f"  -> Skipping BUY for {ticker}, notional value ${notional_value:.2f} is less than $1.")
            return False

        print(f"  -> Submitting BUY order for {ticker} (Notional: ${notional_value:.2f}).")
        try:
            market_order_data = MarketOrderRequest(
                symbol=ticker,
                notional=notional_value,
                side=OrderSide.BUY,
                time_in_force=TimeInForce.DAY
            )
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(None, self.trading_client.submit_order, order_data=market_order_data)
            
            # On successful submission, record ownership with targets
            self.position_ownership[ticker] = {
                "owner": agent_name,
                "stop_loss_price": stop_loss_price,
                "take_profit_price": take_profit_price,
            }
            print(f"  -> Successfully submitted BUY order for {ticker}.")
            return True
        except APIError as e:
            print(f"  -> Failed to submit order for {ticker}. Reason: {e}")
            return False

    async def _get_latest_price(self, ticker: str) -> float | None:
        """Fetches the latest trade price for a ticker."""
        try:
            loop = asyncio.get_running_loop()
            latest_trade = await loop.run_in_executor(None, self.data_client.get_stock_latest_trade, ticker, DataFeed.IEX)
            if latest_trade:
                return latest_trade.price
        except Exception as e:
            print(f"Could not fetch latest price for {ticker}: {e}")
        return None

    async def _update_subscriptions(self):
        """
        Syncs the WebSocket subscriptions with the current positions.
        """
        print("\nUpdating WebSocket subscriptions...")
        current_subs = set(self.stream_client.get_trade_subscriptions())
        desired_subs = set(self.position_ownership.keys())

        to_sub = desired_subs - current_subs
        to_unsub = current_subs - desired_subs

        if to_sub:
            print(f"  Subscribing to: {list(to_sub)}")
            await self.stream_client.subscribe_trades(self.on_trade, *to_sub)
        if to_unsub:
            print(f"  Unsubscribing from: {list(to_unsub)}")
            await self.stream_client.unsubscribe_trades(*to_unsub)
        print("Subscriptions are up to date.")