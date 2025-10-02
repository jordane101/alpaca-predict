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
from alpaca.trading.stream import TradingStream
from alpaca.data.live.stock import StockDataStream
from alpaca.data.live.crypto import CryptoDataStream
from alpaca.trading.requests import MarketOrderRequest, ClosePositionRequest
from alpaca.data.requests import StockLatestTradeRequest as LatestTradeRequest, CryptoLatestTradeRequest
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.common.exceptions import APIError
from alpaca.data.enums import DataFeed
from apscheduler.schedulers.asyncio import AsyncIOScheduler

from trading_agent import TradingAgent
from strategies import HMMStrategy, DonchianBreakoutStrategy
from position import ManagedPosition, PositionState, CooldownReason

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
        # For market data (trades for SL/TP), we need separate clients for stocks and crypto
        self.stock_market_stream_client = StockDataStream(self.KEY, self.SECRET, feed=DataFeed.IEX)
        self.crypto_market_stream_client = CryptoDataStream(self.KEY, self.SECRET)
        # For account data (order fills)
        self.trade_stream_client = TradingStream(self.KEY, self.SECRET, paper=True)
        self.scheduler = AsyncIOScheduler()

        self.agents = [
            TradingAgent(
                trading_client=self.trading_client,
                data_client=self.data_client,
                **config
            ) for config in agent_configs
        ]

        self.managed_positions = self._load_managed_positions()
        self.current_stock_subscriptions = set()
        self.current_crypto_subscriptions = set()
        print(f"Initialized {len(self.agents)} agents: {[agent.name for agent in self.agents]}")

    def _get_agent_by_name(self, name: str) -> TradingAgent | None:
        """Finds an agent instance by its name."""
        for agent in self.agents:
            if agent.name == name:
                return agent
        return None

    def _load_managed_positions(self):
        """Loads and reconstructs ManagedPosition objects from the ownership file."""
        if os.path.exists(self.OWNERSHIP_FILE):
            try:
                with open(self.OWNERSHIP_FILE, 'r') as f:
                    print(f"Loading managed positions from {self.OWNERSHIP_FILE}")
                    positions_data = json.load(f)
                    if not positions_data: # Handles empty JSON object or null
                        return {}
                    
                    managed_positions = {}
                    for symbol, data in positions_data.items():
                        # Handle backward compatibility with older file formats.
                        position = ManagedPosition(
                            symbol=symbol, # The symbol is the key in the JSON object.
                            entry_price=data.get('entry_price', 0.0), # Default to 0.0, will be updated from live data.
                            quantity=data.get('quantity', 0.0), # Default to 0.0, will be updated from live data.
                            stop_loss_price=data.get('stop_loss_price'),
                            take_profit_price=data.get('take_profit_price')
                        )
                        # State might be missing in old format, default to OPEN.
                        position.state = PositionState[data.get('state', 'OPEN')]

                        if data.get('cooldown_reason'):
                            position.cooldown_reason = CooldownReason[data['cooldown_reason']]
                        
                        # Associate the agent for SL/TP fallbacks on rebuy.
                        # The key for the agent's name was 'owner' in the old format.
                        position.agent_name = data.get('agent_name') or data.get('owner')

                        managed_positions[symbol] = position
                    return managed_positions

            except (json.JSONDecodeError, IOError, KeyError) as e:
                print(f"Warning: Could not load ownership file. Starting fresh. Error: {e}")
        return {}

    def _save_managed_positions(self):
        """Converts ManagedPosition objects to dictionaries and saves them to a file."""
        try:
            positions_to_save = {}
            for symbol, position in self.managed_positions.items():
                positions_to_save[symbol] = {
                    "symbol": position.symbol,
                    "agent_name": getattr(position, 'agent_name', None),
                    "entry_price": position.entry_price,
                    "quantity": position.quantity,
                    "stop_loss_price": position.stop_loss_price,
                    "take_profit_price": position.take_profit_price,
                    "state": position.state.name,
                    "cooldown_reason": position.cooldown_reason.name if position.cooldown_reason else None,
                }

            with open(self.OWNERSHIP_FILE, 'w') as f:
                print(f"Saving {len(positions_to_save)} managed positions to {self.OWNERSHIP_FILE}")
                json.dump(positions_to_save, f, indent=4)
        except IOError as e:
            print(f"Error: Could not save ownership file. Error: {e}")

    def _sync_ownership_with_account(self, account_positions):
        """
        Aligns the ownership map with actual account positions.
        - Removes ownership for tickers that are no longer held.
        - Keeps positions in COOLING_DOWN state for re-entry monitoring.
        """
        print("\nSyncing managed positions with live account positions...")
        live_tickers = {p.symbol for p in account_positions}

        managed_tickers = list(self.managed_positions.keys())
        for ticker in managed_tickers:
            position = self.managed_positions[ticker]
            if ticker not in live_tickers and position.state != PositionState.COOLING_DOWN:
                print(f"  -> Position {ticker} (State: {position.state.name}) no longer held. Removing from management.")
                del self.managed_positions[ticker]
        print(f"Synced. Now managing: {list(self.managed_positions.keys())}")

    async def _run_stock_market_stream(self):
        """Manages the stock market data websocket connection with a resilient loop."""
        loop = asyncio.get_running_loop()
        while True:
            try:
                print("Connecting to Stock Market Data WebSocket...")
                # Re-create the client to ensure a clean state on each connection attempt.
                self.stock_market_stream_client = StockDataStream(self.KEY, self.SECRET, feed=DataFeed.IEX)
                self.current_stock_subscriptions = set()  # Reset our subscription state

                await self._update_subscriptions()  # Subscribe to trades for managed positions

                # Run the blocking stream in a separate thread
                await loop.run_in_executor(None, self.stock_market_stream_client.run)

                print("Stock Market Data WebSocket stream has been closed. Exiting monitoring loop.")
                break
            except (asyncio.CancelledError, KeyboardInterrupt):
                print("\nStock Market Data WebSocket stream task cancelled.")
                break
            except Exception as e:
                print(f"\nStock Market Data WebSocket stream encountered an error: {e}. Reconnecting in 30 seconds...")
                await asyncio.sleep(30)

    async def _run_crypto_market_stream(self):
        """Manages the crypto market data websocket connection with a resilient loop."""
        loop = asyncio.get_running_loop()
        while True:
            try:
                print("Connecting to Crypto Market Data WebSocket...")
                self.crypto_market_stream_client = CryptoDataStream(self.KEY, self.SECRET)
                self.current_crypto_subscriptions = set()

                await self._update_subscriptions()

                await loop.run_in_executor(None, self.crypto_market_stream_client.run)
                print("Crypto Market Data WebSocket stream has been closed. Exiting monitoring loop.")
                break
            except (asyncio.CancelledError, KeyboardInterrupt):
                print("\nMarket Data WebSocket stream task cancelled.")
                break
            except Exception as e:
                print(f"\nMarket Data WebSocket stream encountered an error: {e}. Reconnecting in 30 seconds...")
                await asyncio.sleep(30)

    async def _run_trade_stream(self):
        """Manages the trade/account data websocket connection with a resilient loop."""
        loop = asyncio.get_running_loop()
        while True:
            try:
                print("Connecting to Account/Trade Update WebSocket...")
                self.trade_stream_client = TradingStream(self.KEY, self.SECRET, paper=True)
                # The trade stream subscribes to all account updates. The correct method is `subscribe_trade_updates`.
                self.trade_stream_client.subscribe_trade_updates(self.on_order_update)
                await loop.run_in_executor(None, self.trade_stream_client.run)

                print("Account/Trade Update WebSocket stream has been closed. Exiting monitoring loop.")
                break
            except (asyncio.CancelledError, KeyboardInterrupt):
                print("\nAccount/Trade Update WebSocket stream task cancelled.")
                break
            except Exception as e:
                print(f"\nAccount/Trade Update WebSocket stream encountered an error: {e}. Reconnecting in 30 seconds...")
                await asyncio.sleep(30)

    async def start(self, schedule_config: dict):
        """
        Starts the orchestrator's main event loop.
        - Runs an initial analysis cycle.
        - Schedules recurring analysis cycles.
        - Connects to WebSockets for real-time price and order monitoring.
        """
        # 1. Run an initial analysis cycle to get positions and decisions
        await self.run_analysis_cycle()

        # 2. Set up the scheduler for recurring analysis
        self.scheduler.add_job(self.run_analysis_cycle, 'cron', **schedule_config)
        self.scheduler.start()
        print(f"\nAnalysis cycle scheduled. First run will be at the next configured time: {schedule_config}")

        # 3. Start the two WebSocket stream handlers concurrently.
        stock_stream_task = asyncio.create_task(self._run_stock_market_stream())
        crypto_stream_task = asyncio.create_task(self._run_crypto_market_stream())
        trade_stream_task = asyncio.create_task(self._run_trade_stream())

        # This will run until one of the tasks finishes or is cancelled.
        await asyncio.gather(stock_stream_task, crypto_stream_task, trade_stream_task)

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

            # Update managed positions with live data, especially for those loaded from an old file format.
            print("\nUpdating managed positions with live data from account...")
            for live_pos in positions:
                if live_pos.symbol in self.managed_positions:
                    managed_pos = self.managed_positions[live_pos.symbol]
                    # If entry_price or quantity were placeholders, update them with live data.
                    if managed_pos.entry_price == 0.0:
                        managed_pos.entry_price = float(live_pos.avg_entry_price)
                        print(f"  -> Updated {live_pos.symbol} entry price from live data: ${managed_pos.entry_price:.2f}")
                    if managed_pos.quantity == 0.0:
                        managed_pos.quantity = float(live_pos.qty)
                        print(f"  -> Updated {live_pos.symbol} quantity from live data: {managed_pos.quantity}")

            self._sync_ownership_with_account(positions)

            all_decisions = {'buys': [], 'sells': []}

            # Gather decisions from all agents (run blocking analysis in executor)
            for agent in self.agents:
                print(f"\n--- Getting decisions from agent: {agent.name} ---")
                # An agent "owns" a ticker if it's in an OPEN or PENDING_SELL state.
                owned_tickers = {
                    ticker for ticker, pos in self.managed_positions.items()
                    if getattr(pos, 'agent_name', None) == agent.name and pos.state in [PositionState.OPEN, PositionState.PENDING_SELL]
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
            self._save_managed_positions()
            print("\n" + "="*20 + " Scheduled Analysis Cycle Complete " + "="*20)

    async def shutdown(self):
        """Gracefully shuts down all components."""
        print("\n--- Shutting Down Orchestrator ---")
        if self.scheduler.running:
            self.scheduler.shutdown(wait=False)
            print("Scheduler has been shut down.")

        # The stream clients' `run()` method will exit when `close()` is called.
        print("Closing WebSocket streams...")
        await self.stock_market_stream_client.close()
        await self.crypto_market_stream_client.close()
        await self.trade_stream_client.close()
        print("WebSocket stream clients have been closed.")

        # Save final state
        self._save_managed_positions()

    async def on_trade(self, trade):
        """
        Callback for real-time trade updates from the WebSocket.
        Handles real-time SL/TP and re-entry checks.
        """
        symbol = trade.symbol
        price = trade.price

        position = self.managed_positions.get(symbol)
        if not position:
            return

        # Prevent acting on a position that is already pending a state change
        if position.state in [PositionState.PENDING_SELL, PositionState.PENDING_REBUY]:
            return

        action = position.check_price(price)

        if action == 'SELL' and position.state == PositionState.OPEN:
            position.transition_on_sell_submit(price)
            await self._execute_sell(symbol)

        elif action == 'REBUY' and position.state == PositionState.COOLING_DOWN:
            position.transition_on_rebuy_submit()
            await self._execute_rebuy(position)

    async def on_order_update(self, trade_update):
        """Callback for real-time order updates from the WebSocket."""
        # The trade_update object is a wrapper. The actual order data is in the 'order' attribute.
        order = trade_update.order

        # We only care about final fills for state transitions.
        if order.status != 'filled':
            return

        symbol = order.symbol
        position = self.managed_positions.get(symbol)
        if not position:
            return

        print(f"\nReceived FILL confirmation for {order.side} order on {symbol} (Status: {order.status}).")

        if order.side == 'sell' and position.state == PositionState.PENDING_SELL:
            position.transition_on_sell_fill()
            print(f"  -> Position for {symbol} is now in COOLING_DOWN.")
            self._save_managed_positions()

        elif order.side == 'buy':
            if position.state == PositionState.PENDING_REBUY:
                # This is a re-buy fill
                agent = self._get_agent_by_name(getattr(position, 'agent_name', None))
                if not agent:
                    print(f"  -> CRITICAL: Could not find agent '{position.agent_name}' for rebuy of {symbol}. Cannot set SL/TP.")
                    del self.managed_positions[symbol] # Failsafe
                    return

                new_sl_price, new_tp_price = await self._calculate_price_targets(agent, symbol, float(order.filled_avg_price))

                position.transition_on_rebuy_fill(
                    entry_price=float(order.filled_avg_price),
                    quantity=float(order.filled_qty),
                    stop_loss_price=new_sl_price,
                    take_profit_price=new_tp_price
                )
                print(f"  -> Re-buy for {symbol} successful. Position is now OPEN.")
                self._save_managed_positions()
            else:
                # This is a fill for a new position from the analysis cycle
                print(f"  -> New position for {symbol} filled. Management is active.")
                # The ManagedPosition object was already created in _execute_buy
                # We just need to update it with the actual fill details.
                position.entry_price = float(order.filled_avg_price)
                position.quantity = float(order.filled_qty)
                self._save_managed_positions()

    async def _resolve_and_execute_trades(self, all_decisions):
        """Processes sell and buy decisions, handling ownership and conflicts."""
        
        # --- Process Sells First ---
        print("\n--- Resolving and Executing Sells ---")
        for agent, ticker_to_sell in all_decisions['sells']:
            position = self.managed_positions.get(ticker_to_sell)
            if position and getattr(position, 'agent_name', None) == agent.name and position.state == PositionState.OPEN:
                print(f"Decision: {agent.name} to SELL {ticker_to_sell} -> APPROVED")
                agent_for_sell = self._get_agent_by_name(getattr(position, 'agent_name', None))
                if not agent_for_sell:
                    print(f"  -> CRITICAL: Could not find agent for selling {ticker_to_sell}. Cannot get latest price. Skipping.")
                    continue

                # Use the current price for the transition reason, though it will be sold at market
                latest_price = await self._get_latest_price(agent_for_sell, ticker_to_sell) or position.entry_price
                position.transition_on_sell_submit(latest_price)
                await self._execute_sell(ticker_to_sell)
            elif position: # It's managed, but not by this agent or not in an open state
                print(f"Decision: {agent.name} to SELL {ticker_to_sell} -> DENIED (Not owned by agent).")

        # --- Process Buys ---
        print("\n--- Resolving and Executing Buys ---")
        sorted_buys = sorted(all_decisions['buys'], key=lambda item: item[1]['ranking_strength'], reverse=True)

        for agent, pick in sorted_buys:
            ticker = pick['ticker']
            notional = pick['notional_value']

            # Check if already owned
            if ticker in self.managed_positions:
                position = self.managed_positions[ticker]
                print(f"Decision: {agent.name} to BUY {ticker} -> DENIED (Position already managed, state: {position.state.name}).")
                continue

            latest_price = await self._get_latest_price(agent, ticker)
            if not latest_price:
                print(f"  -> Could not get latest price for {ticker}. Skipping buy.")
                continue

            # Pass the agent's pick data to the target calculator
            stop_loss_price, take_profit_price = await self._calculate_price_targets(agent, ticker, latest_price, pick)

            print(f"Decision: {agent.name} to BUY {ticker} for ${notional:.2f} -> APPROVED")
            await self._execute_buy(agent, ticker, notional, latest_price, stop_loss_price, take_profit_price)

    async def _calculate_price_targets(self, agent: TradingAgent, ticker: str, price: float, pick_data: dict = None) -> tuple[float | None, float | None]:
        """Calculates stop-loss and take-profit prices based on strategy type."""
        stop_loss_price = None
        take_profit_price = None

        # Dynamic Target Logic for HMM Strategy
        if isinstance(agent.strategy, HMMStrategy) and pick_data and pick_data.get('predicted_state_mean_return', 0) > 0:
            # DYNAMIC STOP-LOSS: Based on 1x the state's std dev (expected volatility/risk)
            sl_pct = pick_data.get('predicted_state_std_return', 0) # Default to 0 if not present
            
            if sl_pct > 0:
                # Set SL based on 1 standard deviation
                stop_loss_price = price * (1 - sl_pct)
                print(f"  -> Using DYNAMIC stop-loss for {ticker} (1x StDev). Target: ${stop_loss_price:.2f} ({sl_pct:.2%})")

                # DYNAMIC TAKE-PROFIT: Based on 2x the state's std dev (2:1 risk/reward ratio)
                tp_pct = sl_pct * 2
                take_profit_price = price * (1 + tp_pct)
                print(f"  -> Using DYNAMIC take-profit for {ticker} (2x StDev). Target: ${take_profit_price:.2f} ({tp_pct:.2%})")
            else:
                # Fallback to static percentages if dynamic std dev is not available
                print("  -> HMM state std dev not available. Falling back to static SL/TP.")
                if agent.stop_loss_pct:
                    stop_loss_price = price * (1 - agent.stop_loss_pct)
                    print(f"  -> Using STATIC stop-loss for {ticker}. Target: ${stop_loss_price:.2f} ({agent.stop_loss_pct:.2%})")
                if agent.take_profit_pct:
                    take_profit_price = price * (1 + agent.take_profit_pct)
                    print(f"  -> Using STATIC take-profit for {ticker}. Target: ${take_profit_price:.2f} ({agent.take_profit_pct:.2%})")

        # Static Target Logic for all other strategies
        else:
            if agent.take_profit_pct:
                take_profit_price = price * (1 + agent.take_profit_pct)
                print(f"  -> Using STATIC take-profit for {ticker}. Target: ${take_profit_price:.2f} ({agent.take_profit_pct:.2%})")
            if agent.stop_loss_pct:
                stop_loss_price = price * (1 - agent.stop_loss_pct)
                print(f"  -> Using STATIC stop-loss for {ticker}. Target: ${stop_loss_price:.2f} ({agent.stop_loss_pct:.2%})")
        
        return stop_loss_price, take_profit_price

    async def _execute_sell(self, ticker: str) -> bool:
        """Executes a sell order for the given ticker."""
        print(f"  -> Submitting SELL order for {ticker}.")
        try:
            loop = asyncio.get_running_loop()
            # This closes the entire position.
            await loop.run_in_executor(None, self.trading_client.close_position, ticker)
            print(f"  -> Successfully submitted SELL order for {ticker}.")
            return True
        except APIError as e:
            print(f"  -> Failed to close position for {ticker}. Reason: {e}")
            return False

    async def _execute_buy(self, agent: TradingAgent, ticker: str, notional_value: float, entry_price: float, stop_loss_price: float | None, take_profit_price: float | None) -> bool:
        """Executes a buy order for the given ticker and notional value."""
        if notional_value < 1:
            print(f"  -> Skipping BUY for {ticker}, notional value ${notional_value:.2f} is less than $1.")
            return False

        print(f"  -> Submitting BUY order for {ticker} (Notional: ${notional_value:.2f}).")
        try:
            # Crypto orders must use GTC (Good 'Til Canceled) time in force.
            tif = TimeInForce.GTC if agent.asset_class == 'crypto' else TimeInForce.DAY
            market_order_data = MarketOrderRequest(
                symbol=ticker,
                notional=notional_value,
                side=OrderSide.BUY,
                time_in_force=tif
            )
            loop = asyncio.get_running_loop()
            # Pass market_order_data as a positional argument, as run_in_executor
            # does not accept keyword arguments for the target function.
            await loop.run_in_executor(None, self.trading_client.submit_order, market_order_data)
            
            # Create a ManagedPosition object to track this new position.
            # The quantity and final entry price will be updated by on_order_update.
            position = ManagedPosition(
                symbol=ticker,
                entry_price=entry_price, # Use latest price as estimate
                quantity=notional_value / entry_price, # Estimate
                stop_loss_price=stop_loss_price,
                take_profit_price=take_profit_price
            )
            position.agent_name = agent.name
            self.managed_positions[ticker] = position

            print(f"  -> Successfully submitted BUY order for {ticker}.")
            return True
        except APIError as e:
            print(f"  -> Failed to submit order for {ticker}. Reason: {e}")
            return False

    async def _execute_rebuy(self, position: ManagedPosition) -> bool:
        """Executes a buy order to re-enter a position after a cooldown."""
        # Re-buy the same quantity that was previously held.
        quantity_to_buy = position.quantity
        if quantity_to_buy <= 0:
            print(f"  -> Skipping REBUY for {position.symbol}, invalid quantity {quantity_to_buy}.")
            return False

        print(f"  -> Submitting REBUY order for {position.symbol} (Qty: {quantity_to_buy}).")
        try:
            agent = self._get_agent_by_name(getattr(position, 'agent_name', None))
            if not agent:
                print(f"  -> CRITICAL: Could not find agent for rebuy of {position.symbol}. Cannot determine time_in_force. Skipping.")
                return False

            # Crypto orders must use GTC (Good 'Til Canceled) time in force.
            tif = TimeInForce.GTC if agent.asset_class == 'crypto' else TimeInForce.DAY

            market_order_data = MarketOrderRequest(
                symbol=position.symbol,
                qty=quantity_to_buy,
                side=OrderSide.BUY,
                time_in_force=tif
            )
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(None, self.trading_client.submit_order, market_order_data)
            return True
        except APIError as e:
            print(f"  -> Failed to submit REBUY order for {position.symbol}. Reason: {e}")
            return False

    async def _get_latest_price(self, agent: TradingAgent, ticker: str) -> float | None:
        """Fetches the latest trade price for a ticker using the agent's data client."""
        try:
            loop = asyncio.get_running_loop()
            if agent.asset_class == 'crypto':
                request_params = CryptoLatestTradeRequest(symbol_or_symbols=ticker)
                # The agent has the correct data client
                latest_trade_map = await loop.run_in_executor(None, agent.data_client.get_crypto_latest_trade, request_params)
            else:
                request_params = LatestTradeRequest(symbol_or_symbols=ticker, feed=DataFeed.IEX)
                latest_trade_map = await loop.run_in_executor(None, agent.data_client.get_stock_latest_trade, request_params)

            if latest_trade_map and ticker in latest_trade_map:
                return latest_trade_map[ticker].price
        except Exception as e:
            print(f"Could not fetch latest price for {ticker}: {e}")
        return None

    async def _update_subscriptions(self):
        """
        Syncs the market data WebSocket subscriptions for both stocks and crypto
        with the current managed positions.
        """
        print("\nUpdating WebSocket market data subscriptions...")

        desired_stock_subs = set()
        desired_crypto_subs = set()

        for symbol, position in self.managed_positions.items():
            agent = self._get_agent_by_name(getattr(position, 'agent_name', None))
            if agent:
                if agent.asset_class == 'crypto':
                    desired_crypto_subs.add(symbol)
                else:  # us_equity
                    desired_stock_subs.add(symbol)

        # --- Sync Stock Subscriptions ---
        to_sub_stock = desired_stock_subs - self.current_stock_subscriptions
        to_unsub_stock = self.current_stock_subscriptions - desired_stock_subs

        if to_sub_stock:
            print(f"  Subscribing to stock trades for: {list(to_sub_stock)}")
            self.stock_market_stream_client.subscribe_trades(self.on_trade, *to_sub_stock)
            self.current_stock_subscriptions.update(to_sub_stock)
        if to_unsub_stock:
            print(f"  Unsubscribing from stock trades for: {list(to_unsub_stock)}")
            self.stock_market_stream_client.unsubscribe_trades(*to_unsub_stock)
            self.current_stock_subscriptions.difference_update(to_unsub_stock)

        # --- Sync Crypto Subscriptions ---
        to_sub_crypto = desired_crypto_subs - self.current_crypto_subscriptions
        to_unsub_crypto = self.current_crypto_subscriptions - desired_crypto_subs

        if to_sub_crypto:
            print(f"  Subscribing to crypto trades for: {list(to_sub_crypto)}")
            self.crypto_market_stream_client.subscribe_trades(self.on_trade, *to_sub_crypto)
            self.current_crypto_subscriptions.update(to_sub_crypto)
        if to_unsub_crypto:
            print(f"  Unsubscribing from crypto trades for: {list(to_unsub_crypto)}")
            self.crypto_market_stream_client.unsubscribe_trades(*to_unsub_crypto)
            self.current_crypto_subscriptions.difference_update(to_unsub_crypto)

        if not (to_sub_stock or to_unsub_stock or to_sub_crypto or to_unsub_crypto):
            print("  Subscriptions are already up to date.")