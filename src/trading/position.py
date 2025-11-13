"""
Defines the ManagedPosition class, which encapsulates the state
and logic for a single trading position, including re-entry monitoring.

Author - Eli Jordan
Date - 07/29/2025
"""

from enum import Enum, auto

class PositionState(Enum):
    """Represents the current state of a managed position."""
    OPEN = auto()              # Position is currently held and being monitored against SL/TP.
    PENDING_SELL = auto()      # A sell order has been submitted for the open position.
    COOLING_DOWN = auto()      # Position was sold; now monitoring for a re-entry signal.
    PENDING_REBUY = auto()     # Re-entry condition met; a buy order has been submitted.

class CooldownReason(Enum):
    """The reason a position was sold and entered the cooldown state."""
    STOP_LOSS = auto()
    TAKE_PROFIT = auto()

class ManagedPosition:
    """
    Represents and manages the state of a single asset, including open,
    cooldown, and re-entry logic. An instance of this class would be
    managed by the Orchestrator.
    """
    def __init__(self, symbol: str, entry_price: float, quantity: float, stop_loss_price: float | None, take_profit_price: float | None):
        self.symbol = symbol
        self.entry_price = entry_price
        self.quantity = quantity
        self.stop_loss_price = stop_loss_price
        self.take_profit_price = take_profit_price

        self.state = PositionState.OPEN
        self.cooldown_reason: CooldownReason | None = None

    def check_price(self, current_price: float) -> str | None:
        """
        Checks the current price against the position's state and targets.
        This is the core logic for the real-time websocket handler.

        Returns: A string action ('SELL', 'REBUY') or None.
        """
        if self.state == PositionState.OPEN:
            if self.stop_loss_price is not None and current_price <= self.stop_loss_price:
                print(f"[{self.symbol}] Event: Stop-loss triggered at {current_price:.2f} (target: <= {self.stop_loss_price:.2f})")
                return 'SELL'
            if self.take_profit_price is not None and current_price >= self.take_profit_price:
                print(f"[{self.symbol}] Event: Take-profit triggered at {current_price:.2f} (target: >= {self.take_profit_price:.2f})")
                return 'SELL'

        elif self.state == PositionState.COOLING_DOWN:
            # If sold at stop-loss, re-enter if price moves back *above* the SL price.
            if self.cooldown_reason == CooldownReason.STOP_LOSS and self.stop_loss_price is not None and current_price > self.stop_loss_price:
                print(f"[{self.symbol}] Event: Re-entry signal. Price {current_price:.2f} crossed back above SL of {self.stop_loss_price:.2f}")
                return 'REBUY'
            # If sold at take-profit, re-enter if price moves back *below* the TP price.
            elif self.cooldown_reason == CooldownReason.TAKE_PROFIT and self.take_profit_price is not None and current_price < self.take_profit_price:
                print(f"[{self.symbol}] Event: Re-entry signal. Price {current_price:.2f} crossed back below TP of {self.take_profit_price:.2f}")
                return 'REBUY'

        return None

    def transition_on_sell_submit(self, sell_price: float):
        """Transitions state when a sell order is submitted."""
        if self.state != PositionState.OPEN: return

        if self.stop_loss_price is not None and sell_price <= self.stop_loss_price:
            self.cooldown_reason = CooldownReason.STOP_LOSS
        else:
            self.cooldown_reason = CooldownReason.TAKE_PROFIT
        
        self.state = PositionState.PENDING_SELL
        print(f"[{self.symbol}] State -> PENDING_SELL (Reason: {self.cooldown_reason.name})")

    def transition_on_sell_fill(self):
        """Transitions state after a sell order is confirmed filled."""
        if self.state != PositionState.PENDING_SELL: return
        self.state = PositionState.COOLING_DOWN
        print(f"[{self.symbol}] State -> COOLING_DOWN. Monitoring for re-entry into the 'middle ground'.")

    def transition_on_rebuy_submit(self):
        """Transitions state when a rebuy order is submitted."""
        if self.state != PositionState.COOLING_DOWN: return
        self.state = PositionState.PENDING_REBUY
        print(f"[{self.symbol}] State -> PENDING_REBUY.")

    def transition_on_rebuy_fill(self, entry_price: float, quantity: float, stop_loss_price: float | None, take_profit_price: float | None):
        """Resets the state to OPEN after a successful rebuy."""
        self.entry_price, self.quantity = entry_price, quantity
        self.stop_loss_price, self.take_profit_price = stop_loss_price, take_profit_price
        self.state = PositionState.OPEN
        self.cooldown_reason = None
        print(f"[{self.symbol}] State -> OPEN. Re-buy successful.")