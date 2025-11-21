#!/usr/bin/env python3
"""
Config-Driven Live Trading Bot

Reads configuration from config/trader_config.yaml
Manages multiple trading agents with different strategies.

Usage:
    python scripts/trader_with_config.py
    python scripts/trader_with_config.py --config my_config.yaml
    python scripts/trader_with_config.py --dry-run  # Test without executing trades

Author - Eli Jordan
Date - 11/13/2025
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import asyncio
import yaml
import argparse
from src.trading.orchestrator import Orchestrator
from src.trading.strategies import HMMStrategy, DonchianBreakoutStrategy


def load_config(config_path=None):
    """Load trading configuration from YAML file."""
    if config_path is None:
        config_path = Path(__file__).parent.parent / 'config' / 'trader_config.yaml'
    else:
        config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    print(f"✓ Loaded config from: {config_path}")
    return config


def create_strategy(strategy_config):
    """Create strategy instance from config."""
    strategy_type = strategy_config.get('type', 'HMM')
    
    if strategy_type == 'HMM':
        return HMMStrategy(
            n_components=strategy_config.get('n_components', 3),
            model_order=strategy_config.get('model_order', 1),
            optimize_order=strategy_config.get('optimize_order', False),
            max_order_to_test=strategy_config.get('max_order_to_test', 5),
            ranking_metric=strategy_config.get('ranking_metric', 'sharpe'),
            retrain_max_age_days=strategy_config.get('retrain_max_age_days', 1),
            walk_forward_window=strategy_config.get('walk_forward_window', 252),
            retrain_period=strategy_config.get('retrain_period', 63),
            use_causality_filter=strategy_config.get('use_causality_filter', False),
            causality_significance=strategy_config.get('causality_significance', 0.05),
            use_causal_features=strategy_config.get('use_causal_features', True),
            causal_dag_file=strategy_config.get('causal_dag_file'),
            optimize_n_components=strategy_config.get('optimize_n_components', True),
            n_components_range=tuple(strategy_config.get('n_components_range', [2, 4]))
        )
    elif strategy_type == 'Donchian':
        # Add Donchian strategy if configured
        return DonchianBreakoutStrategy()
    else:
        raise ValueError(f"Unknown strategy type: {strategy_type}")


def build_agent_configs(config):
    """Build agent configuration list from YAML config."""
    agent_configs = []
    
    for agent_cfg in config['agents']:
        # Skip disabled agents
        if not agent_cfg.get('enabled', True):
            print(f"  Skipping disabled agent: {agent_cfg['name']}")
            continue
        
        # Create strategy instance
        strategy = create_strategy(agent_cfg['strategy'])
        
        # Build agent config dict
        agent_config = {
            'name': agent_cfg['name'],
            'strategy': strategy,
            'max_positions': agent_cfg.get('max_positions', 10),
            'total_allocation_pct': agent_cfg.get('total_allocation_pct', 0.5),
            'stop_loss_pct': agent_cfg.get('stop_loss_pct', 0.05),
            'take_profit_pct': agent_cfg.get('take_profit_pct', 0.10),
            'max_analysis_workers': agent_cfg.get('max_analysis_workers', 4)
        }
        
        # Add stop-loss and take-profit SD multipliers from strategy config
        strategy_cfg = agent_cfg.get('strategy', {})
        agent_config['stop_loss_sd_multiplier'] = strategy_cfg.get('stop_loss_sd_multiplier', 1.0)
        agent_config['take_profit_sd_multiplier'] = strategy_cfg.get('take_profit_sd_multiplier', 2.0)
        
        # Add asset_class if specified (for crypto agents)
        if 'asset_class' in agent_cfg:
            agent_config['asset_class'] = agent_cfg['asset_class']
        
        agent_configs.append(agent_config)
        print(f"  ✓ Configured agent: {agent_cfg['name']}")
    
    return agent_configs


def build_schedule_config(config):
    """Build schedule configuration from YAML config."""
    schedule_cfg = config.get('schedule', {})
    
    return {
        'hour': schedule_cfg.get('hour', '9,15'),
        'minute': schedule_cfg.get('minute', '45'),
        'timezone': schedule_cfg.get('timezone', 'America/New_York')
    }


def print_config_summary(config, agent_configs, schedule_config):
    """Print a summary of the loaded configuration."""
    print("\n" + "="*70)
    print("  TRADING BOT CONFIGURATION SUMMARY")
    print("="*70)
    
    # Schedule
    print(f"\n📅 Schedule:")
    print(f"   Run at: {schedule_config['hour']}:{schedule_config['minute']} {schedule_config['timezone']}")
    
    # Agents
    print(f"\n🤖 Agents: {len(agent_configs)}")
    total_allocation = 0
    for agent_cfg in agent_configs:
        allocation = agent_cfg['total_allocation_pct']
        total_allocation += allocation
        print(f"   • {agent_cfg['name']}: {allocation:.0%} allocation, {agent_cfg['max_positions']} max positions")
    
    print(f"\n   Total Allocation: {total_allocation:.0%}")
    if total_allocation > 1.0:
        print(f"   ⚠️  WARNING: Total allocation exceeds 100%!")
    
    # Risk Management
    risk_cfg = config.get('risk', {})
    print(f"\n⚠️  Risk Management:")
    print(f"   Max Total Allocation: {risk_cfg.get('max_total_allocation', 0.9):.0%}")
    print(f"   Min Cash Reserve: {risk_cfg.get('min_cash_reserve', 0.1):.0%}")
    print(f"   Max Trades/Day: {risk_cfg.get('max_trades_per_day', 50)}")
    
    # API Settings
    api_cfg = config.get('api', {})
    print(f"\n🔌 API Settings:")
    print(f"   Paper Trading: {api_cfg.get('paper_trading', True)}")
    print(f"   Data Feed: {api_cfg.get('data_feed', 'IEX')}")
    print(f"   Order Type: {api_cfg.get('order_type', 'market')}")
    
    # Advanced
    advanced_cfg = config.get('advanced', {})
    print(f"\n⚙️  Advanced:")
    print(f"   Debug Mode: {advanced_cfg.get('debug', False)}")
    print(f"   Dry Run: {advanced_cfg.get('dry_run', False)}")
    
    print("\n" + "="*70 + "\n")


async def main(orchestrator, schedule_config):
    """The main async entry point for the trader."""
    try:
        print("--- Starting Orchestrator Event Loop ---")
        print("Press Ctrl+C to stop.")
        # Shield the orchestrator's start method from cancellation
        await asyncio.shield(orchestrator.start(schedule_config=schedule_config))
    except asyncio.CancelledError:
        print("\nShutdown signal received. Proceeding with graceful shutdown.")
        pass
    finally:
        # Ensure graceful shutdown
        await orchestrator.shutdown()


def run_trading_bot(config_path=None, dry_run=False):
    """
    Run the trading bot with the specified configuration.
    
    Args:
        config_path: Path to YAML config file (None = use default)
        dry_run: If True, log trades without executing
    """
    # Load configuration
    print("\n" + "="*70)
    print("  INITIALIZING TRADING BOT")
    print("="*70 + "\n")
    
    config = load_config(config_path)
    
    # Override dry_run if specified via command line
    if dry_run:
        if 'advanced' not in config:
            config['advanced'] = {}
        config['advanced']['dry_run'] = True
        print("⚠️  DRY RUN MODE: Trades will be logged but NOT executed\n")
    
    # Build agent configurations
    print("Building agent configurations...")
    agent_configs = build_agent_configs(config)
    
    if not agent_configs:
        print("❌ No enabled agents found in config. Exiting.")
        return
    
    # Build schedule configuration
    schedule_config = build_schedule_config(config)
    
    # Print configuration summary
    print_config_summary(config, agent_configs, schedule_config)
    
    # Validate total allocation
    total_allocation = sum(a['total_allocation_pct'] for a in agent_configs)
    max_allocation = config.get('risk', {}).get('max_total_allocation', 1.0)
    
    if total_allocation > max_allocation:
        print(f"❌ ERROR: Total allocation ({total_allocation:.0%}) exceeds max allowed ({max_allocation:.0%})")
        print("   Please adjust agent allocations in config file.")
        return
    
    # Initialize orchestrator
    print("--- Initializing Orchestrator ---")
    orchestrator = Orchestrator(agent_configs=agent_configs)
    
    # Start the orchestrator
    try:
        asyncio.run(main(orchestrator, schedule_config))
    except KeyboardInterrupt:
        print("\n✓ Trading bot stopped.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Config-driven live trading bot',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with default config
  python scripts/trader_with_config.py
  
  # Run with custom config
  python scripts/trader_with_config.py --config my_config.yaml
  
  # Test without executing trades
  python scripts/trader_with_config.py --dry-run
  
  # Use custom config in dry-run mode
  python scripts/trader_with_config.py --config aggressive.yaml --dry-run
        """
    )
    
    parser.add_argument(
        '--config',
        type=str,
        help='Path to YAML configuration file (default: config/trader_config.yaml)',
        default=None
    )
    
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Log trades without executing (test mode)',
        default=False
    )
    
    args = parser.parse_args()
    
    run_trading_bot(config_path=args.config, dry_run=args.dry_run)
