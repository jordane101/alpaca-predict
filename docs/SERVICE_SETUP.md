# Service & Automation Setup Guide

## Systemctl Service Setup

The project includes a systemd service configuration to run the trading bot continuously.

### Files

- **`alpaca-trader.service`** - Systemd service file
- **`setup_service.sh`** - Helper script to install/uninstall/test the service
- **`scripts/run_trader.sh`** - Cron job script (alternative to systemctl)
- **`scripts/run_monthly_report.sh`** - Monthly P&L report script

## Quick Start

### 1. Test Configuration

Before installing, verify everything is configured correctly:

```bash
./setup_service.sh test
```

This will check:
- ✓ Service file exists
- ✓ Trader script exists
- ✓ Virtual environment exists
- ✓ .env file with API keys exists
- ✓ Log directory exists
- ✓ Python imports work correctly

### 2. Install Service

If the test passes, install the service:

```bash
sudo ./setup_service.sh install
```

This will:
- Copy service file to `/etc/systemd/system/`
- Reload systemd daemon
- Enable service to start on boot

### 3. Start the Service

```bash
sudo systemctl start alpaca-trader
```

### 4. Check Status

```bash
# Using the helper script
./setup_service.sh status

# Or directly with systemctl
sudo systemctl status alpaca-trader
```

### 5. View Logs

```bash
# Real-time logs
tail -f data/logs/trader_service.log

# Error logs
tail -f data/logs/trader_service_error.log

# Last 100 lines
tail -n 100 data/logs/trader_service.log
```

## Service Management Commands

```bash
# Start service
sudo systemctl start alpaca-trader

# Stop service
sudo systemctl stop alpaca-trader

# Restart service
sudo systemctl restart alpaca-trader

# Enable start on boot
sudo systemctl enable alpaca-trader

# Disable start on boot
sudo systemctl disable alpaca-trader

# View status
sudo systemctl status alpaca-trader

# View journal logs
sudo journalctl -u alpaca-trader -f
```

## Uninstall Service

To completely remove the service:

```bash
sudo ./setup_service.sh uninstall
```

## Service Configuration

The service file (`alpaca-trader.service`) includes:

```ini
[Unit]
Description=Alpaca Trading Bot
After=network.target

[Service]
Type=simple
User=eli
WorkingDirectory=/home/eli/alpaca-predict
Environment="PATH=/home/eli/alpaca-predict/.venv/bin:..."
ExecStart=/home/eli/alpaca-predict/.venv/bin/python3 /home/eli/alpaca-predict/scripts/trader.py
Restart=on-failure
RestartSec=10
StandardOutput=append:/home/eli/alpaca-predict/data/logs/trader_service.log
StandardError=append:/home/eli/alpaca-predict/data/logs/trader_service_error.log
```

**Key Features:**
- Runs as user `eli`
- Automatically restarts on failure (after 10 seconds)
- Logs to `data/logs/trader_service.log`
- Starts after network is available

## Alternative: Cron Job

If you prefer cron over systemctl, use the shell script:

### Setup Cron Job

```bash
# Edit crontab
crontab -e

# Add entry to run every day at market open (9:30 AM ET)
30 9 * * 1-5 /home/eli/alpaca-predict/scripts/run_trader.sh

# Or run every hour during market hours
0 9-16 * * 1-5 /home/eli/alpaca-predict/scripts/run_trader.sh
```

The script:
- Activates virtual environment
- Runs `scripts/trader.py`
- Logs to `data/logs/trader_log_YYYY-MM-DD.log`

## Monthly P&L Report

### Manual Execution

Run the monthly report manually:

```bash
# Using shell script (recommended)
./scripts/run_monthly_report.sh

# Or directly with Python
python3 scripts/monthly_pnl_reporter.py
```

### Automated Monthly Reports

Setup a cron job to run on the 1st of each month:

```bash
# Edit crontab
crontab -e

# Add entry to run on 1st day of month at 9:00 AM
0 9 1 * * /home/eli/alpaca-predict/scripts/run_monthly_report.sh
```

The report will:
- Calculate P&L for the previous month
- Generate detailed trade summary
- Save report to `data/reports/pnl_report_YYYY-MM-DD.log`
- Display summary in terminal

### Report Output

The monthly report includes:
- Total realized P&L
- Number of trades (buys/sells)
- Breakdown by ticker
- Trade-by-trade details
- Summary statistics

Example output:
```
=== ALPACA MONTHLY P/L REPORT ===
Period: 2024-10-01 to 2024-10-31

Total Realized P/L: $1,234.56
Total Trades: 45
  - Buys: 23
  - Sells: 22

Top Performers:
  AAPL: $456.78 (15 trades)
  NVDA: $321.45 (8 trades)
  ...
```

## Troubleshooting

### Service Won't Start

1. **Check test output:**
   ```bash
   ./setup_service.sh test
   ```

2. **Check service status:**
   ```bash
   sudo systemctl status alpaca-trader
   ```

3. **View error logs:**
   ```bash
   tail -n 50 data/logs/trader_service_error.log
   ```

4. **Check journal:**
   ```bash
   sudo journalctl -u alpaca-trader -n 50
   ```

### Import Errors

If you see import errors:
```bash
# Test imports manually
cd /home/eli/alpaca-predict
.venv/bin/python3 -c "from src.trading import Orchestrator; print('OK')"
```

### Permission Issues

If logs aren't being written:
```bash
# Ensure log directory is writable
chmod -R 755 data/logs
chown -R eli:eli data/logs
```

### Path Issues

If the service can't find files:
1. Check `WorkingDirectory` in service file matches project location
2. Verify paths in `scripts/trader.py` use absolute paths or `sys.path.insert(0, ...)`

## Log Rotation

To prevent logs from growing too large, setup log rotation:

```bash
# Create logrotate config
sudo nano /etc/logrotate.d/alpaca-trader

# Add configuration:
/home/eli/alpaca-predict/data/logs/*.log {
    daily
    rotate 30
    compress
    delaycompress
    missingok
    notifempty
    create 0644 eli eli
}
```

## Best Practices

1. **Test before deploying**: Always run `./setup_service.sh test` first
2. **Monitor logs**: Regularly check logs for errors
3. **Backup strategies**: Keep backups of working configurations
4. **Resource monitoring**: Monitor CPU/memory usage
5. **Market hours**: Consider only running during market hours to save resources

## Security Considerations

1. **.env file**: Ensure `.env` is readable only by your user
   ```bash
   chmod 600 .env
   ```

2. **Service user**: Service runs as `eli` (not root) for security

3. **API keys**: Never commit `.env` to git (already in `.gitignore`)

4. **Log permissions**: Logs should only be readable by your user

## Summary

✅ **Systemctl Service**: Best for continuous 24/7 operation  
✅ **Cron Job**: Good for scheduled trading during market hours  
✅ **Monthly Report**: Automated P&L tracking  
✅ **Logging**: All output captured to files  
✅ **Auto-restart**: Service automatically restarts on failure  

Choose the method that best fits your trading strategy!
