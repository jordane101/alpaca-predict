"""
A script to calculate and report the realized Profit and Loss (P/L)
for the previous month from an Alpaca trading account.

This script fetches all trade activities (fills) for the last full calendar month,
sums the realized P/L from all sell transactions, and provides a summary.

It can be run on-demand, for example, at the beginning of each new month.
"""

import os
from datetime import datetime, date, timedelta
from decimal import Decimal
import pandas as pd
from dotenv import load_dotenv
import requests
from alpaca.trading.client import TradingClient
from alpaca.common.exceptions import APIError

def get_last_month_range():
    """
    Calculates the start and end datetimes for the previous full calendar month.

    Returns:
        tuple: A tuple containing the start and end datetime objects for the last month.
    """
    today = date.today()
    # Go to the first day of the current month
    first_day_of_current_month = today.replace(day=1)
    # Subtract one day to get the last day of the previous month
    last_day_of_last_month = first_day_of_current_month - timedelta(days=1)
    # Go to the first day of the previous month
    first_day_of_last_month = last_day_of_last_month.replace(day=1)

    # The API expects datetime objects. We'll use start of the first day and end of the last day.
    start_date = datetime.combine(first_day_of_last_month, datetime.min.time())
    end_date = datetime.combine(last_day_of_last_month, datetime.max.time())

    return start_date, end_date

def calculate_monthly_pnl():
    """
    Connects to the Alpaca API, fetches trades from the last month,
    and calculates the total realized P/L.
    """
    # --- Configuration ---
    load_dotenv(".env")
    KEY = os.getenv("PAPER_KEY")
    SECRET = os.getenv("PAPER_SEC")

    if not KEY or not SECRET:
        print("Error: PAPER_KEY and PAPER_SEC must be set in the .env file.")
        return

    # --- Client Initialization ---
    try:
        # Use the modern alpaca-py client, consistent with trader.py
        trading_client = TradingClient(KEY, SECRET, paper=True)
        account = trading_client.get_account()
        print(f"Successfully connected to Alpaca account: {account.account_number}")
    except APIError as e:
        print(f"Error connecting to Alpaca API: {e}")
        return

    # --- Date Range Calculation ---
    start_date, end_date = get_last_month_range()
    print(f"\nCalculating P/L for the period: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")

    # --- Fetching Activities (Direct API Call) ---
    # We use the requests library to make a direct API call to the activities endpoint.
    # This is the most reliable way to get the P/L data calculated by Alpaca,
    # which correctly accounts for shares held from previous periods.
    print("\nFetching activities via direct API call...")
    # Per user feedback, use the specific /FILL endpoint for the initial fetch.
    monthly_activities_url = "https://paper-api.alpaca.markets/v2/account/activities/FILL"
    headers = {
        "APCA-API-KEY-ID": KEY,
        "APCA-API-SECRET-KEY": SECRET,
        "accept": "application/json"
    }
    monthly_params = {
        # activity_types is redundant when using the /FILL endpoint.
        "after": start_date.isoformat() + "Z",
        "until": end_date.isoformat() + "Z"
    }

    try:
        # Use the specific URL and params for the monthly fetch.
        response = requests.get(monthly_activities_url, headers=headers, params=monthly_params)
        response.raise_for_status()  # Raise an exception for bad status codes (4xx or 5xx)
        activities_data = response.json()

        if not activities_data:
            print("No trade activities (fills) found for the specified period.")
            return

        print(f"Found {len(activities_data)} trade activities (fills).")

    except requests.exceptions.RequestException as e:
        print(f"Error fetching activities from Alpaca API: {e}")
        return

    # --- Net Flow P/L Calculation by Asset Class and Asset ---
    pnl_by_class_and_asset = {}

    # 1. Aggregate Buy and Sell volumes for the month
    for activity in activities_data:
        original_symbol = activity['symbol']

        # Per user request, determine asset class based on ticker length.
        # Tickers > 4 chars are options, otherwise equity.
        if len(original_symbol) > 4:
            asset_class = 'us_option'
        else:
            asset_class = 'us_equity'

        # --- Group options tickers under a common symbol ---
        if asset_class == 'us_option':
            # Extract the underlying ticker. Find the first digit.
            first_digit_index = -1
            for i, char in enumerate(original_symbol):
                if char.isdigit():
                    first_digit_index = i
                    break

            if first_digit_index > 0:  # Ensure we found a digit and it's not the first char
                underlying = original_symbol[:first_digit_index]
                symbol = f"{underlying} (OPT)"
            else:  # Fallback if no digit is found or format is unexpected
                symbol = original_symbol
        else:
            symbol = original_symbol

        side = activity['side']
        qty = Decimal(activity['qty'])
        price = Decimal(activity['price'])

        # For options, the notional value is price * qty * 100 (the multiplier)
        if asset_class == 'us_option':
            notional = qty * price * 100
        else:
            notional = qty * price

        # Initialize dictionaries if not present
        class_data = pnl_by_class_and_asset.setdefault(asset_class, {})
        symbol_data = class_data.setdefault(symbol, {
            'P/L': Decimal('0.0'),
            'Qty Bought': Decimal('0.0'),
            'Buy Volume': Decimal('0.0'),
            'Qty Sold': Decimal('0.0'),
            'Sell Volume': Decimal('0.0'),
            'Note': ''
        })

        # Group debit and credit transactions
        if side in ('buy', 'buy_to_cover'):
            symbol_data['Qty Bought'] += qty
            symbol_data['Buy Volume'] += notional
        elif side in ('sell', 'sell_short'):
            symbol_data['Qty Sold'] += qty
            symbol_data['Sell Volume'] += notional

    # 2. Calculate P/L as Net Flow (Sell Volume - Buy Volume) and add notes
    for asset_class, assets in pnl_by_class_and_asset.items():
        for symbol, data in assets.items():
            data['P/L'] = data['Sell Volume'] - data['Buy Volume']
            data['Note'] = "Net flow (Sell Vol - Buy Vol)"

    # --- Reporting ---
    print("\n--- Monthly P/L Report (Net Flow Calculation) ---")
    print("NOTE: P/L is calculated as Total Sell Volume - Total Buy Volume for the month.")

    all_summaries_list = []
    overall_total_pnl = Decimal('0.0')
    overall_total_buy_volume = Decimal('0.0')
    overall_total_sell_volume = Decimal('0.0')

    def calculate_pnl_percent(row):
        if row['Buy Volume'] > 0:
            # Calculate percentage P/L based on the cost (Buy Volume)
            return f"{(row['P/L'] / row['Buy Volume']) * 100:.2f}%"
        elif row['P/L'] > 0:
            # If no buys but there were sells, profit is infinite
            return "inf"
        else:
            return "0.00%"

    # Define friendlier names for asset classes in the report
    asset_class_display_names = {
        'us_equity': 'Equity',
        'us_option': 'Options'
    }

    # Sort asset classes for consistent report order
    sorted_asset_classes = sorted(pnl_by_class_and_asset.keys())

    for asset_class in sorted_asset_classes:
        pnl_by_asset = pnl_by_class_and_asset[asset_class]
        display_name = asset_class_display_names.get(asset_class, asset_class.upper())
        print(f"\n\n{'='*20} Asset Class: {display_name} {'='*20}")

        # Convert the dictionary to a list of records for DataFrame creation
        summary_list = []
        for symbol, data in pnl_by_asset.items():
            record = {'Symbol': symbol, 'Asset Class': asset_class}
            record.update(data)
            summary_list.append(record)

        all_summaries_list.extend(summary_list)

        if not summary_list:
            print("No activities for this asset class.")
            continue

        summary_df = pd.DataFrame(summary_list).sort_values(by='Symbol').reset_index(drop=True)
        summary_df['P/L %'] = summary_df.apply(calculate_pnl_percent, axis=1)

        # Reorder columns for display
        display_df = summary_df[['Symbol', 'P/L', 'P/L %', 'Qty Bought', 'Buy Volume', 'Qty Sold', 'Sell Volume', 'Note']]

        print("\n--- P/L by Asset ---")
        with pd.option_context('display.max_rows', None, 'display.width', 140, 'display.colheader_justify', 'center', 'display.float_format', '{:,.2f}'.format):
            print(display_df.to_string(index=False))

        # --- Subtotal for Asset Class ---
        class_total_pnl = summary_df['P/L'].sum()
        class_total_buy_volume = summary_df['Buy Volume'].sum()
        class_total_sell_volume = summary_df['Sell Volume'].sum()

        overall_total_pnl += class_total_pnl
        overall_total_buy_volume += class_total_buy_volume
        overall_total_sell_volume += class_total_sell_volume

        pnl_percent_str = "0.00%"
        if class_total_buy_volume > 0:
            pnl_percent_str = f"{(class_total_pnl / class_total_buy_volume) * 100:.2f}%"
        elif class_total_pnl > 0:
            pnl_percent_str = "inf"

        print(f"\n--- Subtotal for {display_name} ---")
        print(f"Total Net P/L:      ${class_total_pnl:,.2f}")
        print(f"Total Buy Volume:   ${class_total_buy_volume:,.2f}")
        print(f"Total Sell Volume:  ${class_total_sell_volume:,.2f}")
        print(f"Total P/L %%:        {pnl_percent_str}")
        print("--------------------------")

    # --- Overall Summary ---
    print("\n\n" + "="*20 + " OVERALL SUMMARY " + "="*20)
    pnl_percent_str = "0.00%"
    if overall_total_buy_volume > 0:
        total_percentage_pnl = (overall_total_pnl / overall_total_buy_volume) * 100
        pnl_percent_str = f"{total_percentage_pnl:.2f}%"
    elif overall_total_pnl > 0:
        pnl_percent_str = "inf"

    print(f"Total Net P/L:      ${overall_total_pnl:,.2f}")
    print(f"Total Buy Volume:   ${overall_total_buy_volume:,.2f}")
    print(f"Total Sell Volume:  ${overall_total_sell_volume:,.2f}")
    print(f"Total P/L %%:        {pnl_percent_str}")
    print("="*57)

    # --- Export to CSV ---
    if all_summaries_list:
        report_dir = "reports"
        os.makedirs(report_dir, exist_ok=True)

        # Create a full DataFrame for the CSV export
        full_summary_df = pd.DataFrame(all_summaries_list)
        full_summary_df['P/L %'] = full_summary_df.apply(calculate_pnl_percent, axis=1)
        full_summary_df = full_summary_df[['Asset Class', 'Symbol', 'P/L', 'P/L %', 'Qty Bought', 'Buy Volume', 'Qty Sold', 'Sell Volume', 'Note']].sort_values(by=['Asset Class', 'Symbol'])

        # Format filename with month and P/L
        report_month_str = start_date.strftime('%Y-%m')
        pnl_for_filename = int(overall_total_pnl)
        filename = f"pnl_report_{report_month_str}_PL_{pnl_for_filename}.csv"
        filepath = os.path.join(report_dir, filename)

        try:
            full_summary_df.to_csv(filepath, index=False, float_format='%.2f')
            print(f"\nReport successfully exported to: {filepath}")
        except Exception as e:
            print(f"\nError exporting report to CSV: {e}")

if __name__ == "__main__":
    calculate_monthly_pnl()