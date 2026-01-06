"""
USDTRYForecast - Data Viewer
View Yahoo Finance USD/TRY Data
Displays the historical data that will be used for forecasting
"""

PROJECT_NAME = "USDTRYForecast"

import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

def fetch_and_display_data(years=10):
    """Fetch and display USD/TRY exchange rate data"""
    print("="*80)
    print(f"{PROJECT_NAME} - USD/TRY Exchange Rate Data from Yahoo Finance")
    print("="*80)
    
    # Calculate date range
    end_date = datetime.now()
    start_date = end_date - timedelta(days=years * 365)
    
    print(f"\nFetching data from {start_date.date()} to {end_date.date()}...")
    print(f"Ticker: USDTRY=X\n")
    
    try:
        # Fetch USD/TRY data
        ticker = yf.Ticker("USDTRY=X")
        data = ticker.history(start=start_date, end=end_date)
        
        if data.empty:
            print("⚠️  No data retrieved from Yahoo Finance API")
            return None
        
        print("✅ Successfully fetched data from Yahoo Finance!")
        print("\n" + "-"*80)
        print("DATA SUMMARY")
        print("-"*80)
        print(f"Total records: {len(data):,}")
        print(f"Date range: {data.index[0].date()} to {data.index[-1].date()}")
        print(f"Columns available: {', '.join(data.columns.tolist())}")
        
        # Display first few rows
        print("\n" + "-"*80)
        print("FIRST 10 ROWS")
        print("-"*80)
        print(data.head(10).to_string())
        
        # Display last few rows
        print("\n" + "-"*80)
        print("LAST 10 ROWS")
        print("-"*80)
        print(data.tail(10).to_string())
        
        # Statistics
        print("\n" + "-"*80)
        print("STATISTICAL SUMMARY")
        print("-"*80)
        print(data.describe().to_string())
        
        # Current rate information
        print("\n" + "-"*80)
        print("CURRENT RATE INFORMATION")
        print("-"*80)
        latest = data.iloc[-1]
        print(f"Date: {data.index[-1].date()}")
        print(f"Open:  {latest['Open']:.4f}")
        print(f"High:  {latest['High']:.4f}")
        print(f"Low:   {latest['Low']:.4f}")
        print(f"Close: {latest['Close']:.4f}")
        print(f"Volume: {latest['Volume']:,.0f}")
        
        # Historical statistics
        print("\n" + "-"*80)
        print("HISTORICAL STATISTICS (Close Price)")
        print("-"*80)
        close_prices = data['Close']
        print(f"Minimum:  {close_prices.min():.4f} (Date: {close_prices.idxmin().date()})")
        print(f"Maximum:  {close_prices.max():.4f} (Date: {close_prices.idxmax().date()})")
        print(f"Mean:     {close_prices.mean():.4f}")
        print(f"Median:   {close_prices.median():.4f}")
        print(f"Std Dev:  {close_prices.std():.4f}")
        
        # Year-over-year comparison
        print("\n" + "-"*80)
        print("YEAR-OVER-YEAR COMPARISON")
        print("-"*80)
        data['Year'] = data.index.year
        yearly_avg = data.groupby('Year')['Close'].mean()
        for year, avg_rate in yearly_avg.items():
            print(f"{year}: {avg_rate:.4f}")
        
        # Save to CSV
        csv_filename = 'usd_try_data.csv'
        data.to_csv(csv_filename)
        print(f"\n✅ Data saved to '{csv_filename}'")
        
        # Create visualization
        print("\nGenerating visualization...")
        fig, axes = plt.subplots(2, 1, figsize=(15, 10))
        
        # Plot 1: Close price over time
        axes[0].plot(data.index, data['Close'], linewidth=1.5, color='blue', alpha=0.7)
        axes[0].set_title('USD/TRY Exchange Rate - Close Price (Last 10 Years)', 
                         fontsize=14, fontweight='bold')
        axes[0].set_xlabel('Date')
        axes[0].set_ylabel('USD/TRY Rate')
        axes[0].grid(True, alpha=0.3)
        axes[0].fill_between(data.index, data['Close'], alpha=0.3, color='blue')
        
        # Plot 2: Volume
        axes[1].bar(data.index, data['Volume'], alpha=0.6, color='green', width=1)
        axes[1].set_title('Trading Volume', fontsize=14, fontweight='bold')
        axes[1].set_xlabel('Date')
        axes[1].set_ylabel('Volume')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('usd_try_data_visualization.png', dpi=300, bbox_inches='tight')
        print(f"✅ Visualization saved to 'usd_try_data_visualization.png'")
        plt.show()
        
        return data
        
    except Exception as e:
        print(f"\n❌ Error fetching data: {e}")
        print("\nPossible reasons:")
        print("  - Internet connection issue")
        print("  - Yahoo Finance API temporarily unavailable")
        print("  - Ticker symbol 'USDTRY=X' not found")
        return None


def display_recent_data(data, days=30):
    """Display the most recent N days of data"""
    if data is None:
        return
    
    print("\n" + "="*80)
    print(f"MOST RECENT {days} DAYS")
    print("="*80)
    recent = data.tail(days)
    print(recent.to_string())


if __name__ == "__main__":
    # Fetch and display data
    data = fetch_and_display_data(years=10)
    
    # Display recent 30 days
    if data is not None:
        display_recent_data(data, days=30)
        
        print("\n" + "="*80)
        print("DATA VIEWING COMPLETE")
        print("="*80)
        print("\nFiles created:")
        print("  - usd_try_data.csv (full dataset)")
        print("  - usd_try_data_visualization.png (charts)")

