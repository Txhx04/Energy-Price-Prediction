import os
import sys
from datetime import datetime, timedelta

# Ensure we can import from backend
sys.path.append(os.path.dirname(__file__))

from dotenv import load_dotenv
from entsoe_fetcher import ENTSOEApiFetcher

def get_next_24h_highs():
    load_dotenv()
    api_key = os.environ.get("ENTSOE_API_KEY")
    
    if not api_key or api_key == "your_key_here":
        print("ERROR: Valid ENTSOE_API_KEY not found in .env")
        return

    fetcher = ENTSOEApiFetcher()
    
    # Get today's date for the 24-hour day-ahead forecast 
    # (Tomorrow's isn't published until 13:00 CET, and it's currently 10:45 CET)
    target_date = datetime.now().strftime("%Y-%m-%d")
    print(f"Fetching True Day-Ahead Price Forecast for {target_date}...\n")
    
    try:
        prices_df = fetcher.get_day_ahead_prices(target_date)
        prices = prices_df['price_eur_mwh'].tolist()
        
        print("Hour | Price (EUR/MWh)")
        print("----------------------")
        
        max_price = max(prices)
        max_hour = prices.index(max_price)
        
        for hour, price in enumerate(prices):
            marker = " (DAILY HIGH)" if hour == max_hour else ""
            print(f"{hour:02d}:00 | {price:.2f}{marker}")
            
        print("\nSUMMARY:")
        print(f"Average Price: {sum(prices)/len(prices):.2f} EUR/MWh")
        print(f"Peak Price:    {max_price:.2f} EUR/MWh @ {max_hour:02d}:00")
        
    except Exception as e:
        import traceback
        print(f"Failed to fetch forecast: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    get_next_24h_highs()
