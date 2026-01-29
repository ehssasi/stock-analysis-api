"""
Alpha Vantage data fetcher for stock analysis
"""
import requests
import pandas as pd
from datetime import datetime, timedelta
import time

# Alpha Vantage API key
API_KEY = 'H2I70Z8BPMEWDX4Q'
BASE_URL = 'https://www.alphavantage.co/query'

def get_quote(symbol):
    """Get real-time quote for a symbol"""
    try:
        params = {
            'function': 'GLOBAL_QUOTE',
            'symbol': symbol,
            'apikey': API_KEY
        }

        response = requests.get(BASE_URL, params=params, timeout=10)
        data = response.json()

        if 'Global Quote' not in data:
            return None

        quote = data['Global Quote']

        return {
            'symbol': symbol.upper(),
            'company_name': symbol.upper(),  # Alpha Vantage free tier doesn't include company name
            'current_price': float(quote.get('05. price', 0)),
            'change': float(quote.get('09. change', 0)),
            'change_pct': float(quote.get('10. change percent', '0').replace('%', '')),
            'volume': int(quote.get('06. volume', 0)),
            'timestamp': datetime.now().isoformat(),
            'source': 'Alpha Vantage'
        }
    except Exception as e:
        print(f"Error fetching quote from Alpha Vantage: {e}")
        return None

def get_daily_data(symbol, outputsize='compact'):
    """
    Get daily historical data
    compact = last 100 days
    full = 20+ years
    """
    try:
        params = {
            'function': 'TIME_SERIES_DAILY',
            'symbol': symbol,
            'outputsize': outputsize,
            'apikey': API_KEY
        }

        response = requests.get(BASE_URL, params=params, timeout=10)
        data = response.json()

        if 'Time Series (Daily)' not in data:
            return None

        time_series = data['Time Series (Daily)']

        # Convert to pandas DataFrame
        df_data = []
        for date_str, values in time_series.items():
            df_data.append({
                'Date': pd.to_datetime(date_str),
                'Open': float(values['1. open']),
                'High': float(values['2. high']),
                'Low': float(values['3. low']),
                'Close': float(values['4. close']),
                'Volume': int(values['5. volume'])
            })

        df = pd.DataFrame(df_data)
        df.set_index('Date', inplace=True)
        df.sort_index(inplace=True)

        return df

    except Exception as e:
        print(f"Error fetching daily data from Alpha Vantage: {e}")
        return None

def get_company_overview(symbol):
    """Get company fundamentals and overview"""
    try:
        params = {
            'function': 'OVERVIEW',
            'symbol': symbol,
            'apikey': API_KEY
        }

        response = requests.get(BASE_URL, params=params, timeout=10)
        data = response.json()

        if not data or 'Symbol' not in data:
            return None

        return {
            'company_name': data.get('Name', symbol),
            'sector': data.get('Sector', 'N/A'),
            'industry': data.get('Industry', 'N/A'),
            'market_cap': float(data.get('MarketCapitalization', 0)),
            'pe_ratio': float(data.get('PERatio', 0) or 0),
            'forward_pe': float(data.get('ForwardPE', 0) or 0),
            'eps': float(data.get('EPS', 0) or 0),
            'profit_margin': float(data.get('ProfitMargin', 0) or 0) * 100,
            'roe': float(data.get('ReturnOnEquityTTM', 0) or 0) * 100,
            'revenue': float(data.get('RevenueTTM', 0)),
            'analyst_target': float(data.get('AnalystTargetPrice', 0) or 0),
            'dividend_yield': float(data.get('DividendYield', 0) or 0) * 100,
            '52_week_high': float(data.get('52WeekHigh', 0) or 0),
            '52_week_low': float(data.get('52WeekLow', 0) or 0),
        }

    except Exception as e:
        print(f"Error fetching company overview from Alpha Vantage: {e}")
        return None

def get_intraday_data(symbol, interval='5min'):
    """Get intraday data for day trading analysis"""
    try:
        params = {
            'function': 'TIME_SERIES_INTRADAY',
            'symbol': symbol,
            'interval': interval,
            'apikey': API_KEY
        }

        response = requests.get(BASE_URL, params=params, timeout=10)
        data = response.json()

        key = f'Time Series ({interval})'
        if key not in data:
            return None

        time_series = data[key]

        # Convert to pandas DataFrame
        df_data = []
        for datetime_str, values in time_series.items():
            df_data.append({
                'DateTime': pd.to_datetime(datetime_str),
                'Open': float(values['1. open']),
                'High': float(values['2. high']),
                'Low': float(values['3. low']),
                'Close': float(values['4. close']),
                'Volume': int(values['5. volume'])
            })

        df = pd.DataFrame(df_data)
        df.set_index('DateTime', inplace=True)
        df.sort_index(inplace=True)

        return df

    except Exception as e:
        print(f"Error fetching intraday data from Alpha Vantage: {e}")
        return None

# Rate limiting helper
_last_request_time = 0
_min_request_interval = 12  # Alpha Vantage free tier: 5 calls/min = 12 sec between calls

def rate_limit():
    """Ensure we don't exceed Alpha Vantage rate limits"""
    global _last_request_time

    current_time = time.time()
    time_since_last = current_time - _last_request_time

    if time_since_last < _min_request_interval:
        sleep_time = _min_request_interval - time_since_last
        print(f"⏱️ Rate limiting: waiting {sleep_time:.1f}s...")
        time.sleep(sleep_time)

    _last_request_time = time.time()
