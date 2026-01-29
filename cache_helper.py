"""
Simple file-based cache to reduce Alpha Vantage API calls
Different cache durations for different data types to maximize free tier usage
"""
import json
import os
from datetime import datetime, timedelta
import hashlib

CACHE_DIR = '/tmp/stock_cache'

# Cache durations optimized for Alpha Vantage free tier (25 calls/day)
CACHE_DURATIONS = {
    'quick_quote': 30,      # 30 minutes - prices update frequently
    'analysis': 240,         # 4 hours - fundamentals change slowly
    'chart': 120,            # 2 hours - historical data is stable
    'default': 60            # 1 hour - fallback
}

def get_cache_path(key):
    """Get cache file path for a given key"""
    if not os.path.exists(CACHE_DIR):
        os.makedirs(CACHE_DIR)
    hash_key = hashlib.md5(key.encode()).hexdigest()
    return os.path.join(CACHE_DIR, f"{hash_key}.json")

def get_cached_data(key, data_type='default'):
    """Get cached data if it exists and is not expired"""
    cache_path = get_cache_path(key)

    if not os.path.exists(cache_path):
        return None

    try:
        with open(cache_path, 'r') as f:
            cache_data = json.load(f)

        # Get cache duration for this data type
        cache_duration = CACHE_DURATIONS.get(data_type, CACHE_DURATIONS['default'])

        # Check if cache is expired
        cached_time = datetime.fromisoformat(cache_data['timestamp'])
        if datetime.now() - cached_time > timedelta(minutes=cache_duration):
            return None

        print(f"✅ Cache hit for {key} ({data_type}, age: {(datetime.now() - cached_time).seconds // 60} min)")
        return cache_data['data']
    except Exception as e:
        print(f"Cache read error: {e}")
        return None

def set_cached_data(key, data):
    """Cache data with timestamp"""
    cache_path = get_cache_path(key)

    try:
        cache_data = {
            'timestamp': datetime.now().isoformat(),
            'data': data
        }
        with open(cache_path, 'w') as f:
            json.dump(cache_data, f)
    except Exception as e:
        print(f"Cache write error: {e}")
