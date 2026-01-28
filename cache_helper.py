"""
Simple file-based cache to reduce Yahoo Finance API calls
"""
import json
import os
from datetime import datetime, timedelta
import hashlib

CACHE_DIR = '/tmp/stock_cache'
CACHE_DURATION_MINUTES = 15  # Cache data for 15 minutes

def get_cache_path(key):
    """Get cache file path for a given key"""
    if not os.path.exists(CACHE_DIR):
        os.makedirs(CACHE_DIR)
    hash_key = hashlib.md5(key.encode()).hexdigest()
    return os.path.join(CACHE_DIR, f"{hash_key}.json")

def get_cached_data(key):
    """Get cached data if it exists and is not expired"""
    cache_path = get_cache_path(key)

    if not os.path.exists(cache_path):
        return None

    try:
        with open(cache_path, 'r') as f:
            cache_data = json.load(f)

        # Check if cache is expired
        cached_time = datetime.fromisoformat(cache_data['timestamp'])
        if datetime.now() - cached_time > timedelta(minutes=CACHE_DURATION_MINUTES):
            return None

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
