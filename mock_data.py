"""
Mock stock data for testing without Yahoo Finance API
"""
from datetime import datetime, timedelta
import random

MOCK_STOCKS = {
    'AAPL': {
        'company_name': 'Apple Inc.',
        'current_price': 234.52,
        'sector': 'Technology',
        'industry': 'Consumer Electronics',
    },
    'MSFT': {
        'company_name': 'Microsoft Corporation',
        'current_price': 428.15,
        'sector': 'Technology',
        'industry': 'Software—Infrastructure',
    },
    'NVDA': {
        'company_name': 'NVIDIA Corporation',
        'current_price': 875.28,
        'sector': 'Technology',
        'industry': 'Semiconductors',
    },
    'TSLA': {
        'company_name': 'Tesla, Inc.',
        'current_price': 207.83,
        'sector': 'Consumer Cyclical',
        'industry': 'Auto Manufacturers',
    },
    'GOOGL': {
        'company_name': 'Alphabet Inc.',
        'current_price': 178.92,
        'sector': 'Communication Services',
        'industry': 'Internet Content & Information',
    },
    'AMZN': {
        'company_name': 'Amazon.com, Inc.',
        'current_price': 218.34,
        'sector': 'Consumer Cyclical',
        'industry': 'Internet Retail',
    },
    'META': {
        'company_name': 'Meta Platforms, Inc.',
        'current_price': 612.45,
        'sector': 'Communication Services',
        'industry': 'Internet Content & Information',
    },
}

def get_mock_quote(symbol):
    """Get mock quick quote data"""
    symbol = symbol.upper()

    if symbol not in MOCK_STOCKS:
        # Generate random data for unknown symbols
        base_price = random.uniform(50, 500)
        change_pct = random.uniform(-5, 5)
    else:
        stock = MOCK_STOCKS[symbol]
        base_price = stock['current_price']
        change_pct = random.uniform(-2, 3)  # Random daily change

    change = base_price * (change_pct / 100)

    return {
        'symbol': symbol,
        'company_name': MOCK_STOCKS.get(symbol, {}).get('company_name', symbol),
        'current_price': round(base_price, 2),
        'change': round(change, 2),
        'change_pct': round(change_pct, 2),
        'timestamp': datetime.now().isoformat(),
        'cached': False,
        'is_mock': True  # Flag to indicate this is demo data
    }

def get_mock_chart_data(symbol, period='90d'):
    """Get mock chart data"""
    import pandas as pd

    symbol = symbol.upper()
    stock = MOCK_STOCKS.get(symbol, {})
    base_price = stock.get('current_price', random.uniform(50, 500))

    # Generate 90 days of historical data
    days = int(period.replace('d', '')) if 'd' in period else 90
    chart_data = []

    current_date = datetime.now()
    price = base_price * 0.85  # Start 15% lower

    for i in range(days):
        date = current_date - timedelta(days=days - i)

        # Skip weekends
        if date.weekday() >= 5:
            continue

        # Generate realistic OHLC data
        daily_change = random.uniform(-0.03, 0.04)
        open_price = price
        close_price = price * (1 + daily_change)
        high_price = max(open_price, close_price) * random.uniform(1.00, 1.02)
        low_price = min(open_price, close_price) * random.uniform(0.98, 1.00)
        volume = random.randint(10000000, 100000000)

        chart_data.append({
            'date': date.isoformat(),
            'open': round(open_price, 2),
            'high': round(high_price, 2),
            'low': round(low_price, 2),
            'close': round(close_price, 2),
            'volume': volume
        })

        price = close_price

    return {
        'symbol': symbol,
        'period': period,
        'data': chart_data,
        'is_mock': True
    }

def get_mock_analysis(symbol):
    """Get mock comprehensive analysis data"""
    symbol = symbol.upper()
    stock = MOCK_STOCKS.get(symbol, {})

    base_price = stock.get('current_price', random.uniform(50, 500))
    change_pct = random.uniform(-2, 3)

    return {
        'symbol': symbol,
        'timestamp': datetime.now().isoformat(),

        # Basic Info
        'info': {
            'company_name': stock.get('company_name', symbol),
            'sector': stock.get('sector', 'Technology'),
            'industry': stock.get('industry', 'Software'),
            'current_price': base_price,
            'change': base_price * (change_pct / 100),
            'change_pct': change_pct,
            'signal': 'BUY' if change_pct > 1 else 'HOLD' if change_pct > -1 else 'SELL',
            'signal_color': 'green' if change_pct > 1 else 'yellow' if change_pct > -1 else 'red'
        },

        # Fundamentals
        'fundamentals': {
            'market_cap': base_price * 1000000000,
            'pe_ratio': random.uniform(15, 35),
            'forward_pe': random.uniform(14, 30),
            'eps_current': random.uniform(2, 15),
            'eps_forward': random.uniform(2.5, 16),
            'profit_margin': random.uniform(15, 30),
            'roe': random.uniform(20, 40),
            'roa': random.uniform(10, 25),
            'revenue_growth': random.uniform(5, 25),
            'earnings_growth': random.uniform(8, 30),
            'debt_to_equity': random.uniform(0.2, 1.5),
            'current_ratio': random.uniform(1.2, 2.5),
            'dividend_yield': random.uniform(0, 2.5),
            'fundamental_score': random.uniform(65, 90),
            'recommendation': 'BUY' if change_pct > 1 else 'HOLD',
            'target_mean_price': base_price * 1.15,
            'target_high_price': base_price * 1.25,
            'target_low_price': base_price * 0.95,
            'number_of_analysts': random.randint(25, 45)
        },

        # Technicals
        'technicals': {
            'rsi': random.uniform(30, 70),
            'macd': random.uniform(-2, 2),
            'macd_signal': random.uniform(-2, 2),
            'moving_averages': {
                'SMA_20': base_price * 0.98,
                'SMA_50': base_price * 0.95,
                'SMA_200': base_price * 0.90,
                'EMA_12': base_price * 0.99,
                'EMA_26': base_price * 0.97
            },
            'bollinger': {
                'upper': base_price * 1.05,
                'middle': base_price,
                'lower': base_price * 0.95,
                'position': random.uniform(40, 60)  # Percentage position in bollinger bands
            },
            'signal_reasons': ['Positive momentum', 'Strong fundamentals', 'Above moving averages']
        },

        # Earnings
        'earnings': {
            'next_earnings_date': (datetime.now() + timedelta(days=30)).isoformat(),
            'days_to_earnings': 30,
            'last_earnings_date': (datetime.now() - timedelta(days=60)).isoformat(),
            'expected_eps': random.uniform(2, 5),
            'actual_eps': random.uniform(2.5, 5.5),
            'surprise_percent': random.uniform(-10, 15),
            'sentiment': 'Positive',
            'recommendation': 'Hold through earnings',
            'short_term_impact': 'Potentially positive',
            'earnings_history': [
                {'date': (datetime.now() - timedelta(days=90)).isoformat(), 'eps': random.uniform(2, 5)}
                for _ in range(4)
            ]
        },

        # Fibonacci
        'fibonacci': {
            'swing_high': base_price * 1.10,
            'swing_low': base_price * 0.90,
            'level_236': base_price * 0.95,
            'level_382': base_price * 0.93,
            'level_500': base_price * 1.00,
            'level_618': base_price * 1.03,
            'level_786': base_price * 1.07,
            'position': 'Between 50% and 61.8%',
            'next_support': base_price * 0.95,
            'next_resistance': base_price * 1.05
        },

        # Candlestick Patterns
        'candlestick_patterns': [
            {
                'pattern': 'Bullish Engulfing',
                'signal': 'Bullish',
                'strength': 'Strong',
                'description': 'Strong bullish reversal pattern indicating potential upward momentum'
            }
        ],

        # Day Trading
        'day_trading': {
            'recommendation': 'Good for day trading',
            'strategy': 'Momentum trading',
            'volatility_atr': random.uniform(2, 8),
            'volatility_percent': random.uniform(1.5, 4.5),
            'volume_signal': 'High',
            'volume_ratio': random.uniform(1.2, 2.5),
            'momentum_5d': random.uniform(-5, 8),
            'momentum_10d': random.uniform(-8, 12),
            'support_levels': [base_price * 0.98, base_price * 0.95, base_price * 0.92],
            'resistance_levels': [base_price * 1.02, base_price * 1.05, base_price * 1.08],
            'pivot_point': base_price,
            'target_profit': base_price * 1.03,
            'stop_loss': base_price * 0.97,
            'risk_reward_ratio': 2.5,
            'signals': ['Strong volume', 'Upward momentum', 'Above VWAP']
        },

        # Short-term Forecast (5 weekdays)
        'short_term_forecast': [
            {
                'date': (datetime.now() + timedelta(days=i)).isoformat(),
                'day_name': (datetime.now() + timedelta(days=i)).strftime('%A'),
                'predicted_price': base_price * (1 + (pct := random.uniform(-0.02, 0.03))),
                'change_from_current': pct * 100,
                'confidence': random.uniform(70, 90)
            }
            for i in range(1, 6)
        ],

        # Long-term Projection
        'long_term_projection': {
            'projections': {
                '12_months': {
                    'bullish': base_price * 1.30,
                    'base': base_price * 1.15,
                    'bearish': base_price * 0.90
                }
            },
            'analyst_targets': {
                'upside_potential': ((base_price * 1.15) - base_price) / base_price * 100
            },
            'timeframe': '12 months',
            'probability_bull': 30,
            'probability_base': 50,
            'probability_bear': 20
        },

        # Sector Forecast
        'sector_forecast': {
            'sector': stock.get('sector', 'Technology'),
            'sector_outlook': 'Positive outlook with strong growth drivers',
            'growth_potential': random.uniform(10, 25),  # Percentage
            'risk_level': 'Medium',
            'expected_growth_rate': random.uniform(10, 25),
            'key_drivers': ['AI adoption', 'Digital transformation', 'Cloud computing']
        },

        # Investor Strategies
        'investor_strategies': [
            {
                'strategy': 'Growth Investing',
                'suitability': 'High',
                'timeframe': 'Long-term (3-5 years)',
                'risk_level': 'Medium'
            },
            {
                'strategy': 'Value Investing',
                'suitability': 'Medium',
                'timeframe': 'Medium-term (1-3 years)',
                'risk_level': 'Low'
            }
        ],

        'is_mock': True  # Flag to indicate this is demo data
    }
