#!/usr/bin/env python3
"""
Stock Analysis API Server
Flask API to expose stock analysis functionality for mobile app
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
from StockAnalysis import StockAnalyzer
from cache_helper import get_cached_data, set_cached_data
from mock_data import get_mock_quote, get_mock_analysis, get_mock_chart_data
import alpha_vantage_data as av
from alpha_analysis import analyze_stock_alpha, get_chart_data_alpha
import json
from datetime import datetime
import traceback

# Data source selection
# 'mock' = Demo data for testing
# 'alpha_vantage' = Real data from Alpha Vantage (25 calls/day free)
# 'yahoo' = Yahoo Finance (currently blocked on corporate networks)
DATA_SOURCE = 'alpha_vantage'

app = Flask(__name__)
CORS(app)  # Enable CORS for mobile app requests

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'version': '1.0.0'
    })

@app.route('/api/analyze/<symbol>', methods=['GET'])
def analyze_stock(symbol):
    """Analyze a stock and return comprehensive data"""
    try:
        # Use mock data
        if DATA_SOURCE == 'mock':
            print(f"🎭 Returning mock analysis for {symbol}")
            return jsonify(get_mock_analysis(symbol))

        # Use Alpha Vantage (no fallback - show upgrade message if rate limited)
        if DATA_SOURCE == 'alpha_vantage':
            print(f"📊 Analyzing {symbol} with Alpha Vantage...")
            analysis = analyze_stock_alpha(symbol)

            if analysis:
                return jsonify(analysis)
            else:
                # Return error message asking user to upgrade
                print(f"⚠️ Alpha Vantage API limit exceeded for {symbol}")
                return jsonify({
                    'error': 'API_LIMIT_EXCEEDED',
                    'message': 'Free tier API limit exceeded. Please upgrade to premium to access real-time stock data.',
                    'upgrade_info': {
                        'title': 'Upgrade to Premium',
                        'description': 'Get unlimited access to real-time stock analysis with our premium plan.',
                        'benefits': [
                            'Unlimited API calls',
                            'Real-time data updates',
                            'Advanced technical indicators',
                            'Priority support'
                        ]
                    }
                }), 429  # 429 Too Many Requests

        print(f"📊 Analyzing {symbol}...")

        # Initialize analyzer
        analyzer = StockAnalyzer(symbol.upper())

        # Run analysis
        success = analyzer.run_complete_analysis()

        if not success:
            return jsonify({
                'error': 'Analysis failed',
                'message': f'Could not fetch data for {symbol}. Please check the symbol and try again.'
            }), 400

        # Prepare response data
        response = {
            'symbol': analyzer.symbol,
            'timestamp': datetime.now().isoformat(),

            # Basic Info
            'info': {
                'company_name': analyzer.fundamental_analysis.get('company_name', ''),
                'sector': analyzer.fundamental_analysis.get('sector', ''),
                'industry': analyzer.fundamental_analysis.get('industry', ''),
                'current_price': analyzer.technical_analysis['current_price'],
                'change': analyzer.technical_analysis['change'],
                'change_pct': analyzer.technical_analysis['change_pct'],
                'signal': analyzer.technical_analysis['signal'],
                'signal_color': analyzer.technical_analysis['signal_color']
            },

            # Fundamental Analysis
            'fundamentals': {
                'market_cap': analyzer.fundamental_analysis.get('market_cap', 0),
                'pe_ratio': analyzer.fundamental_analysis.get('pe_ratio', 0),
                'forward_pe': analyzer.fundamental_analysis.get('forward_pe', 0),
                'eps_current': analyzer.fundamental_analysis.get('eps_current', 0),
                'eps_forward': analyzer.fundamental_analysis.get('eps_forward', 0),
                'profit_margin': analyzer.fundamental_analysis.get('profit_margin', 0),
                'roe': analyzer.fundamental_analysis.get('roe', 0),
                'roa': analyzer.fundamental_analysis.get('roa', 0),
                'revenue_growth': analyzer.fundamental_analysis.get('revenue_growth', 0),
                'earnings_growth': analyzer.fundamental_analysis.get('earnings_growth', 0),
                'debt_to_equity': analyzer.fundamental_analysis.get('debt_to_equity', 0),
                'current_ratio': analyzer.fundamental_analysis.get('current_ratio', 0),
                'dividend_yield': analyzer.fundamental_analysis.get('dividend_yield', 0),
                'fundamental_score': analyzer.fundamental_analysis.get('fundamental_score', 0),
                'recommendation': analyzer.fundamental_analysis.get('recommendation', ''),
                'target_mean_price': analyzer.fundamental_analysis.get('target_mean_price', 0),
                'target_high_price': analyzer.fundamental_analysis.get('target_high_price', 0),
                'target_low_price': analyzer.fundamental_analysis.get('target_low_price', 0),
                'number_of_analysts': analyzer.fundamental_analysis.get('number_of_analysts', 0)
            },

            # Technical Analysis
            'technicals': {
                'rsi': analyzer.technical_analysis['rsi'],
                'macd': analyzer.technical_analysis['macd'],
                'macd_signal': analyzer.technical_analysis['macd_signal'],
                'moving_averages': analyzer.technical_analysis['moving_averages'],
                'bollinger': analyzer.technical_analysis['bollinger'],
                'signal_reasons': analyzer.technical_analysis['signal_reasons']
            },

            # Earnings Analysis
            'earnings': {
                'next_earnings_date': analyzer.earnings_analysis.get('next_earnings_date', None),
                'days_to_earnings': analyzer.earnings_analysis.get('days_to_earnings', None),
                'last_earnings_date': analyzer.earnings_analysis.get('last_earnings_date', None),
                'expected_eps': analyzer.earnings_analysis.get('expected_eps', 0),
                'actual_eps': analyzer.earnings_analysis.get('actual_eps', 0),
                'surprise_percent': analyzer.earnings_analysis.get('surprise_percent', 0),
                'sentiment': analyzer.earnings_analysis.get('sentiment', ''),
                'recommendation': analyzer.earnings_analysis.get('recommendation', ''),
                'short_term_impact': analyzer.earnings_analysis.get('short_term_impact', ''),
                'earnings_history': analyzer.earnings_analysis.get('earnings_history', [])[:4]
            },

            # Fibonacci Levels
            'fibonacci': {
                'swing_high': analyzer.fibonacci_levels.get('swing_high', 0),
                'swing_low': analyzer.fibonacci_levels.get('swing_low', 0),
                'level_236': analyzer.fibonacci_levels.get('level_236', 0),
                'level_382': analyzer.fibonacci_levels.get('level_382', 0),
                'level_500': analyzer.fibonacci_levels.get('level_500', 0),
                'level_618': analyzer.fibonacci_levels.get('level_618', 0),
                'level_786': analyzer.fibonacci_levels.get('level_786', 0),
                'position': analyzer.fibonacci_levels.get('position', ''),
                'next_support': analyzer.fibonacci_levels.get('next_support', 0),
                'next_resistance': analyzer.fibonacci_levels.get('next_resistance', 0)
            },

            # Candlestick Patterns
            'candlestick_patterns': analyzer.candlestick_patterns,

            # Day Trading Analysis
            'day_trading': {
                'recommendation': analyzer.day_trading_analysis.get('recommendation', ''),
                'strategy': analyzer.day_trading_analysis.get('strategy', ''),
                'volatility_atr': analyzer.day_trading_analysis.get('volatility_atr', 0),
                'volatility_percent': analyzer.day_trading_analysis.get('volatility_percent', 0),
                'volume_signal': analyzer.day_trading_analysis.get('volume_signal', ''),
                'volume_ratio': analyzer.day_trading_analysis.get('volume_ratio', 0),
                'momentum_5d': analyzer.day_trading_analysis.get('momentum_5d', 0),
                'momentum_10d': analyzer.day_trading_analysis.get('momentum_10d', 0),
                'support_levels': analyzer.day_trading_analysis.get('support_levels', []),
                'resistance_levels': analyzer.day_trading_analysis.get('resistance_levels', []),
                'pivot_point': analyzer.day_trading_analysis.get('pivot_point', 0),
                'target_profit': analyzer.day_trading_analysis.get('target_profit', 0),
                'stop_loss': analyzer.day_trading_analysis.get('stop_loss', 0),
                'risk_reward_ratio': analyzer.day_trading_analysis.get('risk_reward_ratio', 0),
                'signals': analyzer.day_trading_analysis.get('signals', [])
            },

            # Short-term Forecast (5 weekdays)
            'short_term_forecast': analyzer.short_term_forecast,

            # Long-term Projection
            'long_term_projection': analyzer.long_term_projection,

            # Sector Analysis
            'sector_forecast': analyzer.sector_forecast,

            # Investor Strategies
            'investor_strategies': analyzer.investor_strategies[:5]  # Top 5
        }

        # Convert datetime objects to strings
        def convert_dates(obj):
            if isinstance(obj, dict):
                return {k: convert_dates(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_dates(item) for item in obj]
            elif hasattr(obj, 'isoformat'):
                return obj.isoformat()
            else:
                return obj

        response = convert_dates(response)

        print(f"✅ Analysis complete for {symbol}")
        return jsonify(response)

    except Exception as e:
        print(f"❌ Error analyzing {symbol}: {str(e)}")
        traceback.print_exc()
        return jsonify({
            'error': 'Internal server error',
            'message': str(e)
        }), 500

@app.route('/api/chart/<symbol>', methods=['GET'])
def get_chart_data(symbol):
    """Get historical price data for charting"""
    try:
        period = request.args.get('period', '90d')  # Default 90 days

        # Use mock data
        if DATA_SOURCE == 'mock':
            print(f"🎭 Returning mock chart data for {symbol}")
            return jsonify(get_mock_chart_data(symbol, period))

        # Use Alpha Vantage (no fallback - show upgrade message if rate limited)
        if DATA_SOURCE == 'alpha_vantage':
            print(f"📈 Fetching chart data from Alpha Vantage for {symbol}")
            chart_data = get_chart_data_alpha(symbol, period)

            if chart_data:
                return jsonify(chart_data)
            else:
                # Return error message asking user to upgrade
                print(f"⚠️ Alpha Vantage API limit exceeded for chart data {symbol}")
                return jsonify({
                    'error': 'API_LIMIT_EXCEEDED',
                    'message': 'Free tier API limit exceeded. Please upgrade to premium to access real-time chart data.',
                    'upgrade_info': {
                        'title': 'Upgrade to Premium',
                        'description': 'Get unlimited access to real-time stock charts with our premium plan.'
                    }
                }), 429  # 429 Too Many Requests

        analyzer = StockAnalyzer(symbol.upper())

        if not analyzer.fetch_stock_data(period=period):
            return jsonify({
                'error': 'Failed to fetch data',
                'message': f'Could not fetch chart data for {symbol}'
            }), 400

        # Prepare chart data
        data = analyzer.data.tail(int(period.replace('d', '')))

        chart_data = []
        for index, row in data.iterrows():
            chart_data.append({
                'date': index.isoformat(),
                'open': float(row['Open']),
                'high': float(row['High']),
                'low': float(row['Low']),
                'close': float(row['Close']),
                'volume': int(row['Volume'])
            })

        return jsonify({
            'symbol': symbol.upper(),
            'period': period,
            'data': chart_data
        })

    except Exception as e:
        print(f"❌ Error fetching chart data for {symbol}: {str(e)}")
        return jsonify({
            'error': 'Internal server error',
            'message': str(e)
        }), 500

@app.route('/api/quick-quote/<symbol>', methods=['GET'])
def quick_quote(symbol):
    """Get quick quote without full analysis (with 15-min caching)"""
    try:
        # Use mock data
        if DATA_SOURCE == 'mock':
            print(f"🎭 Returning mock quote for {symbol}")
            return jsonify(get_mock_quote(symbol))

        # Use Alpha Vantage (no fallback - show upgrade message if rate limited)
        if DATA_SOURCE == 'alpha_vantage':
            print(f"📊 Fetching real quote from Alpha Vantage for {symbol}")
            av.rate_limit()  # Respect rate limits
            quote = av.get_quote(symbol)

            if quote:
                return jsonify(quote)
            else:
                # Return error message asking user to upgrade
                print(f"⚠️ Alpha Vantage API limit exceeded for quick quote {symbol}")
                return jsonify({
                    'error': 'API_LIMIT_EXCEEDED',
                    'message': 'Free tier API limit exceeded. Please upgrade to premium to continue accessing stock quotes.',
                    'upgrade_info': {
                        'title': 'Upgrade to Premium',
                        'description': 'Get unlimited access to real-time stock quotes with our premium plan.'
                    }
                }), 429  # 429 Too Many Requests

        cache_key = f"quick_quote_{symbol.upper()}"

        # Try to get cached data first
        cached = get_cached_data(cache_key)
        if cached:
            print(f"📦 Returning cached quote for {symbol}")
            cached['cached'] = True
            return jsonify(cached)

        # Fetch fresh data - use lightweight approach to avoid rate limiting
        print(f"🔄 Fetching fresh quote for {symbol}")
        import yfinance as yf
        from StockAnalysis import yf_session

        ticker = yf.Ticker(symbol.upper(), session=yf_session)

        # Get only price history (avoids the .info API call that causes 429)
        data = ticker.history(period='5d')

        if len(data) == 0:
            return jsonify({
                'error': 'Failed to fetch data',
                'message': f'Could not fetch data for {symbol}'
            }), 400

        current_price = data['Close'].iloc[-1]
        prev_close = data['Close'].iloc[-2] if len(data) > 1 else current_price
        change = current_price - prev_close
        change_pct = (change / prev_close) * 100 if prev_close != 0 else 0

        result = {
            'symbol': symbol.upper(),
            'company_name': symbol.upper(),  # Skip company name to avoid API calls
            'current_price': float(current_price),
            'change': float(change),
            'change_pct': float(change_pct),
            'timestamp': datetime.now().isoformat(),
            'cached': False
        }

        # Cache the result
        set_cached_data(cache_key, result)

        return jsonify(result)

    except Exception as e:
        print(f"❌ Error getting quote for {symbol}: {str(e)}")
        return jsonify({
            'error': 'Internal server error',
            'message': str(e)
        }), 500

@app.route('/api/search/<query>', methods=['GET'])
def search_stocks(query):
    """Search for stock symbols"""
    # Simple implementation - in production, use a proper stock search API
    common_stocks = {
        'AAPL': 'Apple Inc.',
        'MSFT': 'Microsoft Corporation',
        'GOOGL': 'Alphabet Inc.',
        'AMZN': 'Amazon.com Inc.',
        'NVDA': 'NVIDIA Corporation',
        'TSLA': 'Tesla, Inc.',
        'META': 'Meta Platforms Inc.',
        'AMD': 'Advanced Micro Devices',
        'NFLX': 'Netflix Inc.',
        'DIS': 'The Walt Disney Company',
        'PLTR': 'Palantir Technologies',
        'COIN': 'Coinbase Global',
        'PYPL': 'PayPal Holdings',
        'SQ': 'Block Inc.',
        'SHOP': 'Shopify Inc.'
    }

    query_upper = query.upper()
    results = [
        {'symbol': symbol, 'name': name}
        for symbol, name in common_stocks.items()
        if query_upper in symbol or query_upper in name.upper()
    ]

    return jsonify({'results': results[:10]})

if __name__ == '__main__':
    import os

    # Get port from environment variable (for cloud deployment) or use 5000 for local
    port = int(os.environ.get('PORT', 5000))

    print("🚀 Starting Stock Analysis API Server...")
    print("📱 Mobile app can connect to this server")
    print(f"🌐 Server running on http://0.0.0.0:{port}")
    print("\n📝 API Endpoints:")
    print("  GET /health                    - Health check")
    print("  GET /api/analyze/<symbol>      - Full stock analysis")
    print("  GET /api/chart/<symbol>        - Chart data")
    print("  GET /api/quick-quote/<symbol>  - Quick quote")
    print("  GET /api/search/<query>        - Search stocks")
    print(f"\n💡 Test with: curl http://localhost:{port}/api/quick-quote/AAPL")
    print()

    # Run server
    # Use 0.0.0.0 to allow connections from anywhere
    # debug=False for production deployment
    debug_mode = os.environ.get('FLASK_ENV') != 'production'
    app.run(host='0.0.0.0', port=port, debug=debug_mode)
