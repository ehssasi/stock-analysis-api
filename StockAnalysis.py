#!/usr/bin/env python3
"""
Comprehensive Stock Analysis - Fundamental & Technical Analysis with Investor Strategies
Analyzes a single stock with detailed fundamental analysis, technical indicators,
top investor strategies, and provides both short-term (5 weekdays) and long-term forecasts.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import warnings
import os
import time
import sys
import json
warnings.filterwarnings('ignore')

# For interactive charts
try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    print("Warning: plotly not available. Install with: pip install plotly")

# Import yfinance
import yfinance as yf

# For PDF generation
try:
    import pdfkit
    PDFKIT_AVAILABLE = True
except ImportError:
    PDFKIT_AVAILABLE = False
    print("Warning: pdfkit not available. Install with: pip install pdfkit")
    print("Also install wkhtmltopdf: https://wkhtmltopdf.org/downloads.html")


class StockAnalyzer:
    """Comprehensive stock analysis with fundamental, technical analysis and investor strategies"""

    def __init__(self, symbol):
        self.symbol = symbol.upper()
        self.stock = None
        self.data = None
        self.info = {}
        self.fundamental_analysis = {}
        self.technical_analysis = {}
        self.investor_strategies = []
        self.short_term_forecast = []
        self.long_term_projection = {}
        self.earnings_analysis = {}
        self.fibonacci_levels = {}
        self.candlestick_patterns = []
        self.day_trading_analysis = {}
        self.sector_forecast = {}
        self.yearly_forecast = {}

    def fetch_stock_data(self, period='1y'):
        """Fetch historical stock data and company info"""
        try:
            print(f"\n📊 Fetching data for {self.symbol}...")
            self.stock = yf.Ticker(self.symbol)

            print(f"📥 Downloading history for period={period}...")
            self.data = self.stock.history(period=period)

            print(f"📋 Fetching company info...")
            self.info = self.stock.info

            print(f"✅ Data length: {len(self.data)}")
            print(f"✅ Info keys: {len(self.info.keys()) if self.info else 0}")

            if len(self.data) == 0:
                print(f"⚠️ No historical data returned for {self.symbol}")
                print(f"⚠️ This might be due to network issues or Yahoo Finance blocking")
                return False

            if len(self.data) < 60 and period == '1y':
                print(f"⚠️ Insufficient data for {self.symbol}: only {len(self.data)} days")
                return False

            print(f"✅ Successfully fetched {len(self.data)} days of data")
            return True
        except Exception as e:
            print(f"❌ Error fetching {self.symbol}: {e}")
            import traceback
            traceback.print_exc()
            return False

    def analyze_fundamentals(self):
        """Perform comprehensive fundamental analysis"""
        print(f"\n💼 Analyzing fundamentals for {self.symbol}...")

        fundamentals = {
            'company_name': self.info.get('longName', self.symbol),
            'sector': self.info.get('sector', 'N/A'),
            'industry': self.info.get('industry', 'N/A'),
            'market_cap': self.info.get('marketCap', 0),
            'enterprise_value': self.info.get('enterpriseValue', 0),

            # Valuation Metrics
            'pe_ratio': self.info.get('trailingPE', 0),
            'forward_pe': self.info.get('forwardPE', 0),
            'peg_ratio': self.info.get('pegRatio', 0),
            'price_to_book': self.info.get('priceToBook', 0),
            'price_to_sales': self.info.get('priceToSalesTrailing12Months', 0),

            # Profitability Metrics
            'profit_margin': self.info.get('profitMargins', 0) * 100 if self.info.get('profitMargins') else 0,
            'operating_margin': self.info.get('operatingMargins', 0) * 100 if self.info.get('operatingMargins') else 0,
            'roe': self.info.get('returnOnEquity', 0) * 100 if self.info.get('returnOnEquity') else 0,
            'roa': self.info.get('returnOnAssets', 0) * 100 if self.info.get('returnOnAssets') else 0,

            # Growth Metrics
            'revenue_growth': self.info.get('revenueGrowth', 0) * 100 if self.info.get('revenueGrowth') else 0,
            'earnings_growth': self.info.get('earningsGrowth', 0) * 100 if self.info.get('earningsGrowth') else 0,
            'eps_current': self.info.get('trailingEps', 0),
            'eps_forward': self.info.get('forwardEps', 0),

            # Financial Health
            'current_ratio': self.info.get('currentRatio', 0),
            'debt_to_equity': self.info.get('debtToEquity', 0),
            'free_cash_flow': self.info.get('freeCashflow', 0),
            'operating_cash_flow': self.info.get('operatingCashflow', 0),

            # Dividend Info
            'dividend_yield': self.info.get('dividendYield', 0) * 100 if self.info.get('dividendYield') else 0,
            'payout_ratio': self.info.get('payoutRatio', 0) * 100 if self.info.get('payoutRatio') else 0,

            # Analyst Recommendations
            'target_mean_price': self.info.get('targetMeanPrice', 0),
            'target_high_price': self.info.get('targetHighPrice', 0),
            'target_low_price': self.info.get('targetLowPrice', 0),
            'recommendation': self.info.get('recommendationKey', 'N/A'),
            'number_of_analysts': self.info.get('numberOfAnalystOpinions', 0),
        }

        # Calculate fundamental score
        fundamentals['fundamental_score'] = self._calculate_fundamental_score(fundamentals)

        self.fundamental_analysis = fundamentals
        print(f"✅ Fundamental analysis complete")
        return fundamentals

    def _calculate_fundamental_score(self, fundamentals):
        """Calculate overall fundamental score (0-100)"""
        score = 50  # Start with neutral score

        # P/E Ratio (lower is better, typically)
        pe = fundamentals['pe_ratio']
        if 0 < pe < 15:
            score += 10
        elif 15 <= pe < 25:
            score += 5
        elif pe >= 40:
            score -= 10

        # Profit Margin (higher is better)
        if fundamentals['profit_margin'] > 20:
            score += 10
        elif fundamentals['profit_margin'] > 10:
            score += 5
        elif fundamentals['profit_margin'] < 0:
            score -= 10

        # ROE (higher is better)
        if fundamentals['roe'] > 20:
            score += 10
        elif fundamentals['roe'] > 10:
            score += 5
        elif fundamentals['roe'] < 0:
            score -= 10

        # Revenue Growth (higher is better)
        if fundamentals['revenue_growth'] > 20:
            score += 10
        elif fundamentals['revenue_growth'] > 10:
            score += 5
        elif fundamentals['revenue_growth'] < 0:
            score -= 10

        # Debt to Equity (lower is better)
        if fundamentals['debt_to_equity'] < 50:
            score += 5
        elif fundamentals['debt_to_equity'] > 200:
            score -= 10

        return max(0, min(100, score))  # Clamp between 0-100

    def calculate_rsi(self, data, period=14):
        """Calculate Relative Strength Index"""
        delta = data['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi.iloc[-1]

    def calculate_macd(self, data):
        """Calculate MACD (Moving Average Convergence Divergence)"""
        exp1 = data['Close'].ewm(span=12, adjust=False).mean()
        exp2 = data['Close'].ewm(span=26, adjust=False).mean()
        macd = exp1 - exp2
        signal = macd.ewm(span=9, adjust=False).mean()
        return macd.iloc[-1], signal.iloc[-1]

    def calculate_bollinger_bands(self, data, period=20):
        """Calculate Bollinger Bands"""
        sma = data['Close'].rolling(window=period).mean()
        std = data['Close'].rolling(window=period).std()
        upper_band = sma + (std * 2)
        lower_band = sma - (std * 2)
        current_price = data['Close'].iloc[-1]
        return {
            'upper': upper_band.iloc[-1],
            'middle': sma.iloc[-1],
            'lower': lower_band.iloc[-1],
            'position': (current_price - lower_band.iloc[-1]) / (upper_band.iloc[-1] - lower_band.iloc[-1]) * 100
        }

    def calculate_moving_averages(self, data):
        """Calculate various moving averages"""
        return {
            'SMA_20': data['Close'].rolling(window=20).mean().iloc[-1],
            'SMA_50': data['Close'].rolling(window=50).mean().iloc[-1],
            'EMA_12': data['Close'].ewm(span=12, adjust=False).mean().iloc[-1],
            'EMA_26': data['Close'].ewm(span=26, adjust=False).mean().iloc[-1]
        }

    def get_investor_strategies(self):
        """Analyze top institutional investors and their strategies"""
        print(f"\n👥 Analyzing top investors for {self.symbol}...")

        try:
            # Get institutional holders
            institutional_holders = self.stock.institutional_holders
            major_holders = self.stock.major_holders

            strategies = []

            if institutional_holders is not None and not institutional_holders.empty:
                # Top 5 institutional investors
                top_investors = institutional_holders.head(5)

                for idx, row in top_investors.iterrows():
                    investor = {
                        'holder': row.get('Holder', 'N/A'),
                        'shares': row.get('Shares', 0),
                        'date_reported': row.get('Date Reported', 'N/A'),
                        'value': row.get('Value', 0),
                        'pct_out': row.get('% Out', 0) if '% Out' in row else 0
                    }

                    # Analyze strategy based on holdings
                    if investor['pct_out'] > 5:
                        investor['strategy'] = 'Major Position - Long-term holder'
                        investor['sentiment'] = 'Bullish'
                    elif investor['pct_out'] > 2:
                        investor['strategy'] = 'Significant Position - Growth focused'
                        investor['sentiment'] = 'Bullish'
                    else:
                        investor['strategy'] = 'Moderate Position - Diversified portfolio'
                        investor['sentiment'] = 'Neutral'

                    strategies.append(investor)

            # Add major holders summary
            if major_holders is not None and not major_holders.empty:
                major_info = {
                    'type': 'Major Holders Summary',
                    'holder': 'Institutional & Insider Ownership',
                    'strategy': major_holders.to_dict() if not major_holders.empty else {}
                }
                strategies.append(major_info)

            self.investor_strategies = strategies
            print(f"✅ Found {len(strategies)} investor entries")
            return strategies

        except Exception as e:
            print(f"⚠️ Could not fetch investor data: {e}")
            return []

    def predict_next_5_weekdays(self):
        """Predict prices for next 5 weekdays (excluding weekends)"""
        print(f"\n📅 Predicting next 5 weekdays for {self.symbol}...")

        try:
            # Prepare data with multiple features
            data_subset = self.data.tail(90)  # Use last 90 days

            # Create feature set
            data_subset['Returns'] = data_subset['Close'].pct_change()
            data_subset['Volatility'] = data_subset['Returns'].rolling(window=10).std()
            data_subset['MA_10'] = data_subset['Close'].rolling(window=10).mean()
            data_subset['MA_30'] = data_subset['Close'].rolling(window=30).mean()
            data_subset['Volume_MA'] = data_subset['Volume'].rolling(window=10).mean()

            # Drop NaN values
            data_subset = data_subset.dropna()

            # Prepare features
            X = np.arange(len(data_subset)).reshape(-1, 1)
            y = data_subset['Close'].values

            # Use both Linear Regression and Random Forest for ensemble prediction
            lr_model = LinearRegression()
            rf_model = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=10)

            lr_model.fit(X, y)
            rf_model.fit(X, y)

            # Generate next 5 weekdays
            last_date = self.data.index[-1]
            weekday_predictions = []
            current_date = last_date

            days_count = 0
            while days_count < 5:
                current_date += timedelta(days=1)

                # Skip weekends (5=Saturday, 6=Sunday)
                if current_date.weekday() < 5:  # Monday=0, Friday=4
                    day_index = len(data_subset) + days_count
                    X_pred = np.array([[day_index]])

                    # Ensemble prediction (average of both models)
                    lr_pred = lr_model.predict(X_pred)[0]
                    rf_pred = rf_model.predict(X_pred)[0]
                    ensemble_pred = (lr_pred * 0.4 + rf_pred * 0.6)

                    # Add some volatility adjustment
                    volatility = data_subset['Close'].pct_change().std()
                    confidence_lower = ensemble_pred * (1 - volatility * 2)
                    confidence_upper = ensemble_pred * (1 + volatility * 2)

                    weekday_predictions.append({
                        'date': current_date,
                        'day_name': current_date.strftime('%A'),
                        'predicted_price': ensemble_pred,
                        'confidence_lower': confidence_lower,
                        'confidence_upper': confidence_upper,
                        'change_from_current': ((ensemble_pred - y[-1]) / y[-1]) * 100
                    })

                    days_count += 1

            self.short_term_forecast = weekday_predictions
            print(f"✅ Generated 5 weekday predictions")
            return weekday_predictions

        except Exception as e:
            print(f"⚠️ Error in prediction: {e}")
            return []

    def predict_long_term(self):
        """Generate long-term projection (3, 6, 12 months)"""
        print(f"\n📈 Generating long-term projection for {self.symbol}...")

        try:
            current_price = self.data['Close'].iloc[-1]

            # Calculate historical growth rates
            data_1m = self.data.tail(21)  # ~1 month
            data_3m = self.data.tail(63)  # ~3 months
            data_6m = self.data.tail(126)  # ~6 months

            growth_1m = ((data_1m['Close'].iloc[-1] - data_1m['Close'].iloc[0]) / data_1m['Close'].iloc[0]) * 100
            growth_3m = ((data_3m['Close'].iloc[-1] - data_3m['Close'].iloc[0]) / data_3m['Close'].iloc[0]) * 100
            growth_6m = ((data_6m['Close'].iloc[-1] - data_6m['Close'].iloc[0]) / data_6m['Close'].iloc[0]) * 100

            # Calculate volatility
            volatility = self.data['Close'].pct_change().std() * np.sqrt(252)  # Annualized

            # Project based on moving averages and trends
            ma_50 = self.data['Close'].rolling(window=50).mean().iloc[-1]
            ma_200 = self.data['Close'].rolling(window=200).mean().iloc[-1]

            # Use analyst targets if available
            target_mean = self.fundamental_analysis.get('target_mean_price', current_price)
            target_high = self.fundamental_analysis.get('target_high_price', current_price * 1.3)
            target_low = self.fundamental_analysis.get('target_low_price', current_price * 0.8)

            # Calculate projections with multiple scenarios
            projection = {
                'current_price': current_price,
                'historical_growth': {
                    '1_month': growth_1m,
                    '3_months': growth_3m,
                    '6_months': growth_6m
                },
                'volatility_annual': volatility * 100,
                'moving_averages': {
                    'MA_50': ma_50,
                    'MA_200': ma_200,
                    'trend': 'Bullish' if current_price > ma_50 > ma_200 else 'Bearish' if current_price < ma_50 < ma_200 else 'Neutral'
                },
                'projections': {
                    '3_months': {
                        'bullish': current_price * (1 + abs(growth_3m) / 100 * 1.2),
                        'base': target_mean if target_mean > 0 else current_price * (1 + growth_3m / 100),
                        'bearish': current_price * (1 - abs(growth_3m) / 100 * 0.5)
                    },
                    '6_months': {
                        'bullish': current_price * (1 + abs(growth_6m) / 100 * 1.5),
                        'base': target_mean if target_mean > 0 else current_price * (1 + growth_6m / 100 * 1.2),
                        'bearish': current_price * (1 - abs(growth_6m) / 100 * 0.8)
                    },
                    '12_months': {
                        'bullish': target_high if target_high > 0 else current_price * 1.5,
                        'base': target_mean if target_mean > 0 else current_price * 1.2,
                        'bearish': target_low if target_low > 0 else current_price * 0.85
                    }
                },
                'analyst_targets': {
                    'mean': target_mean,
                    'high': target_high,
                    'low': target_low,
                    'upside_potential': ((target_mean - current_price) / current_price * 100) if target_mean > 0 else 0
                }
            }

            self.long_term_projection = projection
            print(f"✅ Long-term projection complete")
            return projection

        except Exception as e:
            print(f"⚠️ Error in long-term projection: {e}")
            return {}

    def analyze_earnings(self):
        """Analyze earnings, next earnings call, and market sentiment"""
        print(f"\n📅 Analyzing earnings for {self.symbol}...")

        try:
            earnings_data = {
                'next_earnings_date': None,
                'last_earnings_date': None,
                'earnings_history': [],
                'expected_eps': None,
                'actual_eps': None,
                'surprise_percent': None,
                'days_to_earnings': None,
                'sentiment': 'Neutral',
                'recommendation': ''
            }

            # Get earnings dates
            try:
                earnings_dates = self.stock.earnings_dates
                if earnings_dates is not None and not earnings_dates.empty:
                    # Filter future dates
                    future_earnings = earnings_dates[earnings_dates.index > pd.Timestamp.now()]
                    if not future_earnings.empty:
                        next_date = future_earnings.index[0]
                        earnings_data['next_earnings_date'] = next_date
                        days_to = (next_date - pd.Timestamp.now()).days
                        earnings_data['days_to_earnings'] = days_to

                    # Get last earnings
                    past_earnings = earnings_dates[earnings_dates.index <= pd.Timestamp.now()]
                    if not past_earnings.empty:
                        last_date = past_earnings.index[0]
                        earnings_data['last_earnings_date'] = last_date
            except Exception as e:
                print(f"  ⚠️ Could not fetch earnings dates: {e}")

            # Get earnings history
            try:
                earnings_history = self.stock.earnings_history
                if earnings_history is not None and not earnings_history.empty:
                    recent_earnings = earnings_history.head(4)

                    for idx, row in recent_earnings.iterrows():
                        eps_data = {
                            'date': idx,
                            'eps_estimate': row.get('epsEstimate', 0),
                            'eps_actual': row.get('epsActual', 0),
                            'surprise': row.get('surprisePercent', 0)
                        }
                        earnings_data['earnings_history'].append(eps_data)

                    # Get most recent earnings for sentiment
                    if len(earnings_data['earnings_history']) > 0:
                        latest = earnings_data['earnings_history'][0]
                        earnings_data['expected_eps'] = latest.get('eps_estimate', 0)
                        earnings_data['actual_eps'] = latest.get('eps_actual', 0)
                        earnings_data['surprise_percent'] = latest.get('surprise', 0)

                        # Determine sentiment based on surprise
                        surprise = earnings_data['surprise_percent']
                        if surprise > 5:
                            earnings_data['sentiment'] = 'Very Bullish'
                            earnings_data['recommendation'] = 'Strong earnings beat indicates positive momentum'
                        elif surprise > 0:
                            earnings_data['sentiment'] = 'Bullish'
                            earnings_data['recommendation'] = 'Earnings beat expectations'
                        elif surprise < -5:
                            earnings_data['sentiment'] = 'Bearish'
                            earnings_data['recommendation'] = 'Earnings miss may pressure stock'
                        elif surprise < 0:
                            earnings_data['sentiment'] = 'Slightly Bearish'
                            earnings_data['recommendation'] = 'Minor earnings miss'
                        else:
                            earnings_data['sentiment'] = 'Neutral'
                            earnings_data['recommendation'] = 'Met expectations'

            except Exception as e:
                print(f"  ⚠️ Could not fetch earnings history: {e}")

            # Add guidance for upcoming earnings
            if earnings_data['days_to_earnings'] is not None:
                days = earnings_data['days_to_earnings']
                if days <= 7:
                    earnings_data['short_term_impact'] = 'HIGH - Earnings within a week, expect increased volatility'
                elif days <= 30:
                    earnings_data['short_term_impact'] = 'MEDIUM - Earnings within a month, watch for pre-earnings momentum'
                else:
                    earnings_data['short_term_impact'] = 'LOW - Earnings date is far, minimal immediate impact'
            else:
                earnings_data['short_term_impact'] = 'UNKNOWN - No upcoming earnings date available'

            self.earnings_analysis = earnings_data
            print(f"✅ Earnings analysis complete - Sentiment: {earnings_data['sentiment']}")
            return earnings_data

        except Exception as e:
            print(f"⚠️ Error in earnings analysis: {e}")
            return {}

    def calculate_fibonacci_retracement(self):
        """Calculate Fibonacci retracement levels"""
        print(f"\n📐 Calculating Fibonacci retracement for {self.symbol}...")

        try:
            # Use last 60 days to find swing high and low
            data_60d = self.data.tail(60)

            swing_high = data_60d['High'].max()
            swing_low = data_60d['Low'].min()
            diff = swing_high - swing_low

            # Calculate Fibonacci levels
            fib_levels = {
                'swing_high': swing_high,
                'swing_low': swing_low,
                'level_0': swing_high,  # 0% (High)
                'level_236': swing_high - (0.236 * diff),
                'level_382': swing_high - (0.382 * diff),
                'level_500': swing_high - (0.500 * diff),
                'level_618': swing_high - (0.618 * diff),
                'level_786': swing_high - (0.786 * diff),
                'level_100': swing_low,  # 100% (Low)
                'level_1272': swing_high - (1.272 * diff),  # Extension
                'level_1618': swing_high - (1.618 * diff),  # Extension
            }

            # Determine current position
            current_price = self.data['Close'].iloc[-1]

            if current_price >= fib_levels['level_236']:
                fib_levels['position'] = 'Above 23.6% - Strong uptrend'
                fib_levels['next_support'] = fib_levels['level_236']
                fib_levels['next_resistance'] = swing_high
            elif current_price >= fib_levels['level_382']:
                fib_levels['position'] = 'Between 23.6% - 38.2% - Moderate retracement'
                fib_levels['next_support'] = fib_levels['level_382']
                fib_levels['next_resistance'] = fib_levels['level_236']
            elif current_price >= fib_levels['level_500']:
                fib_levels['position'] = 'Between 38.2% - 50% - Mid-level retracement'
                fib_levels['next_support'] = fib_levels['level_500']
                fib_levels['next_resistance'] = fib_levels['level_382']
            elif current_price >= fib_levels['level_618']:
                fib_levels['position'] = 'Between 50% - 61.8% - Deep retracement'
                fib_levels['next_support'] = fib_levels['level_618']
                fib_levels['next_resistance'] = fib_levels['level_500']
            else:
                fib_levels['position'] = 'Below 61.8% - Very deep retracement'
                fib_levels['next_support'] = swing_low
                fib_levels['next_resistance'] = fib_levels['level_618']

            self.fibonacci_levels = fib_levels
            print(f"✅ Fibonacci analysis complete - Position: {fib_levels['position']}")
            return fib_levels

        except Exception as e:
            print(f"⚠️ Error calculating Fibonacci: {e}")
            return {}

    def detect_candlestick_patterns(self):
        """Detect common candlestick patterns"""
        print(f"\n🕯️ Detecting candlestick patterns for {self.symbol}...")

        try:
            patterns = []
            data = self.data.tail(10)  # Look at last 10 days

            for i in range(2, len(data)):
                curr = data.iloc[i]
                prev = data.iloc[i-1]
                prev2 = data.iloc[i-2]

                open_price = curr['Open']
                close_price = curr['Close']
                high = curr['High']
                low = curr['Low']

                body = abs(close_price - open_price)
                total_range = high - low

                # Doji
                if body < (total_range * 0.1):
                    patterns.append({
                        'date': curr.name,
                        'pattern': 'Doji',
                        'signal': 'Neutral',
                        'description': 'Indecision in the market'
                    })

                # Hammer (bullish)
                if close_price > open_price and (high - close_price) < body * 0.3 and (open_price - low) > body * 2:
                    patterns.append({
                        'date': curr.name,
                        'pattern': 'Hammer',
                        'signal': 'Bullish',
                        'description': 'Potential reversal from downtrend'
                    })

                # Shooting Star (bearish)
                if open_price > close_price and (close_price - low) < body * 0.3 and (high - open_price) > body * 2:
                    patterns.append({
                        'date': curr.name,
                        'pattern': 'Shooting Star',
                        'signal': 'Bearish',
                        'description': 'Potential reversal from uptrend'
                    })

                # Engulfing patterns
                if i >= 1:
                    # Bullish Engulfing
                    if prev['Close'] < prev['Open'] and close_price > open_price:
                        if open_price < prev['Close'] and close_price > prev['Open']:
                            patterns.append({
                                'date': curr.name,
                                'pattern': 'Bullish Engulfing',
                                'signal': 'Bullish',
                                'description': 'Strong bullish reversal signal'
                            })

                    # Bearish Engulfing
                    if prev['Close'] > prev['Open'] and close_price < open_price:
                        if open_price > prev['Close'] and close_price < prev['Open']:
                            patterns.append({
                                'date': curr.name,
                                'pattern': 'Bearish Engulfing',
                                'signal': 'Bearish',
                                'description': 'Strong bearish reversal signal'
                            })

                # Morning Star (bullish)
                if i >= 2:
                    if (prev2['Close'] < prev2['Open'] and  # First candle bearish
                        abs(prev['Close'] - prev['Open']) < body * 0.3 and  # Second candle small
                        close_price > open_price and  # Third candle bullish
                        close_price > prev2['Open']):  # Closes above first candle
                        patterns.append({
                            'date': curr.name,
                            'pattern': 'Morning Star',
                            'signal': 'Bullish',
                            'description': 'Strong three-candle bullish reversal'
                        })

                # Evening Star (bearish)
                if i >= 2:
                    if (prev2['Close'] > prev2['Open'] and  # First candle bullish
                        abs(prev['Close'] - prev['Open']) < body * 0.3 and  # Second candle small
                        close_price < open_price and  # Third candle bearish
                        close_price < prev2['Open']):  # Closes below first candle
                        patterns.append({
                            'date': curr.name,
                            'pattern': 'Evening Star',
                            'signal': 'Bearish',
                            'description': 'Strong three-candle bearish reversal'
                        })

            # Get most recent unique patterns
            unique_patterns = []
            seen_patterns = set()
            for pattern in reversed(patterns):  # Most recent first
                if pattern['pattern'] not in seen_patterns:
                    unique_patterns.append(pattern)
                    seen_patterns.add(pattern['pattern'])

            self.candlestick_patterns = unique_patterns[:5]  # Keep top 5
            print(f"✅ Detected {len(self.candlestick_patterns)} significant patterns")
            return self.candlestick_patterns

        except Exception as e:
            print(f"⚠️ Error detecting patterns: {e}")
            return []

    def analyze_day_trading_signals(self):
        """Analyze signals specifically for day traders"""
        print(f"\n⚡ Analyzing day trading signals for {self.symbol}...")

        try:
            current_price = self.data['Close'].iloc[-1]

            # Calculate intraday indicators
            data_recent = self.data.tail(20)

            # Average True Range (ATR) for volatility
            high_low = data_recent['High'] - data_recent['Low']
            high_close = abs(data_recent['High'] - data_recent['Close'].shift())
            low_close = abs(data_recent['Low'] - data_recent['Close'].shift())
            true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
            atr = true_range.rolling(window=14).mean().iloc[-1]

            # Volume analysis
            avg_volume = data_recent['Volume'].mean()
            current_volume = data_recent['Volume'].iloc[-1]
            volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1

            # Momentum
            momentum_5 = ((current_price - data_recent['Close'].iloc[-5]) / data_recent['Close'].iloc[-5]) * 100
            momentum_10 = ((current_price - data_recent['Close'].iloc[-10]) / data_recent['Close'].iloc[-10]) * 100

            # Support and Resistance levels
            support_1 = data_recent['Low'].min()
            support_2 = data_recent['Low'].nsmallest(2).iloc[-1]
            resistance_1 = data_recent['High'].max()
            resistance_2 = data_recent['High'].nlargest(2).iloc[-1]

            # Calculate pivot points
            high = data_recent['High'].iloc[-1]
            low = data_recent['Low'].iloc[-1]
            close = data_recent['Close'].iloc[-1]
            pivot = (high + low + close) / 3

            day_trading = {
                'volatility_atr': atr,
                'volatility_percent': (atr / current_price) * 100,
                'volume_ratio': volume_ratio,
                'volume_signal': 'High' if volume_ratio > 1.5 else 'Normal' if volume_ratio > 0.7 else 'Low',
                'momentum_5d': momentum_5,
                'momentum_10d': momentum_10,
                'support_levels': [support_1, support_2],
                'resistance_levels': [resistance_1, resistance_2],
                'pivot_point': pivot,
                'target_profit': current_price + (atr * 2),
                'stop_loss': current_price - (atr * 1.5),
                'risk_reward_ratio': 2 / 1.5,
                'signals': []
            }

            # Generate day trading signals
            if volume_ratio > 1.5:
                day_trading['signals'].append('High volume - Increased interest')

            if momentum_5 > 2:
                day_trading['signals'].append('Strong 5-day momentum - Bullish')
            elif momentum_5 < -2:
                day_trading['signals'].append('Negative 5-day momentum - Bearish')

            if current_price < support_2:
                day_trading['signals'].append('Below support - Potential bounce')
            elif current_price > resistance_2:
                day_trading['signals'].append('Above resistance - Potential breakout')

            # Overall day trading recommendation
            tech = self.technical_analysis
            rsi = tech.get('rsi', 50)

            bullish_count = sum([
                volume_ratio > 1.3,
                momentum_5 > 1,
                rsi < 30,
                current_price > pivot
            ])

            bearish_count = sum([
                momentum_5 < -1,
                rsi > 70,
                current_price < pivot
            ])

            if bullish_count >= 3:
                day_trading['recommendation'] = 'BULLISH - Good for long day trades'
                day_trading['strategy'] = 'Look for dips to buy, target resistance levels'
            elif bearish_count >= 3:
                day_trading['recommendation'] = 'BEARISH - Consider short opportunities'
                day_trading['strategy'] = 'Look for rallies to short, target support levels'
            else:
                day_trading['recommendation'] = 'NEUTRAL - Wait for clear signals'
                day_trading['strategy'] = 'Range trading between support and resistance'

            self.day_trading_analysis = day_trading
            print(f"✅ Day trading analysis complete - {day_trading['recommendation']}")
            return day_trading

        except Exception as e:
            print(f"⚠️ Error in day trading analysis: {e}")
            return {}

    def forecast_sector_industry(self):
        """Forecast based on sector and industry trends"""
        print(f"\n🏭 Analyzing sector/industry forecast for {self.symbol}...")

        try:
            sector = self.fundamental_analysis.get('sector', 'Unknown')
            industry = self.fundamental_analysis.get('industry', 'Unknown')

            sector_data = {
                'sector': sector,
                'industry': industry,
                'sector_outlook': '',
                'growth_potential': 0,
                'risk_level': 'Medium',
                'key_drivers': [],
                'year_forecast': {}
            }

            # Sector-specific analysis (simplified - in production, this would come from external APIs)
            sector_outlooks = {
                'Technology': {
                    'outlook': 'Positive - Continued digital transformation and AI adoption',
                    'growth': 15,
                    'risk': 'Medium',
                    'drivers': ['AI/ML adoption', 'Cloud computing growth', 'Cybersecurity demand']
                },
                'Healthcare': {
                    'outlook': 'Stable - Aging population and medical innovation',
                    'growth': 10,
                    'risk': 'Low',
                    'drivers': ['Aging demographics', 'Medical technology', 'Drug development']
                },
                'Financial Services': {
                    'outlook': 'Moderate - Interest rate environment affects margins',
                    'growth': 8,
                    'risk': 'Medium',
                    'drivers': ['Interest rates', 'Digital banking', 'Fintech disruption']
                },
                'Consumer Cyclical': {
                    'outlook': 'Variable - Depends on consumer spending',
                    'growth': 7,
                    'risk': 'High',
                    'drivers': ['Consumer confidence', 'E-commerce growth', 'Inflation impact']
                },
                'Energy': {
                    'outlook': 'Volatile - Commodity price dependent',
                    'growth': 12,
                    'risk': 'High',
                    'drivers': ['Oil prices', 'Green energy transition', 'Geopolitical factors']
                }
            }

            if sector in sector_outlooks:
                outlook = sector_outlooks[sector]
                sector_data['sector_outlook'] = outlook['outlook']
                sector_data['growth_potential'] = outlook['growth']
                sector_data['risk_level'] = outlook['risk']
                sector_data['key_drivers'] = outlook['drivers']
            else:
                sector_data['sector_outlook'] = 'Neutral - No specific sector data available'
                sector_data['growth_potential'] = 8
                sector_data['key_drivers'] = ['Market conditions', 'Economic growth', 'Industry trends']

            # Calculate year forecast based on sector and fundamentals
            current_price = self.technical_analysis['current_price']
            fund_score = self.fundamental_analysis.get('fundamental_score', 50)

            # Combine sector growth with fundamental strength
            combined_factor = (sector_data['growth_potential'] + (fund_score - 50) / 5) / 100

            sector_data['year_forecast'] = {
                'conservative': current_price * (1 + combined_factor * 0.7),
                'moderate': current_price * (1 + combined_factor * 1.0),
                'aggressive': current_price * (1 + combined_factor * 1.3),
                'expected_return': combined_factor * 100
            }

            self.sector_forecast = sector_data
            print(f"✅ Sector forecast complete - {sector}: {sector_data['sector_outlook']}")
            return sector_data

        except Exception as e:
            print(f"⚠️ Error in sector forecast: {e}")
            return {}

    def generate_interactive_chart(self):
        """Generate interactive chart with technical indicators"""
        if not PLOTLY_AVAILABLE:
            print("⚠️ Plotly not available, skipping interactive chart")
            return None

        print(f"\n📊 Generating interactive chart for {self.symbol}...")

        try:
            # Get last 90 days of data
            chart_data = self.data.tail(90)

            # Create subplots
            fig = make_subplots(
                rows=3, cols=1,
                shared_xaxes=True,
                vertical_spacing=0.05,
                row_heights=[0.6, 0.2, 0.2],
                subplot_titles=(f'{self.symbol} Price & Technical Indicators', 'Volume', 'RSI')
            )

            # Candlestick chart
            fig.add_trace(
                go.Candlestick(
                    x=chart_data.index,
                    open=chart_data['Open'],
                    high=chart_data['High'],
                    low=chart_data['Low'],
                    close=chart_data['Close'],
                    name='OHLC'
                ),
                row=1, col=1
            )

            # Add moving averages
            ma_20 = chart_data['Close'].rolling(window=20).mean()
            ma_50 = chart_data['Close'].rolling(window=50).mean()

            fig.add_trace(
                go.Scatter(x=chart_data.index, y=ma_20, name='MA 20', line=dict(color='orange', width=1)),
                row=1, col=1
            )
            fig.add_trace(
                go.Scatter(x=chart_data.index, y=ma_50, name='MA 50', line=dict(color='blue', width=1)),
                row=1, col=1
            )

            # Add Fibonacci levels if available
            if self.fibonacci_levels:
                fib = self.fibonacci_levels
                for level, value in [('23.6%', fib.get('level_236')), ('38.2%', fib.get('level_382')),
                                     ('50%', fib.get('level_500')), ('61.8%', fib.get('level_618'))]:
                    if value:
                        fig.add_hline(y=value, line_dash="dash", line_color="gray",
                                     annotation_text=f"Fib {level}", row=1, col=1)

            # Volume
            colors = ['red' if row['Close'] < row['Open'] else 'green' for idx, row in chart_data.iterrows()]
            fig.add_trace(
                go.Bar(x=chart_data.index, y=chart_data['Volume'], name='Volume', marker_color=colors),
                row=2, col=1
            )

            # RSI
            delta = chart_data['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))

            fig.add_trace(
                go.Scatter(x=chart_data.index, y=rsi, name='RSI', line=dict(color='purple', width=2)),
                row=3, col=1
            )
            fig.add_hline(y=70, line_dash="dash", line_color="red", annotation_text="Overbought", row=3, col=1)
            fig.add_hline(y=30, line_dash="dash", line_color="green", annotation_text="Oversold", row=3, col=1)

            # Update layout
            fig.update_layout(
                title=f'{self.symbol} - Comprehensive Technical Analysis',
                xaxis_rangeslider_visible=False,
                height=900,
                showlegend=True,
                template='plotly_white'
            )

            # Save chart
            chart_filename = f'chart_{self.symbol}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.html'
            fig.write_html(chart_filename)
            print(f"✅ Interactive chart saved: {chart_filename}")
            return chart_filename

        except Exception as e:
            print(f"⚠️ Error generating chart: {e}")
            return None

    def generate_signal(self, rsi, macd, macd_signal, bb_position, current_price, sma_50):
        """Generate buy/sell/hold signal based on technical indicators"""
        signals = []
        score = 0

        # RSI Analysis
        if rsi < 30:
            signals.append("Oversold (RSI)")
            score += 2
        elif rsi > 70:
            signals.append("Overbought (RSI)")
            score -= 2

        # MACD Analysis
        if macd > macd_signal:
            signals.append("Bullish (MACD)")
            score += 1
        else:
            signals.append("Bearish (MACD)")
            score -= 1

        # Bollinger Bands
        if bb_position < 20:
            signals.append("Near Lower Band")
            score += 1
        elif bb_position > 80:
            signals.append("Near Upper Band")
            score -= 1

        # Moving Average
        if current_price > sma_50:
            signals.append("Above SMA50")
            score += 1
        else:
            signals.append("Below SMA50")
            score -= 1

        # Final signal
        if score >= 2:
            final_signal = "BUY"
            color = "#28a745"
        elif score <= -2:
            final_signal = "SELL"
            color = "#dc3545"
        else:
            final_signal = "HOLD"
            color = "#ffc107"

        return final_signal, color, signals

    def perform_technical_analysis(self):
        """Perform complete technical analysis"""
        print(f"\n🔧 Performing technical analysis for {self.symbol}...")

        current_price = self.data['Close'].iloc[-1]
        prev_close = self.data['Close'].iloc[-2]
        change = current_price - prev_close
        change_pct = (change / prev_close) * 100

        # Calculate all technical indicators
        rsi = self.calculate_rsi(self.data)
        macd, macd_signal = self.calculate_macd(self.data)
        bb = self.calculate_bollinger_bands(self.data)
        ma = self.calculate_moving_averages(self.data)

        # Generate trading signal
        signal, signal_color, signal_reasons = self.generate_signal(
            rsi, macd, macd_signal, bb['position'], current_price, ma['SMA_50']
        )

        self.technical_analysis = {
            'current_price': current_price,
            'change': change,
            'change_pct': change_pct,
            'rsi': rsi,
            'macd': macd,
            'macd_signal': macd_signal,
            'bollinger': bb,
            'moving_averages': ma,
            'signal': signal,
            'signal_color': signal_color,
            'signal_reasons': signal_reasons
        }

        print(f"✅ Technical analysis complete - Signal: {signal}")
        return self.technical_analysis

    def run_complete_analysis(self):
        """Run complete comprehensive analysis"""
        print("\n" + "=" * 70)
        print(f"  COMPREHENSIVE STOCK ANALYSIS - {self.symbol}")
        print("=" * 70)

        # Step 1: Fetch data
        if not self.fetch_stock_data():
            return False

        # Step 2: Fundamental analysis
        self.analyze_fundamentals()

        # Step 3: Technical analysis
        self.perform_technical_analysis()

        # Step 4: Earnings analysis
        self.analyze_earnings()

        # Step 5: Fibonacci retracement
        self.calculate_fibonacci_retracement()

        # Step 6: Candlestick patterns
        self.detect_candlestick_patterns()

        # Step 7: Day trading analysis
        self.analyze_day_trading_signals()

        # Step 8: Get investor strategies
        self.get_investor_strategies()

        # Step 9: Short-term forecast (next 5 weekdays)
        self.predict_next_5_weekdays()

        # Step 10: Long-term projection
        self.predict_long_term()

        # Step 11: Sector/Industry forecast
        self.forecast_sector_industry()

        # Step 12: Generate interactive chart
        self.generate_interactive_chart()

        print("\n" + "=" * 70)
        print("  ✅ ANALYSIS COMPLETE")
        print("=" * 70)

        return True

    def generate_html_report(self):
        """Generate comprehensive HTML5 report"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        fund = self.fundamental_analysis
        tech = self.technical_analysis

        html = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Comprehensive Stock Analysis - {self.symbol}</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 20px;
        }}
        .container {{
            max-width: 1400px;
            margin: 0 auto;
            background: white;
            border-radius: 15px;
            padding: 30px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
        }}
        .header {{
            text-align: center;
            margin-bottom: 30px;
            border-bottom: 3px solid #667eea;
            padding-bottom: 20px;
        }}
        .header h1 {{
            color: #333;
            font-size: 2.5em;
            margin-bottom: 10px;
        }}
        .header p {{
            color: #666;
            font-size: 1.1em;
        }}
        .timestamp {{
            text-align: center;
            color: #888;
            margin-bottom: 30px;
            font-style: italic;
        }}
        .section {{
            background: #f8f9fa;
            border-radius: 10px;
            padding: 25px;
            margin-bottom: 30px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }}
        .section h2 {{
            color: #667eea;
            margin-bottom: 20px;
            padding-bottom: 10px;
            border-bottom: 2px solid #667eea;
        }}
        .stock-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 30px;
            padding: 30px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            border-radius: 10px;
            color: white;
        }}
        .stock-title {{
            flex: 1;
        }}
        .stock-title h1 {{
            color: white;
            font-size: 2.5em;
            margin-bottom: 5px;
        }}
        .stock-title .company {{
            color: #f0f0f0;
            font-size: 1.2em;
        }}
        .stock-title .meta {{
            color: #e0e0e0;
            font-size: 0.9em;
            margin-top: 10px;
        }}
        .price-info {{
            text-align: right;
        }}
        .current-price {{
            font-size: 2em;
            font-weight: bold;
            color: #333;
        }}
        .price-change {{
            font-size: 1.2em;
            font-weight: bold;
            margin-top: 5px;
        }}
        .price-change.positive {{
            color: #28a745;
        }}
        .price-change.negative {{
            color: #dc3545;
        }}
        .signal-badge {{
            display: inline-block;
            padding: 10px 25px;
            border-radius: 25px;
            font-weight: bold;
            font-size: 1.2em;
            color: white;
            margin-top: 10px;
        }}
        .indicators {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin: 20px 0;
        }}
        .indicator {{
            background: white;
            padding: 15px;
            border-radius: 8px;
            border: 1px solid #dee2e6;
        }}
        .indicator h4 {{
            color: #667eea;
            margin-bottom: 10px;
            font-size: 0.9em;
            text-transform: uppercase;
        }}
        .indicator-value {{
            font-size: 1.5em;
            font-weight: bold;
            color: #333;
        }}
        .predictions {{
            margin-top: 20px;
            background: white;
            padding: 20px;
            border-radius: 8px;
            border: 2px solid #667eea;
        }}
        .predictions h3 {{
            color: #667eea;
            margin-bottom: 15px;
        }}
        .prediction-table {{
            width: 100%;
            border-collapse: collapse;
        }}
        .prediction-table th {{
            background: #667eea;
            color: white;
            padding: 12px;
            text-align: left;
        }}
        .prediction-table td {{
            padding: 12px;
            border-bottom: 1px solid #dee2e6;
        }}
        .prediction-table tr:hover {{
            background: #f8f9fa;
        }}
        .signal-reasons {{
            margin-top: 15px;
            padding: 15px;
            background: #fff3cd;
            border-radius: 8px;
            border-left: 4px solid #ffc107;
        }}
        .signal-reasons h4 {{
            color: #856404;
            margin-bottom: 10px;
        }}
        .signal-reasons ul {{
            list-style: none;
            padding-left: 0;
        }}
        .signal-reasons li {{
            padding: 5px 0;
            color: #856404;
        }}
        .signal-reasons li:before {{
            content: "→ ";
            font-weight: bold;
        }}
        .footer {{
            text-align: center;
            margin-top: 30px;
            padding-top: 20px;
            border-top: 2px solid #dee2e6;
            color: #666;
            font-size: 0.9em;
        }}
        .disclaimer {{
            background: #fff3cd;
            padding: 15px;
            border-radius: 8px;
            margin-top: 20px;
            border-left: 4px solid #ffc107;
        }}
        .disclaimer strong {{
            color: #856404;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>📊 Comprehensive Stock Analysis Report</h1>
            <p>Fundamental & Technical Analysis with Investor Strategies</p>
        </div>

        <div class="timestamp">
            Generated on: {timestamp}
        </div>

        <!-- Stock Header -->
        <div class="stock-header">
            <div class="stock-title">
                <h1>{self.symbol}</h1>
                <span class="company">{fund.get('company_name', self.symbol)}</span>
                <div class="meta">
                    {fund.get('sector', 'N/A')} | {fund.get('industry', 'N/A')}
                </div>
            </div>
            <div class="price-info">
                <div class="current-price">${tech['current_price']:.2f}</div>
                <div class="price-change {'positive' if tech['change'] >= 0 else 'negative'}">
                    {'+' if tech['change'] >= 0 else ''}{tech['change']:.2f} ({'+' if tech['change_pct'] >= 0 else ''}{tech['change_pct']:.2f}%)
                </div>
                <span class="signal-badge" style="background-color: {tech['signal_color']}">
                    {tech['signal']}
                </span>
            </div>
        </div>

        <!-- Fundamental Analysis Section -->
        <div class="section">
            <h2>💼 Fundamental Analysis</h2>
            <div class="indicators">
                <div class="indicator">
                    <h4>Market Cap</h4>
                    <div class="indicator-value">${fund.get('market_cap', 0) / 1e9:.2f}B</div>
                </div>
                <div class="indicator">
                    <h4>P/E Ratio</h4>
                    <div class="indicator-value">{fund.get('pe_ratio', 0):.2f}</div>
                </div>
                <div class="indicator">
                    <h4>EPS</h4>
                    <div class="indicator-value">${fund.get('eps_current', 0):.2f}</div>
                </div>
                <div class="indicator">
                    <h4>Profit Margin</h4>
                    <div class="indicator-value">{fund.get('profit_margin', 0):.2f}%</div>
                </div>
                <div class="indicator">
                    <h4>ROE</h4>
                    <div class="indicator-value">{fund.get('roe', 0):.2f}%</div>
                </div>
                <div class="indicator">
                    <h4>Revenue Growth</h4>
                    <div class="indicator-value">{fund.get('revenue_growth', 0):.2f}%</div>
                </div>
                <div class="indicator">
                    <h4>Debt/Equity</h4>
                    <div class="indicator-value">{fund.get('debt_to_equity', 0):.2f}</div>
                </div>
                <div class="indicator">
                    <h4>Dividend Yield</h4>
                    <div class="indicator-value">{fund.get('dividend_yield', 0):.2f}%</div>
                </div>
                <div class="indicator">
                    <h4>Fundamental Score</h4>
                    <div class="indicator-value" style="color: {'#28a745' if fund.get('fundamental_score', 0) > 60 else '#ffc107' if fund.get('fundamental_score', 0) > 40 else '#dc3545'}">{fund.get('fundamental_score', 0):.0f}/100</div>
                </div>
            </div>

            <div class="predictions" style="margin-top: 20px;">
                <h3>Analyst Recommendations</h3>
                <p><strong>Recommendation:</strong> {fund.get('recommendation', 'N/A').upper()}</p>
                <p><strong>Number of Analysts:</strong> {fund.get('number_of_analysts', 0)}</p>
                <p><strong>Target Mean Price:</strong> ${fund.get('target_mean_price', 0):.2f}</p>
                <p><strong>Target Range:</strong> ${fund.get('target_low_price', 0):.2f} - ${fund.get('target_high_price', 0):.2f}</p>
            </div>
        </div>

        <!-- Technical Analysis Section -->
        <div class="section">
            <h2>🔧 Technical Analysis</h2>
            <div class="indicators">
                <div class="indicator">
                    <h4>RSI (14)</h4>
                    <div class="indicator-value">{tech['rsi']:.2f}</div>
                </div>
                <div class="indicator">
                    <h4>MACD</h4>
                    <div class="indicator-value">{tech['macd']:.2f}</div>
                </div>
                <div class="indicator">
                    <h4>MACD Signal</h4>
                    <div class="indicator-value">{tech['macd_signal']:.2f}</div>
                </div>
                <div class="indicator">
                    <h4>SMA 20</h4>
                    <div class="indicator-value">${tech['moving_averages']['SMA_20']:.2f}</div>
                </div>
                <div class="indicator">
                    <h4>SMA 50</h4>
                    <div class="indicator-value">${tech['moving_averages']['SMA_50']:.2f}</div>
                </div>
                <div class="indicator">
                    <h4>Bollinger Position</h4>
                    <div class="indicator-value">{tech['bollinger']['position']:.1f}%</div>
                </div>
            </div>

            <div class="signal-reasons">
                <h4>Technical Signals:</h4>
                <ul>
"""

        for reason in tech['signal_reasons']:
            html += f"                    <li>{reason}</li>\n"

        html += f"""
                </ul>
            </div>
        </div>

        <!-- Investor Strategies Section -->
        <div class="section">
            <h2>👥 Top Investor Strategies</h2>
"""

        if self.investor_strategies:
            html += """
            <table class="prediction-table">
                <thead>
                    <tr>
                        <th>Holder</th>
                        <th>Shares</th>
                        <th>Strategy</th>
                        <th>Sentiment</th>
                    </tr>
                </thead>
                <tbody>
"""
            for investor in self.investor_strategies[:5]:  # Top 5
                if investor.get('type') != 'Major Holders Summary':
                    html += f"""
                    <tr>
                        <td>{investor.get('holder', 'N/A')}</td>
                        <td>{investor.get('shares', 0):,}</td>
                        <td>{investor.get('strategy', 'N/A')}</td>
                        <td style="color: {'#28a745' if investor.get('sentiment') == 'Bullish' else '#ffc107'}">{investor.get('sentiment', 'N/A')}</td>
                    </tr>
"""
            html += """
                </tbody>
            </table>
"""
        else:
            html += "<p>No institutional investor data available.</p>"

        html += """
        </div>

        <!-- Next 5 Weekdays Forecast -->
        <div class="section">
            <h2>📅 Next 5 Weekdays Forecast</h2>
            <table class="prediction-table">
                <thead>
                    <tr>
                        <th>Date</th>
                        <th>Day</th>
                        <th>Predicted Price</th>
                        <th>Change %</th>
                        <th>Confidence Range</th>
                    </tr>
                </thead>
                <tbody>
"""

        for forecast in self.short_term_forecast:
            change_class = "positive" if forecast['change_from_current'] >= 0 else "negative"
            change_symbol = "+" if forecast['change_from_current'] >= 0 else ""
            html += f"""
                    <tr>
                        <td>{forecast['date'].strftime('%Y-%m-%d')}</td>
                        <td><strong>{forecast['day_name']}</strong></td>
                        <td style="font-weight: bold;">${forecast['predicted_price']:.2f}</td>
                        <td class="price-change {change_class}">
                            {change_symbol}{forecast['change_from_current']:.2f}%
                        </td>
                        <td>${forecast['confidence_lower']:.2f} - ${forecast['confidence_upper']:.2f}</td>
                    </tr>
"""

        html += """
                </tbody>
            </table>
        </div>

        <!-- Long-term Projection -->
        <div class="section">
            <h2>📈 Long-term Projection</h2>
"""

        if self.long_term_projection:
            proj = self.long_term_projection
            html += f"""
            <div class="predictions">
                <h3>Historical Growth Rates</h3>
                <div class="indicators">
                    <div class="indicator">
                        <h4>1 Month</h4>
                        <div class="indicator-value" style="color: {'#28a745' if proj['historical_growth']['1_month'] > 0 else '#dc3545'}">{proj['historical_growth']['1_month']:.2f}%</div>
                    </div>
                    <div class="indicator">
                        <h4>3 Months</h4>
                        <div class="indicator-value" style="color: {'#28a745' if proj['historical_growth']['3_months'] > 0 else '#dc3545'}">{proj['historical_growth']['3_months']:.2f}%</div>
                    </div>
                    <div class="indicator">
                        <h4>6 Months</h4>
                        <div class="indicator-value" style="color: {'#28a745' if proj['historical_growth']['6_months'] > 0 else '#dc3545'}">{proj['historical_growth']['6_months']:.2f}%</div>
                    </div>
                </div>

                <h3 style="margin-top: 30px;">Price Projections (Scenarios)</h3>
                <table class="prediction-table">
                    <thead>
                        <tr>
                            <th>Period</th>
                            <th>Bearish</th>
                            <th>Base</th>
                            <th>Bullish</th>
                        </tr>
                    </thead>
                    <tbody>
                        <tr>
                            <td><strong>3 Months</strong></td>
                            <td>${proj['projections']['3_months']['bearish']:.2f}</td>
                            <td style="font-weight: bold;">${proj['projections']['3_months']['base']:.2f}</td>
                            <td>${proj['projections']['3_months']['bullish']:.2f}</td>
                        </tr>
                        <tr>
                            <td><strong>6 Months</strong></td>
                            <td>${proj['projections']['6_months']['bearish']:.2f}</td>
                            <td style="font-weight: bold;">${proj['projections']['6_months']['base']:.2f}</td>
                            <td>${proj['projections']['6_months']['bullish']:.2f}</td>
                        </tr>
                        <tr>
                            <td><strong>12 Months</strong></td>
                            <td>${proj['projections']['12_months']['bearish']:.2f}</td>
                            <td style="font-weight: bold;">${proj['projections']['12_months']['base']:.2f}</td>
                            <td>${proj['projections']['12_months']['bullish']:.2f}</td>
                        </tr>
                    </tbody>
                </table>

                <h3 style="margin-top: 30px;">Market Trend</h3>
                <p><strong>Trend:</strong> {proj['moving_averages']['trend']}</p>
                <p><strong>MA 50:</strong> ${proj['moving_averages']['MA_50']:.2f}</p>
                <p><strong>MA 200:</strong> ${proj['moving_averages']['MA_200']:.2f}</p>
                <p><strong>Annual Volatility:</strong> {proj['volatility_annual']:.2f}%</p>
                <p><strong>Analyst Upside Potential:</strong> <span style="color: {'#28a745' if proj['analyst_targets']['upside_potential'] > 0 else '#dc3545'}">{proj['analyst_targets']['upside_potential']:.2f}%</span></p>
            </div>
"""

        html += """
        </div>

        <!-- Earnings Analysis -->
        <div class="section">
            <h2>📅 Earnings Analysis & Market Sentiment</h2>
"""

        if self.earnings_analysis:
            earn = self.earnings_analysis
            html += f"""
            <div class="predictions">
                <h3>Upcoming Earnings</h3>
                <p><strong>Next Earnings Date:</strong> {earn.get('next_earnings_date').strftime('%Y-%m-%d') if earn.get('next_earnings_date') else 'Not Available'}</p>
                <p><strong>Days to Earnings:</strong> {earn.get('days_to_earnings', 'N/A')} days</p>
                <p><strong>Short-term Impact:</strong> {earn.get('short_term_impact', 'N/A')}</p>

                <h3 style="margin-top: 20px;">Last Earnings Results</h3>
                <p><strong>Last Earnings Date:</strong> {earn.get('last_earnings_date').strftime('%Y-%m-%d') if earn.get('last_earnings_date') else 'Not Available'}</p>
                <p><strong>Expected EPS:</strong> ${earn.get('expected_eps', 0):.2f}</p>
                <p><strong>Actual EPS:</strong> ${earn.get('actual_eps', 0):.2f}</p>
                <p><strong>Surprise:</strong> <span style="color: {'#28a745' if earn.get('surprise_percent', 0) > 0 else '#dc3545'}">{earn.get('surprise_percent', 0):.2f}%</span></p>
                <p><strong>Market Sentiment:</strong> <span style="color: {'#28a745' if 'Bullish' in earn.get('sentiment', '') else '#dc3545' if 'Bearish' in earn.get('sentiment', '') else '#ffc107'}"><strong>{earn.get('sentiment', 'N/A')}</strong></span></p>
                <p><strong>Recommendation:</strong> {earn.get('recommendation', 'N/A')}</p>
            </div>
"""

            if earn.get('earnings_history'):
                html += """
            <h3 style="margin-top: 20px;">Earnings History (Last 4 Quarters)</h3>
            <table class="prediction-table">
                <thead>
                    <tr>
                        <th>Date</th>
                        <th>EPS Estimate</th>
                        <th>EPS Actual</th>
                        <th>Surprise %</th>
                    </tr>
                </thead>
                <tbody>
"""
                for hist in earn['earnings_history'][:4]:
                    surprise = hist.get('surprise', 0)
                    html += f"""
                    <tr>
                        <td>{hist.get('date').strftime('%Y-%m-%d') if hasattr(hist.get('date'), 'strftime') else 'N/A'}</td>
                        <td>${hist.get('eps_estimate', 0):.2f}</td>
                        <td>${hist.get('eps_actual', 0):.2f}</td>
                        <td style="color: {'#28a745' if surprise > 0 else '#dc3545'}">{surprise:.2f}%</td>
                    </tr>
"""
                html += """
                </tbody>
            </table>
"""

        html += """
        </div>

        <!-- Fibonacci Retracement -->
        <div class="section">
            <h2>📐 Fibonacci Retracement Levels</h2>
"""

        if self.fibonacci_levels:
            fib = self.fibonacci_levels
            html += f"""
            <div class="predictions">
                <p><strong>Current Position:</strong> {fib.get('position', 'N/A')}</p>
                <p><strong>Next Support:</strong> ${fib.get('next_support', 0):.2f}</p>
                <p><strong>Next Resistance:</strong> ${fib.get('next_resistance', 0):.2f}</p>

                <h3 style="margin-top: 20px;">Fibonacci Levels</h3>
                <div class="indicators">
                    <div class="indicator">
                        <h4>Swing High (0%)</h4>
                        <div class="indicator-value">${fib.get('swing_high', 0):.2f}</div>
                    </div>
                    <div class="indicator">
                        <h4>23.6%</h4>
                        <div class="indicator-value">${fib.get('level_236', 0):.2f}</div>
                    </div>
                    <div class="indicator">
                        <h4>38.2%</h4>
                        <div class="indicator-value">${fib.get('level_382', 0):.2f}</div>
                    </div>
                    <div class="indicator">
                        <h4>50.0%</h4>
                        <div class="indicator-value">${fib.get('level_500', 0):.2f}</div>
                    </div>
                    <div class="indicator">
                        <h4>61.8%</h4>
                        <div class="indicator-value">${fib.get('level_618', 0):.2f}</div>
                    </div>
                    <div class="indicator">
                        <h4>78.6%</h4>
                        <div class="indicator-value">${fib.get('level_786', 0):.2f}</div>
                    </div>
                    <div class="indicator">
                        <h4>Swing Low (100%)</h4>
                        <div class="indicator-value">${fib.get('swing_low', 0):.2f}</div>
                    </div>
                </div>
            </div>
"""

        html += """
        </div>

        <!-- Candlestick Patterns -->
        <div class="section">
            <h2>🕯️ Recent Candlestick Patterns</h2>
"""

        if self.candlestick_patterns:
            html += """
            <table class="prediction-table">
                <thead>
                    <tr>
                        <th>Date</th>
                        <th>Pattern</th>
                        <th>Signal</th>
                        <th>Description</th>
                    </tr>
                </thead>
                <tbody>
"""
            for pattern in self.candlestick_patterns:
                signal_color = '#28a745' if pattern['signal'] == 'Bullish' else '#dc3545' if pattern['signal'] == 'Bearish' else '#ffc107'
                html += f"""
                    <tr>
                        <td>{pattern['date'].strftime('%Y-%m-%d') if hasattr(pattern['date'], 'strftime') else 'N/A'}</td>
                        <td><strong>{pattern['pattern']}</strong></td>
                        <td style="color: {signal_color}"><strong>{pattern['signal']}</strong></td>
                        <td>{pattern['description']}</td>
                    </tr>
"""
            html += """
                </tbody>
            </table>
"""
        else:
            html += "<p>No significant candlestick patterns detected in recent trading days.</p>"

        html += """
        </div>

        <!-- Day Trading Analysis -->
        <div class="section">
            <h2>⚡ Day Trading Analysis</h2>
"""

        if self.day_trading_analysis:
            day = self.day_trading_analysis
            rec_color = '#28a745' if 'BULLISH' in day.get('recommendation', '') else '#dc3545' if 'BEARISH' in day.get('recommendation', '') else '#ffc107'

            html += f"""
            <div class="predictions">
                <h3>Day Trading Recommendation</h3>
                <p style="font-size: 1.3em; color: {rec_color}"><strong>{day.get('recommendation', 'N/A')}</strong></p>
                <p><strong>Strategy:</strong> {day.get('strategy', 'N/A')}</p>

                <h3 style="margin-top: 20px;">Key Metrics for Day Traders</h3>
                <div class="indicators">
                    <div class="indicator">
                        <h4>ATR (Volatility)</h4>
                        <div class="indicator-value">${day.get('volatility_atr', 0):.2f}</div>
                        <small>{day.get('volatility_percent', 0):.2f}% of price</small>
                    </div>
                    <div class="indicator">
                        <h4>Volume Signal</h4>
                        <div class="indicator-value">{day.get('volume_signal', 'N/A')}</div>
                        <small>Ratio: {day.get('volume_ratio', 0):.2f}x</small>
                    </div>
                    <div class="indicator">
                        <h4>5-Day Momentum</h4>
                        <div class="indicator-value" style="color: {'#28a745' if day.get('momentum_5d', 0) > 0 else '#dc3545'}">{day.get('momentum_5d', 0):+.2f}%</div>
                    </div>
                    <div class="indicator">
                        <h4>Pivot Point</h4>
                        <div class="indicator-value">${day.get('pivot_point', 0):.2f}</div>
                    </div>
                </div>

                <h3 style="margin-top: 20px;">Trading Levels</h3>
                <p><strong>Support Levels:</strong> ${day['support_levels'][0]:.2f}, ${day['support_levels'][1]:.2f}</p>
                <p><strong>Resistance Levels:</strong> ${day['resistance_levels'][0]:.2f}, ${day['resistance_levels'][1]:.2f}</p>
                <p><strong>Target Profit:</strong> <span style="color: #28a745">${day.get('target_profit', 0):.2f}</span></p>
                <p><strong>Stop Loss:</strong> <span style="color: #dc3545">${day.get('stop_loss', 0):.2f}</span></p>
                <p><strong>Risk/Reward Ratio:</strong> {day.get('risk_reward_ratio', 0):.2f}</p>
            </div>

            <div class="signal-reasons">
                <h4>Day Trading Signals:</h4>
                <ul>
"""
            for signal in day.get('signals', []):
                html += f"                    <li>{signal}</li>\n"

            html += """
                </ul>
            </div>
"""

        html += """
        </div>

        <!-- Sector/Industry Forecast -->
        <div class="section">
            <h2>🏭 Sector & Industry Forecast</h2>
"""

        if self.sector_forecast:
            sector = self.sector_forecast
            html += f"""
            <div class="predictions">
                <p><strong>Sector:</strong> {sector.get('sector', 'N/A')}</p>
                <p><strong>Industry:</strong> {sector.get('industry', 'N/A')}</p>
                <p><strong>Outlook:</strong> {sector.get('sector_outlook', 'N/A')}</p>
                <p><strong>Growth Potential:</strong> {sector.get('growth_potential', 0)}%</p>
                <p><strong>Risk Level:</strong> {sector.get('risk_level', 'N/A')}</p>

                <h3 style="margin-top: 20px;">Key Sector Drivers</h3>
                <ul>
"""
            for driver in sector.get('key_drivers', []):
                html += f"                    <li>{driver}</li>\n"

            if sector.get('year_forecast'):
                year_f = sector['year_forecast']
                html += f"""
                </ul>

                <h3 style="margin-top: 20px;">1-Year Sector-Based Forecast</h3>
                <div class="indicators">
                    <div class="indicator">
                        <h4>Conservative</h4>
                        <div class="indicator-value">${year_f.get('conservative', 0):.2f}</div>
                    </div>
                    <div class="indicator">
                        <h4>Moderate</h4>
                        <div class="indicator-value">${year_f.get('moderate', 0):.2f}</div>
                    </div>
                    <div class="indicator">
                        <h4>Aggressive</h4>
                        <div class="indicator-value">${year_f.get('aggressive', 0):.2f}</div>
                    </div>
                </div>
                <p style="margin-top: 15px;"><strong>Expected Return (1 Year):</strong> <span style="color: {'#28a745' if year_f.get('expected_return', 0) > 0 else '#dc3545'}"><strong>{year_f.get('expected_return', 0):.2f}%</strong></span></p>
"""

        html += """
            </div>
        </div>

        <!-- Disclaimer -->
        <div class="disclaimer">
            <strong>⚠️ Disclaimer:</strong> This analysis is for educational and informational purposes only.
            It should not be considered as financial advice. Stock predictions are based on fundamental and technical analysis
            and historical data, which cannot guarantee future performance. Institutional investor data is based on latest
            available filings. Always conduct your own research and consult with a qualified financial advisor before making investment decisions.
        </div>

        <div class="footer">
            <p>Generated by Comprehensive Stock Analysis System</p>
            <p>Analysis includes: Fundamental Metrics, Technical Indicators (with Fibonacci & Candlestick Patterns), Earnings Analysis, Investor Strategies, Day Trading Signals, Weekday Forecasts, Sector Forecast, Long-term Projections</p>
            <p>Interactive chart with technical indicators generated separately</p>
        </div>
    </div>
</body>
</html>
"""
        return html

    def save_report(self, html_content):
        """Save HTML and PDF reports"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"stock_analysis_{self.symbol}_{timestamp}"

        # Save HTML
        html_filename = f"{filename}.html"
        with open(html_filename, 'w', encoding='utf-8') as f:
            f.write(html_content)
        print(f"\n✅ HTML report saved: {html_filename}")

        # Save PDF
        if PDFKIT_AVAILABLE:
            try:
                pdf_filename = f"{filename}.pdf"
                pdfkit.from_string(html_content, pdf_filename)
                print(f"✅ PDF report saved: {pdf_filename}")
            except Exception as e:
                print(f"⚠️ PDF generation failed: {e}")
                print("You can manually convert the HTML file to PDF using a browser.")
        else:
            print("⚠️ PDF generation skipped (pdfkit not available)")
            print(f"You can open {html_filename} in a browser and print to PDF.")

        return html_filename

    def print_summary(self):
        """Print analysis summary to console"""
        print("\n" + "=" * 70)
        print("  ANALYSIS SUMMARY")
        print("=" * 70)

        fund = self.fundamental_analysis
        tech = self.technical_analysis

        print(f"\n{self.symbol} - {fund.get('company_name', 'N/A')}")
        print(f"Sector: {fund.get('sector', 'N/A')} | Industry: {fund.get('industry', 'N/A')}")
        print(f"\nCurrent Price: ${tech['current_price']:.2f}")
        print(f"Change: {'+' if tech['change'] >= 0 else ''}{tech['change']:.2f} ({'+' if tech['change_pct'] >= 0 else ''}{tech['change_pct']:.2f}%)")
        print(f"Signal: {tech['signal']}")

        print(f"\n--- FUNDAMENTALS ---")
        print(f"Market Cap: ${fund.get('market_cap', 0) / 1e9:.2f}B")
        print(f"P/E Ratio: {fund.get('pe_ratio', 0):.2f}")
        print(f"EPS: ${fund.get('eps_current', 0):.2f}")
        print(f"ROE: {fund.get('roe', 0):.2f}%")
        print(f"Fundamental Score: {fund.get('fundamental_score', 0):.0f}/100")

        print(f"\n--- TECHNICALS ---")
        print(f"RSI: {tech['rsi']:.2f}")
        print(f"MACD: {tech['macd']:.2f}")
        print(f"SMA 50: ${tech['moving_averages']['SMA_50']:.2f}")

        print(f"\n--- NEXT 5 WEEKDAYS FORECAST ---")
        for forecast in self.short_term_forecast:
            print(f"{forecast['date'].strftime('%Y-%m-%d')} ({forecast['day_name']}): ${forecast['predicted_price']:.2f} ({'+' if forecast['change_from_current'] >= 0 else ''}{forecast['change_from_current']:.2f}%)")

        if self.long_term_projection:
            proj = self.long_term_projection
            print(f"\n--- LONG-TERM PROJECTION (12 MONTHS) ---")
            print(f"Bullish: ${proj['projections']['12_months']['bullish']:.2f}")
            print(f"Base: ${proj['projections']['12_months']['base']:.2f}")
            print(f"Bearish: ${proj['projections']['12_months']['bearish']:.2f}")
            print(f"Analyst Upside: {proj['analyst_targets']['upside_potential']:.2f}%")

        print("\n" + "=" * 70)


def main():
    """Main execution function"""
    print("=" * 70)
    print("  COMPREHENSIVE STOCK ANALYSIS SYSTEM")
    print("  Fundamental & Technical Analysis with Investor Strategies")
    print("=" * 70)
    print()

    # Get stock symbol from user
    if len(sys.argv) > 1:
        symbol = sys.argv[1]
    else:
        symbol = input("Enter stock symbol (e.g., AAPL, MSFT, GOOGL): ").strip().upper()

    if not symbol:
        print("❌ No stock symbol provided. Exiting.")
        return

    # Initialize analyzer
    analyzer = StockAnalyzer(symbol)

    # Run complete analysis
    success = analyzer.run_complete_analysis()

    if not success:
        print("❌ Analysis failed. Please check the stock symbol and try again.")
        return

    # Print summary
    analyzer.print_summary()

    # Generate HTML report
    print("\n📝 Generating HTML report...")
    html_content = analyzer.generate_html_report()

    # Save reports
    analyzer.save_report(html_content)

    print("\n📊 Analysis complete! Check the HTML report for detailed analysis.")
    print("=" * 70)


if __name__ == "__main__":
    main()
