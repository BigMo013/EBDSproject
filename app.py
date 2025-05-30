import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.model_selection import TimeSeriesSplit
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from ta.trend import SMAIndicator, EMAIndicator, MACD, ADXIndicator
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.volatility import BollingerBands, AverageTrueRange
from ta.volume import OnBalanceVolumeIndicator
import datetime
from statsmodels.tsa.arima.model import ARIMA
import requests
from datetime import timedelta

# Page configuration
st.set_page_config(page_title="EBDS project", layout="wide", page_icon="📊")

# Custom CSS with red theme
st.markdown("""
    <style>
        :root {
            --primary: #D32F2F;
            --secondary: #F44336;
            --accent: #FF5252;
            --background: #121212;
            --card: #1E1E1E;
            --text: #FFFFFF;
            --success: #27AE60;
            --danger: #E74C3C;
            --gold: #FFD700;
        }
        body {
            background-color: var(--background);
            color: var(--text);
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        }
        .stButton>button {
            background-color: var(--primary);
            color: white;
            font-weight: bold;
            padding: 10px 24px;
            border-radius: 8px;
            transition: all 0.3s;
            border: none;
        }
        .stButton>button:hover {
            background-color: var(--secondary);
            transform: scale(1.05);
            box-shadow: 0 4px 12px rgba(0,0,0,0.2);
        }
        .stSelectbox, .stTextInput, .stDateInput, .stSlider {
            background-color: var(--card);
            border-radius: 8px;
            padding: 8px;
            border: 1px solid #333;
        }
        .stAlert {
            border-radius: 8px;
        }
        .metric-card {
            background-color: var(--card);
            border-radius: 12px;
            padding: 20px;
            box-shadow: 0 4px 8px rgba(0,0,0,0.1);
            margin-bottom: 20px;
            border-left: 4px solid var(--primary);
        }
        .tabs {
            display: flex;
            margin-bottom: 20px;
            border-bottom: 1px solid #444;
        }
        .tab {
            padding: 10px 20px;
            cursor: pointer;
            border-radius: 8px 8px 0 0;
            margin-right: 5px;
            background-color: #2C3E50;
            font-size: 16px;
            transition: all 0.3s;
        }
        .tab:hover {
            background-color: #34495E;
        }
        .tab.active {
            background-color: var(--primary);
            font-weight: bold;
        }
        .feature-grid {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
            gap: 15px;
            margin-top: 20px;
        }
        .feature-card {
            background-color: var(--card);
            border-radius: 8px;
            padding: 15px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.05);
            border-left: 3px solid var(--primary);
        }
        .gold {
            color: var(--gold);
        }
        .positive {
            color: var(--success);
        }
        .negative {
            color: var(--danger);
        }
        .stTabs [role="tablist"] button {
            background-color: #2C3E50;
            color: white;
        }
        .stTabs [role="tablist"] button[aria-selected="true"] {
            background-color: var(--primary);
            font-weight: bold;
        }
        .red-badge {
            background-color: var(--primary);
            color: white;
            padding: 3px 8px;
            border-radius: 12px;
            font-size: 12px;
            font-weight: bold;
        }
        .header-gradient {
            background: linear-gradient(90deg, var(--primary), #B71C1C);
            padding: 15px;
            border-radius: 8px;
            margin-bottom: 20px;
        }
        .dividend-bar {
            height: 10px;
            background-color: var(--primary);
            border-radius: 5px;
            margin-top: 5px;
        }
        .stock-card {
            background-color: var(--card);
            border-radius: 8px;
            padding: 15px;
            margin-bottom: 15px;
            border-left: 3px solid var(--primary);
            transition: transform 0.3s;
        }
        .stock-card:hover {
            transform: translateY(-5px);
            box-shadow: 0 8px 16px rgba(0,0,0,0.2);
        }
        .screener-grid {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(150px, 1fr));
            gap: 10px;
            margin-bottom: 20px;
        }
        .heatmap-container {
            display: grid;
            grid-template-columns: repeat(7, 1fr);
            gap: 5px;
            margin-top: 20px;
        }
        .heatmap-day {
            background-color: var(--card);
            border-radius: 4px;
            padding: 10px;
            text-align: center;
        }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if 'current_tab' not in st.session_state:
    st.session_state.current_tab = "dashboard"
if 'data' not in st.session_state:
    st.session_state.data = None
if 'model_performance' not in st.session_state:
    st.session_state.model_performance = {}
if 'portfolio_data' not in st.session_state:
    st.session_state.portfolio_data = None

# Function to create navigation tabs
def create_tabs():
    tabs = ["dashboard", "portfolio", "analysis", "backtest", "sentiment", "screener"]
    tab_names = ["📊 Dashboard", "📦 Portfolio", "🔍 Technical", "🧪 Backtesting", "📰 Sentiment", "🔎 Stock Screener"]
    
    # Create columns for tabs
    cols = st.columns(len(tabs))
    for i, tab in enumerate(tabs):
        # Create a button for each tab
        if cols[i].button(tab_names[i], key=f"tab_{tab}"):
            st.session_state.current_tab = tab
        
        # Apply active class styling
        if st.session_state.current_tab == tab:
            st.markdown(f"""
                <style>
                    div[data-testid="stButton"] > button[kind="secondary"][data-testid="baseButton-secondary"][aria-label="tab_{tab}"] {{
                        background-color: var(--primary) !important;
                        font-weight: bold !important;
                    }}
                </style>
            """, unsafe_allow_html=True)

# Enhanced technical indicators
def add_technical_indicators(df):
    df = df.copy()
    if "Date" in df.columns:
        df.sort_values(by="Date", inplace=True)
    df.reset_index(drop=True, inplace=True)
    
    # Trend Indicators
    df["SMA_10"] = SMAIndicator(close=df["Close"], window=10).sma_indicator()
    df["SMA_50"] = SMAIndicator(close=df["Close"], window=50).sma_indicator()
    df["EMA_20"] = EMAIndicator(close=df["Close"], window=20).ema_indicator()
    macd = MACD(close=df["Close"])
    df["MACD"] = macd.macd()
    df["MACD_signal"] = macd.macd_signal()
    df["MACD_hist"] = macd.macd_diff()
    df["ADX"] = ADXIndicator(high=df["High"], low=df["Low"], close=df["Close"], window=14).adx()
    
    # Momentum Indicators
    df["RSI"] = RSIIndicator(close=df["Close"], window=14).rsi()
    stoch = StochasticOscillator(high=df["High"], low=df["Low"], close=df["Close"], window=14, smooth_window=3)
    df["Stoch_%K"] = stoch.stoch()
    df["Stoch_%D"] = stoch.stoch_signal()
    
    # Volatility Indicators
    bb = BollingerBands(close=df["Close"], window=20, window_dev=2)
    df["BB_Upper"] = bb.bollinger_hband()
    df["BB_Lower"] = bb.bollinger_lband()
    df["ATR"] = AverageTrueRange(high=df["High"], low=df["Low"], close=df["Close"], window=14).average_true_range()
    
    # Volume Indicators
    df["OBV"] = OnBalanceVolumeIndicator(close=df["Close"], volume=df["Volume"]).on_balance_volume()
    df["Volume_SMA"] = df["Volume"].rolling(window=10).mean()
    
    # Price Transformations
    df["Returns"] = df["Close"].pct_change()
    df["Log_Returns"] = np.log(df["Close"] / df["Close"].shift(1))
    
    return df.dropna()

# ARIMA Model
def create_arima_model(train_data, order=(5,1,0)):
    model = ARIMA(train_data, order=order)
    model_fit = model.fit()
    return model_fit

# Sentiment analysis placeholder
def analyze_sentiment(ticker):
    # Placeholder implementation
    sentiment_score = np.random.uniform(-1, 1)
    sentiment = "Bullish 🐂" if sentiment_score > 0.1 else "Bearish 🐻" if sentiment_score < -0.1 else "Neutral 😐"
    return sentiment_score, sentiment

# Backtesting function
def backtest_strategy(df, predictions):
    capital = 10000  # Starting capital
    position = 0
    trades = []
    portfolio_values = [capital]
    
    for i in range(1, len(predictions)):
        current_price = df["Close"].iloc[i]
        prev_price = df["Close"].iloc[i-1]
        pred = predictions[i]
        
        # Trading signals
        buy_signal = pred > current_price * 1.01  # Predicted to rise >1%
        sell_signal = pred < current_price * 0.99  # Predicted to fall >1%
        
        # Execute trades
        if buy_signal and position == 0:
            position = capital // current_price
            capital -= position * current_price
            trades.append({"Date": df["Date"].iloc[i], "Action": "BUY", "Price": current_price})
        elif sell_signal and position > 0:
            capital += position * current_price
            trades.append({"Date": df["Date"].iloc[i], "Action": "SELL", "Price": current_price})
            position = 0
        
        # Track portfolio value
        portfolio_values.append(capital + position * current_price)
    
    # Final liquidation
    if position > 0:
        capital += position * df["Close"].iloc[-1]
        portfolio_values[-1] = capital
    
    return trades, portfolio_values

# Portfolio Optimization
def portfolio_optimization(stocks):
    st.subheader("📊 Portfolio Optimization")
    
    with st.spinner("Optimizing portfolio..."):
        # Fetch data for multiple stocks
        data = yf.download(stocks, period="1y")
        if data.empty:
            st.error("Could not download data for the given stocks. Please check the tickers and try again.")
            return
        
        # Check if we have the 'Adj Close' column
        if 'Adj Close' in data.columns:
            portfolio = data['Adj Close']
        elif 'Close' in data.columns:
            portfolio = data['Close']
            st.warning("Using Close price since Adjusted Close is not available")
        else:
            st.error("No price data available for optimization")
            return
            
        returns = portfolio.pct_change().dropna()
        
        # Calculate covariance matrix
        cov_matrix = returns.cov() * 252
        
        # Monte Carlo simulation
        num_portfolios = 10000
        results = np.zeros((3, num_portfolios))
        weights_arr = np.zeros((num_portfolios, len(stocks)))
        
        for i in range(num_portfolios):
            weights = np.random.random(len(stocks))
            weights /= np.sum(weights)
            weights_arr[i] = weights
            
            portfolio_return = np.sum(returns.mean() * weights) * 252
            portfolio_std = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
            
            results[0,i] = portfolio_return
            results[1,i] = portfolio_std
            results[2,i] = portfolio_return / portfolio_std  # Sharpe ratio
        
        # Find optimal portfolios
        max_sharpe_idx = np.argmax(results[2])
        min_vol_idx = np.argmin(results[1])
        max_sharpe_weights = weights_arr[max_sharpe_idx]
        
        # Plot efficient frontier
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=results[1,:], y=results[0,:], 
                            mode='markers', name='Random Portfolios',
                            marker=dict(color=results[2,:], 
                                        colorscale='Viridis',
                                        size=7,
                                        showscale=True,
                                        colorbar=dict(title="Sharpe Ratio"))))
        
        fig.add_trace(go.Scatter(x=[results[1,max_sharpe_idx]], 
                            y=[results[0,max_sharpe_idx]],
                            mode='markers', name='Max Sharpe Ratio',
                            marker=dict(color='red', size=12)))
        
        fig.add_trace(go.Scatter(x=[results[1,min_vol_idx]], 
                            y=[results[0,min_vol_idx]],
                            mode='markers', name='Min Volatility',
                            marker=dict(color='green', size=12)))
        
        fig.update_layout(title='Efficient Frontier',
                        xaxis_title='Volatility',
                        yaxis_title='Returns',
                        template='plotly_dark')
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Show optimal weights
        st.subheader("Optimal Portfolio Allocation")
        
        optimal_weights = {
            stock: weight for stock, weight in zip(stocks, max_sharpe_weights)
        }
        weights_df = pd.DataFrame.from_dict(optimal_weights, orient='index', columns=['Weight'])
        weights_df['Weight'] = weights_df['Weight'].apply(lambda x: f"{x:.2%}")
        
        # Create a nice visualization with colors
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
                 '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
        
        fig_pie = go.Figure(data=[go.Pie(
            labels=weights_df.index,
            values=weights_df['Weight'].str.rstrip('%').astype('float'),
            hole=0.3,
            marker=dict(colors=colors[:len(stocks)]),
            textinfo='label+percent',
            hoverinfo='label+percent'
        )])
        fig_pie.update_layout(
            title='Portfolio Allocation',
            template='plotly_dark',
            height=400
        )
        st.plotly_chart(fig_pie, use_container_width=True)
        
        st.dataframe(weights_df.style.format({"Weight": "{:.2%}"}))

# Monte Carlo Simulation for Risk Analysis
def monte_carlo_simulation(df, days=30, simulations=1000):
    st.subheader("🎲 Monte Carlo Simulation")
    
    with st.spinner("Running simulations..."):
        # Calculate daily returns
        returns = df['Close'].pct_change().dropna()
        
        # Set up simulation parameters
        last_price = df['Close'].iloc[-1]
        simulation_df = pd.DataFrame()
        
        # Run simulations
        for x in range(simulations):
            count = 0
            daily_vol = returns.std()
            
            price_series = [last_price]
            
            for _ in range(days):
                if count == days:
                    break
                price = price_series[count] * (1 + np.random.normal(0, daily_vol))
                price_series.append(price)
                count += 1
            
            simulation_df[x] = price_series
        
        # Plot simulations
        fig = go.Figure()
        for col in simulation_df.columns:
            fig.add_trace(go.Scatter(x=simulation_df.index, y=simulation_df[col],
                                line=dict(width=0.5), opacity=0.5))
        
        # Add mean line
        mean_line = simulation_df.mean(axis=1)
        fig.add_trace(go.Scatter(x=mean_line.index, y=mean_line, 
                                name='Mean', line=dict(color='red', width=2)))
        
        fig.update_layout(title=f"{simulations} Monte Carlo Simulations ({days} Days)",
                        xaxis_title='Days',
                        yaxis_title='Price',
                        template='plotly_dark')
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Calculate Value at Risk (VaR)
        final_prices = simulation_df.iloc[-1]
        initial_investment = 10000  # $10,000 investment
        portfolio_values = initial_investment * (final_prices / last_price)
        
        # Calculate 5% VaR
        var_95 = np.percentile(portfolio_values, 5)
        var_99 = np.percentile(portfolio_values, 1)
        median_val = np.median(portfolio_values)
        
        st.subheader("Value at Risk (VaR) Analysis")
        col1, col2, col3 = st.columns(3)
        col1.metric("95% Confidence VaR", f"${initial_investment - var_95:,.2f}", 
                   delta=f"-{(1 - var_95/initial_investment)*100:.2f}%", delta_color="inverse")
        col2.metric("99% Confidence VaR", f"${initial_investment - var_99:,.2f}", 
                   delta=f"-{(1 - var_99/initial_investment)*100:.2f}%", delta_color="inverse")
        col3.metric("Median Portfolio Value", f"${median_val:,.2f}", 
                   delta=f"+{(median_val/initial_investment-1)*100:.2f}%")

# Earnings Calendar and Analyst Ratings
def earnings_calendar(ticker):
    st.subheader("📅 Earnings Calendar & Analyst Ratings")
    
    try:
        stock = yf.Ticker(ticker)
        earnings = stock.earnings_dates
        
        if earnings is not None and not earnings.empty:
            st.write("Upcoming Earnings Dates:")
            earnings_display = earnings.head(4).copy()
            earnings_display.index = pd.to_datetime(earnings_display.index)
            st.dataframe(earnings_display.style.format({
                'EPS Estimate': '{:.2f}',
                'Reported EPS': '{:.2f}',
                'Surprise(%)': '{:.2f}%'
            }))
        else:
            st.warning("No earnings dates available")
        
        # Analyst recommendations
        rec = stock.recommendations
        if rec is not None and not rec.empty:
            # Get latest recommendations
            rec = rec.sort_index(ascending=False)
            latest_rec = rec.head(10)
            
            st.write("Latest Analyst Recommendations:")
            st.dataframe(latest_rec[['Firm', 'To Grade', 'Action']])
            
            # Recommendation summary
            st.subheader("Recommendation Summary")
            fig = go.Figure(go.Pie(
                labels=rec['To Grade'].value_counts().index,
                values=rec['To Grade'].value_counts().values,
                hole=0.4,
                marker=dict(colors=['#00CC96', '#636EFA', '#EF553B', '#AB63FA'])
            ))
            fig.update_layout(
                title='Recommendation Distribution',
                template='plotly_dark',
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("No analyst recommendations available")
    except Exception as e:
        st.error(f"Could not fetch earnings data: {str(e)}")

# Real-time Market Data Dashboard
def real_time_market_dashboard():
    st.subheader("🌐 Real-time Market Overview")
    
    # Major indices
    indices = {
        "S&P 500": "^GSPC",
        "Dow Jones": "^DJI",
        "NASDAQ": "^IXIC",
        "FTSE 100": "^FTSE",
        "DAX": "^GDAXI",
        "Nikkei 225": "^N225"
    }
    
    data = []
    for name, ticker in indices.items():
        try:
            stock = yf.Ticker(ticker)
            # Get the last two days to compute change
            hist = stock.history(period="2d")
            if not hist.empty and len(hist) >= 2:
                last_close = hist['Close'].iloc[-1]
                prev_close = hist['Close'].iloc[-2]
                change = last_close - prev_close
                pct_change = (change / prev_close) * 100
                data.append({
                    "Index": name,
                    "Price": last_close,
                    "Change": change,
                    "% Change": pct_change
                })
        except:
            # If we can't get data for this index, skip
            continue
    
    if data:
        indices_df = pd.DataFrame(data)
        
        # Format for display
        indices_df['Change Color'] = indices_df['Change'].apply(
            lambda x: 'positive' if x >= 0 else 'negative')
        indices_df['% Change Color'] = indices_df['% Change'].apply(
            lambda x: 'positive' if x >= 0 else 'negative')
        
        # Create a styled table
        st.markdown("""
        <style>
            .positive { color: #27AE60; }
            .negative { color: #E74C3C; }
        </style>
        """, unsafe_allow_html=True)
        
        # Convert to HTML for styling
        indices_html = indices_df.to_html(escape=False, index=False, columns=["Index", "Price", "Change", "% Change"])
        # Add color classes
        for i, row in indices_df.iterrows():
            indices_html = indices_html.replace(f'>{row["Change"]}</td>', f' class="{row["Change Color"]}">{row["Change"]:.2f}</td>')
            indices_html = indices_html.replace(f'>{row["% Change"]}</td>', f' class="{row["% Change Color"]}">{row["% Change"]:.2f}%</td>')
        
        st.markdown(indices_html, unsafe_allow_html=True)
    else:
        st.warning("Could not fetch market indices data")
    
    # Market movers
    st.subheader("📈 Top Market Movers")
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("🔺 Top Gainers")
        gainers_tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "NVDA"]
        gainer_data = []
        for ticker in gainers_tickers:
            try:
                stock = yf.Ticker(ticker)
                hist = stock.history(period="2d")
                if len(hist) >= 2:
                    change = (hist['Close'].iloc[-1] - hist['Close'].iloc[0]) / hist['Close'].iloc[0]
                    gainer_data.append({"Ticker": ticker, "Change": change})
            except:
                continue
        
        if gainer_data:
            gainers_df = pd.DataFrame(gainer_data)
            gainers_df = gainers_df.nlargest(5, 'Change')
            gainers_df['Change'] = gainers_df['Change'].apply(lambda x: f"{x:.2%}")
            # Display with green color
            st.dataframe(gainers_df.set_index('Ticker').style.applymap(lambda x: 'color: #27AE60'))
        else:
            st.warning("Could not fetch gainers data")
    
    with col2:
        st.write("🔻 Top Losers")
        losers_tickers = ["META", "NFLX", "PYPL", "INTC", "CSCO", "DIS"]
        loser_data = []
        for ticker in losers_tickers:
            try:
                stock = yf.Ticker(ticker)
                hist = stock.history(period="2d")
                if len(hist) >= 2:
                    change = (hist['Close'].iloc[-1] - hist['Close'].iloc[0]) / hist['Close'].iloc[0]
                    loser_data.append({"Ticker": ticker, "Change": change})
            except:
                continue
        
        if loser_data:
            losers_df = pd.DataFrame(loser_data)
            losers_df = losers_df.nsmallest(5, 'Change')
            losers_df['Change'] = losers_df['Change'].apply(lambda x: f"{x:.2%}")
            st.dataframe(losers_df.set_index('Ticker').style.applymap(lambda x: 'color: #E74C3C'))
        else:
            st.warning("Could not fetch losers data")

# Financial Health Metrics
def financial_health(ticker):
    st.subheader("💼 Financial Health Analysis")
    
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        
        # Key metrics
        metrics = {
            "Metric": [
                "Market Cap", "P/E Ratio", "PEG Ratio", "EPS", 
                "Debt to Equity", "Current Ratio", "Quick Ratio",
                "Return on Assets", "Return on Equity", "Profit Margins"
            ],
            "Value": [
                info.get('marketCap', 'N/A'),
                info.get('trailingPE', 'N/A'),
                info.get('pegRatio', 'N/A'),
                info.get('trailingEps', 'N/A'),
                info.get('debtToEquity', 'N/A'),
                info.get('currentRatio', 'N/A'),
                info.get('quickRatio', 'N/A'),
                info.get('returnOnAssets', 'N/A'),
                info.get('returnOnEquity', 'N/A'),
                info.get('profitMargins', 'N/A')
            ]
        }
        
        metrics_df = pd.DataFrame(metrics)
        
        # Format large numbers
        if metrics_df.loc[0, 'Value'] != 'N/A':
            metrics_df.loc[0, 'Value'] = f"${metrics_df.loc[0, 'Value']/1e9:,.1f}B"
        
        # Highlight good/bad values
        def highlight_metrics(val):
            if val == 'N/A':
                return 'color: gray'
            try:
                num_val = float(val)
                if 'Ratio' in metrics_df.loc[metrics_df['Value'] == val, 'Metric'].values[0]:
                    if 'Debt' in metrics_df.loc[metrics_df['Value'] == val, 'Metric'].values[0]:
                        return 'color: #E74C3C' if num_val > 1 else 'color: #27AE60'
                    return 'color: #27AE60' if num_val > 0 else 'color: #E74C3C'
                return ''
            except:
                return ''
        
        st.dataframe(metrics_df.style.applymap(highlight_metrics))
        
    except Exception as e:
        st.error(f"Could not fetch financial data: {str(e)}")

# Dividend Analysis
def dividend_analysis(ticker):
    st.subheader("💰 Dividend Analysis")
    
    try:
        stock = yf.Ticker(ticker)
        dividends = stock.dividends
        
        if dividends.empty:
            st.info("This stock does not pay dividends")
            return
            
        # Get dividend history
        st.write("Dividend History (Last 5 Years)")
        dividends_df = dividends.resample('Y').sum().tail(5)
        st.bar_chart(dividends_df)
        
        # Dividend metrics
        current_price = stock.history(period='1d')['Close'].iloc[-1]
        last_dividend = dividends.iloc[-1]
        dividend_yield = (last_dividend * 4) / current_price * 100
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Last Dividend", f"${last_dividend:.2f}")
        col2.metric("Annual Dividend", f"${last_dividend * 4:.2f}")
        col3.metric("Dividend Yield", f"{dividend_yield:.2f}%")
        
        # Dividend sustainability
        payout_ratio = stock.info.get('payoutRatio', None)
        if payout_ratio:
            st.write("Payout Ratio")
            st.progress(min(payout_ratio, 1.0))
            st.caption(f"{payout_ratio*100:.1f}% of earnings paid as dividends")
        
    except Exception as e:
        st.error(f"Could not fetch dividend data: {str(e)}")

# Correlation Matrix
def correlation_matrix(stocks):
    st.subheader("📊 Stock Correlation Matrix")
    
    with st.spinner("Calculating correlations..."):
        # Fetch data
        data = yf.download(stocks, period="1y")['Close']
        if data.empty:
            st.error("Could not download data for the given stocks")
            return
            
        # Calculate correlations
        corr = data.corr()
        
        # Create heatmap
        fig = go.Figure(data=go.Heatmap(
            z=corr.values,
            x=corr.columns,
            y=corr.index,
            colorscale='RdBu',
            zmin=-1,
            zmax=1
        ))
        
        fig.update_layout(
            title='Correlation Matrix',
            xaxis_title='Stocks',
            yaxis_title='Stocks',
            template='plotly_dark'
        )
        
        st.plotly_chart(fig, use_container_width=True)

# Stock Screener
def stock_screener():
    st.subheader("🔎 Stock Screener")
    
    # Screener criteria
    st.write("Set your screening criteria:")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        min_market_cap = st.number_input("Min Market Cap ($B)", min_value=0.0, value=10.0)
    with col2:
        max_pe_ratio = st.number_input("Max P/E Ratio", min_value=0.0, value=30.0)
    with col3:
        min_dividend_yield = st.number_input("Min Dividend Yield (%)", min_value=0.0, value=2.0)
    with col4:
        max_debt_equity = st.number_input("Max Debt/Equity", min_value=0.0, value=1.0)
    
    if st.button("Run Screener", key="run_screener"):
        # Sample stock universe
        stocks = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "JNJ", "JPM", "PG", "V", "MA", "DIS", "NFLX"]
        
        results = []
        with st.spinner("Screening stocks..."):
            for ticker in stocks:
                try:
                    stock = yf.Ticker(ticker)
                    info = stock.info
                    
                    # Get required metrics
                    market_cap = info.get('marketCap', 0) / 1e9  # Convert to billions
                    pe_ratio = info.get('trailingPE', 1000)
                    div_yield = info.get('dividendYield', 0) * 100 if info.get('dividendYield') else 0
                    debt_equity = info.get('debtToEquity', 10)
                    
                    # Apply filters
                    if (market_cap >= min_market_cap and 
                        pe_ratio <= max_pe_ratio and 
                        div_yield >= min_dividend_yield and 
                        debt_equity <= max_debt_equity):
                        
                        results.append({
                            "Ticker": ticker,
                            "Name": info.get('shortName', ticker),
                            "Market Cap ($B)": market_cap,
                            "P/E Ratio": pe_ratio,
                            "Dividend Yield (%)": div_yield,
                            "Debt/Equity": debt_equity
                        })
                except:
                    continue
        
        if results:
            results_df = pd.DataFrame(results)
            st.success(f"Found {len(results)} stocks matching your criteria")
            st.dataframe(results_df)
        else:
            st.warning("No stocks match your criteria")

# Economic Calendar
def economic_calendar():
    st.subheader("📅 Economic Calendar")
    
    # Get next 7 days
    today = datetime.date.today()
    dates = [today + timedelta(days=i) for i in range(7)]
    
    # Sample economic events
    events = [
        {"date": today, "time": "10:00 AM", "event": "Fed Interest Rate Decision", "impact": "High"},
        {"date": today + timedelta(days=1), "time": "8:30 AM", "event": "Unemployment Claims", "impact": "Medium"},
        {"date": today + timedelta(days=2), "time": "10:00 AM", "event": "Consumer Confidence", "impact": "Medium"},
        {"date": today + timedelta(days=3), "time": "8:30 AM", "event": "GDP Growth Rate", "impact": "High"},
        {"date": today + timedelta(days=4), "time": "10:00 AM", "event": "Existing Home Sales", "impact": "Low"},
        {"date": today + timedelta(days=5), "time": "8:30 AM", "event": "Durable Goods Orders", "impact": "Medium"},
    ]
    
    # Display events
    for event in events:
        impact_color = {
            "High": "#E74C3C",
            "Medium": "#F39C12",
            "Low": "#27AE60"
        }.get(event["impact"], "#7F8C8D")
        
        st.markdown(f"""
            <div class="stock-card">
                <div style="display: flex; justify-content: space-between;">
                    <div>
                        <strong>{event['date'].strftime('%A, %b %d')}</strong> | {event['time']}
                        <h4>{event['event']}</h4>
                    </div>
                    <div class="red-badge" style="background-color: {impact_color};">{event['impact']}</div>
                </div>
            </div>
        """, unsafe_allow_html=True)

# Automated Trading Strategy Tester
def strategy_tester(df):
    st.subheader("🤖 Trading Strategy Tester")
    
    strategy = st.selectbox("Select Trading Strategy", 
                          ["Moving Average Crossover", "RSI Divergence", "MACD Signal"])
    
    if strategy == "Moving Average Crossover":
        short_window = st.slider("Short MA Period", 5, 50, 20)
        long_window = st.slider("Long MA Period", 50, 200, 50)
        
        df['Short_MA'] = df['Close'].rolling(short_window).mean()
        df['Long_MA'] = df['Close'].rolling(long_window).mean()
        df['Signal'] = np.where(df['Short_MA'] > df['Long_MA'], 1, 0)
        df['Position'] = df['Signal'].diff()
    
    elif strategy == "RSI Divergence":
        rsi_window = st.slider("RSI Period", 5, 30, 14)
        df['RSI'] = RSIIndicator(close=df['Close'], window=rsi_window).rsi()
        df['Signal'] = np.where(df['RSI'] < 30, 1, np.where(df['RSI'] > 70, -1, 0))
        df['Position'] = df['Signal'].diff()
    
    elif strategy == "MACD Signal":
        fast = st.slider("MACD Fast Period", 5, 20, 12)
        slow = st.slider("MACD Slow Period", 20, 50, 26)
        signal = st.slider("MACD Signal Period", 5, 20, 9)
        
        macd = MACD(close=df['Close'], window_fast=fast, window_slow=slow, window_sign=signal)
        df['MACD'] = macd.macd()
        df['Signal_Line'] = macd.macd_signal()
        df['Signal'] = np.where(df['MACD'] > df['Signal_Line'], 1, -1)
        df['Position'] = df['Signal'].diff()
    
    # Backtest strategy
    capital = 10000
    position = 0
    trades = []
    portfolio = [capital]
    
    for i in range(1, len(df)):
        if df['Position'].iloc[i] == 1:  # Buy signal
            if position == 0:
                position = capital // df['Close'].iloc[i]
                capital -= position * df['Close'].iloc[i]
                trades.append({"Date": df['Date'].iloc[i], "Action": "BUY", "Price": df['Close'].iloc[i]})
        elif df['Position'].iloc[i] == -1:  # Sell signal
            if position > 0:
                capital += position * df['Close'].iloc[i]
                trades.append({"Date": df['Date'].iloc[i], "Action": "SELL", "Price": df['Close'].iloc[i]})
                position = 0
        
        portfolio.append(capital + position * df['Close'].iloc[i])
    
    # Final liquidation
    if position > 0:
        capital += position * df['Close'].iloc[-1]
        portfolio[-1] = capital
    
    # Display results
    st.subheader("Strategy Performance")
    col1, col2 = st.columns(2)
    col1.metric("Initial Capital", f"${10000:,.2f}")
    col2.metric("Final Value", f"${portfolio[-1]:,.2f}", f"{(portfolio[-1]/10000-1)*100:.2f}%")
    
    # Plot portfolio value
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df['Date'], y=portfolio, name='Portfolio Value', line=dict(color='green')))
    fig.update_layout(title='Strategy Performance',
                    xaxis_title='Date',
                    yaxis_title='Value',
                    template='plotly_dark')
    st.plotly_chart(fig, use_container_width=True)
    
    # Show trade history
    if trades:
        st.subheader("Trade History")
        trades_df = pd.DataFrame(trades)
        st.dataframe(trades_df.style.format({"Price": "${:.2f}"}))
    else:
        st.info("No trades executed with this strategy")

# Main app
st.title("🚀 EBDS Project")

# Create navigation tabs
create_tabs()

# Sidebar configuration
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # Data source selection
    data_source = st.radio("Data Source", ["Live Data", "Upload CSV"])
    
    if data_source == "Live Data":
        ticker = st.text_input("Stock Ticker", "AAPL")
        start_date = st.date_input("Start Date", datetime.date.today() - datetime.timedelta(days=365))
        end_date = st.date_input("End Date", datetime.date.today())
        fetch_button = st.button("Fetch Data")
        
        if fetch_button:
            with st.spinner("Fetching data..."):
                stock = yf.Ticker(ticker)
                df = stock.history(start=start_date, end=end_date)[["Open", "High", "Low", "Close", "Volume"]].reset_index()
                df = add_technical_indicators(df)
                st.session_state.data = df
                st.success(f"✅ Data fetched for {ticker}")
    
    else:  # Upload CSV
        uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
        if uploaded_file:
            df = pd.read_csv(uploaded_file)
            if "Date" in df.columns:
                df["Date"] = pd.to_datetime(df["Date"])
            df = add_technical_indicators(df)
            st.session_state.data = df
            st.success("✅ File uploaded successfully")
    
    # Model selection
    st.header("🧠 Prediction Model")
    model_choice = st.selectbox("Select Model", 
                               ["ARIMA", "Random Forest", "Gradient Boosting", "SVM", "Linear Regression", "KNN"])
    
    # Advanced options
    with st.expander("Advanced Options"):
        forecast_days = st.slider("Forecast Days", 1, 30, 7)
        test_size = st.slider("Test Size (%)", 5, 40, 20)
        feature_selection = st.multiselect("Select Features", 
                                         ["SMA_10", "SMA_50", "EMA_20", "RSI", "MACD", 
                                          "Stoch_%K", "BB_Upper", "ATR", "OBV", "Volume_SMA"],
                                         default=["SMA_10", "RSI", "MACD", "Volume_SMA"])

# Dashboard Tab
if st.session_state.current_tab == "dashboard":
    if st.session_state.data is not None:
        df = st.session_state.data
        
        # Real-time Market Dashboard
        st.markdown('<div class="header-gradient"><h2>🌐 Global Market Dashboard</h2></div>', unsafe_allow_html=True)
        real_time_market_dashboard()
        st.divider()
        
        # Display key metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.markdown(f'<div class="metric-card"><h3>Current Price</h3><h2>${df["Close"].iloc[-1]:.2f}</h2></div>', 
                       unsafe_allow_html=True)
        with col2:
            change = (df["Close"].iloc[-1] - df["Close"].iloc[0]) / df["Close"].iloc[0] * 100
            st.markdown(f'<div class="metric-card"><h3>Total Change</h3><h2>{change:.2f}%</h2></div>', 
                       unsafe_allow_html=True)
        with col3:
            st.markdown(f'<div class="metric-card"><h3>Volatility (ATR)</h3><h2>{df["ATR"].iloc[-1]:.2f}</h2></div>', 
                       unsafe_allow_html=True)
        with col4:
            sentiment_score, sentiment = analyze_sentiment(ticker if data_source == "Live Data" else "Stock")
            st.markdown(f'<div class="metric-card"><h3>Market Sentiment</h3><h2>{sentiment}</h2></div>', 
                       unsafe_allow_html=True)
        
        # Financial Health Metrics
        if data_source == "Live Data":
            financial_health(ticker)
            st.divider()
        
        # Dividend Analysis
        if data_source == "Live Data":
            dividend_analysis(ticker)
            st.divider()
        
        # Interactive price chart
        fig = go.Figure()
        fig.add_trace(go.Candlestick(x=df['Date'],
                                    open=df['Open'],
                                    high=df['High'],
                                    low=df['Low'],
                                    close=df['Close'],
                                    name='Price'))
        fig.add_trace(go.Scatter(x=df['Date'], y=df['SMA_50'], name='SMA 50', line=dict(color='#FF5252', width=2)))
        fig.add_trace(go.Scatter(x=df['Date'], y=df['BB_Upper'], name='BB Upper', line=dict(color='#FF5252', width=1, dash='dash')))
        fig.add_trace(go.Scatter(x=df['Date'], y=df['BB_Lower'], name='BB Lower', line=dict(color='#27AE60', width=1, dash='dash')))
        fig.update_layout(title='Price Chart with Indicators', 
                         xaxis_title='Date', 
                         yaxis_title='Price',
                         template='plotly_dark',
                         height=500)
        st.plotly_chart(fig, use_container_width=True)
        
        # Earnings Calendar
        if data_source == "Live Data":
            earnings_calendar(ticker)
            st.divider()
        
        # Economic Calendar
        economic_calendar()
        st.divider()
        
        # Model training and prediction
        if st.button("Train Model & Predict"):
            with st.spinner("Training model..."):
                # Prepare data
                features = [f for f in feature_selection if f in df.columns]
                X = df[features]
                y = df["Close"]
                
                # Time-based split
                test_size_val = int(len(X) * (test_size/100))
                X_train, X_test = X[:-test_size_val], X[-test_size_val:]
                y_train, y_test = y[:-test_size_val], y[-test_size_val:]
                
                # Model selection and training
                if model_choice == "Random Forest":
                    model = RandomForestRegressor(n_estimators=200, random_state=42)
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)
                elif model_choice == "Linear Regression":
                    model = LinearRegression()
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)
                elif model_choice == "Gradient Boosting":
                    model = GradientBoostingRegressor(n_estimators=200, learning_rate=0.1)
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)
                elif model_choice == "SVM":
                    model = SVR(kernel='rbf', C=100, gamma=0.1)
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)
                elif model_choice == "KNN":
                    model = KNeighborsRegressor(n_neighbors=5)
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)
                elif model_choice == "ARIMA":
                    # ARIMA works with univariate data
                    model = create_arima_model(y_train)
                    y_pred = model.forecast(steps=len(y_test))
                
                # Calculate metrics
                mse = mean_squared_error(y_test, y_pred)
                mae = mean_absolute_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)
                
                # Store performance
                st.session_state.model_performance = {
                    "mse": mse,
                    "mae": mae,
                    "r2": r2,
                    "y_test": y_test,
                    "y_pred": y_pred
                }
                
                # Plot results
                fig_pred = go.Figure()
                fig_pred.add_trace(go.Scatter(x=y_test.index, y=y_test, name='Actual', line=dict(color='#3498DB')))
                fig_pred.add_trace(go.Scatter(x=y_test.index, y=y_pred, name='Predicted', line=dict(color='#FF5252', dash='dash')))
                fig_pred.update_layout(title='Actual vs Predicted Prices',
                                     xaxis_title='Days',
                                     yaxis_title='Price',
                                     template='plotly_dark',
                                     height=500)
                st.plotly_chart(fig_pred, use_container_width=True)
                
                # Show metrics
                st.subheader("Model Performance")
                col1, col2, col3 = st.columns(3)
                col1.metric("MSE", f"{mse:.4f}")
                col2.metric("MAE", f"{mae:.4f}")
                col3.metric("R² Score", f"{r2:.4f}")
        
        # Risk Analysis
        st.divider()
        st.subheader("🎲 Risk Analysis")
        col1, col2 = st.columns(2)
        with col1:
            days = st.slider("Simulation Days", 10, 365, 30)
        with col2:
            simulations = st.slider("Number of Simulations", 100, 5000, 1000)
        
        if st.button("Run Monte Carlo Simulation"):
            monte_carlo_simulation(df, days, simulations)
            
    else:
        st.warning("⚠️ Please load data in the sidebar to use the Dashboard")

# Portfolio Tab
if st.session_state.current_tab == "portfolio":
    st.markdown('<div class="header-gradient"><h2>📦 Portfolio Management</h2></div>', unsafe_allow_html=True)
    
    st.info("Enter up to 6 stock tickers separated by commas (e.g., AAPL,MSFT,GOOGL)")
    portfolio_stocks = st.text_input("Stock Tickers", "AAPL,MSFT,GOOGL,AMZN")
    
    if st.button("Optimize Portfolio"):
        stocks = [s.strip().upper() for s in portfolio_stocks.split(",") if s.strip()]
        if len(stocks) < 2:
            st.error("Please enter at least 2 stock tickers")
        elif len(stocks) > 6:
            st.error("Maximum 6 stocks allowed for optimization")
        else:
            portfolio_optimization(stocks)
    
    st.divider()
    
    # Correlation Matrix
    if portfolio_stocks:
        stocks = [s.strip().upper() for s in portfolio_stocks.split(",") if s.strip()]
        if len(stocks) > 1:
            correlation_matrix(stocks)
        else:
            st.warning("Enter at least 2 stocks to see correlations")

# Technical Analysis Tab
if st.session_state.current_tab == "analysis":
    if st.session_state.data is not None:
        df = st.session_state.data
        st.markdown('<div class="header-gradient"><h2>🔍 Technical Analysis</h2></div>', unsafe_allow_html=True)
        
        # Indicator selection
        indicators = st.multiselect("Select Indicators to Display", 
                                   ["RSI", "MACD", "Stochastic", "ADX", "ATR", "OBV"],
                                   default=["RSI", "MACD"])
        
        # Create subplots based on selection
        num_indicators = len(indicators)
        fig = make_subplots(rows=num_indicators+1, cols=1, shared_xaxes=True, 
                          vertical_spacing=0.05, row_heights=[0.6] + [0.4/num_indicators]*num_indicators)
        
        # Price chart
        fig.add_trace(go.Candlestick(x=df['Date'], open=df['Open'], high=df['High'], 
                                   low=df['Low'], close=df['Close'], name="Price"), 
                     row=1, col=1)
        
        # Add selected indicators
        for i, indicator in enumerate(indicators, start=2):
            if indicator == "RSI":
                fig.add_trace(go.Scatter(x=df['Date'], y=df['RSI'], name='RSI', line=dict(color='#FF5252')), row=i, col=1)
                fig.add_hrect(y0=70, y1=100, line_width=0, fillcolor="red", opacity=0.1, row=i, col=1)
                fig.add_hrect(y0=0, y1=30, line_width=0, fillcolor="green", opacity=0.1, row=i, col=1)
            elif indicator == "MACD":
                fig.add_trace(go.Scatter(x=df['Date'], y=df['MACD'], name='MACD', line=dict(color='#FF5252')), row=i, col=1)
                fig.add_trace(go.Scatter(x=df['Date'], y=df['MACD_signal'], name='Signal', line=dict(color='#3498DB')), row=i, col=1)
                colors = np.where(df['MACD_hist'] < 0, 'red', 'green')
                fig.add_trace(go.Bar(x=df['Date'], y=df['MACD_hist'], name='Histogram', 
                                   marker_color=colors), row=i, col=1)
            elif indicator == "Stochastic":
                fig.add_trace(go.Scatter(x=df['Date'], y=df['Stoch_%K'], name='%K', line=dict(color='#FF5252')), row=i, col=1)
                fig.add_trace(go.Scatter(x=df['Date'], y=df['Stoch_%D'], name='%D', line=dict(color='#3498DB')), row=i, col=1)
                fig.add_hrect(y0=80, y1=100, line_width=0, fillcolor="red", opacity=0.1, row=i, col=1)
                fig.add_hrect(y0=0, y1=20, line_width=0, fillcolor="green", opacity=0.1, row=i, col=1)
            elif indicator == "ADX":
                fig.add_trace(go.Scatter(x=df['Date'], y=df['ADX'], name='ADX', line=dict(color='#FF5252')), row=i, col=1)
                fig.add_hrect(y0=25, y1=100, line_width=0, fillcolor="purple", opacity=0.1, row=i, col=1)
            elif indicator == "ATR":
                fig.add_trace(go.Scatter(x=df['Date'], y=df['ATR'], name='ATR', line=dict(color='#FF5252')), row=i, col=1)
            elif indicator == "OBV":
                fig.add_trace(go.Scatter(x=df['Date'], y=df['OBV'], name='OBV', line=dict(color='#FF5252')), row=i, col=1)
        
        fig.update_layout(title='Technical Analysis Dashboard', 
                         height=800, 
                         showlegend=True,
                         template='plotly_dark')
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("⚠️ Please load data in the sidebar to use Technical Analysis")

# Backtesting Tab
if st.session_state.current_tab == "backtest":
    if st.session_state.data is not None:
        df = st.session_state.data
        st.markdown('<div class="header-gradient"><h2>🧪 Backtesting Lab</h2></div>', unsafe_allow_html=True)
        
        tab1, tab2 = st.tabs(["Strategy Tester", "Model Backtest"])
        
        with tab1:
            strategy_tester(df)
        
        with tab2:
            if st.session_state.model_performance:
                y_test = st.session_state.model_performance["y_test"]
                y_pred = st.session_state.model_performance["y_pred"]
                
                st.header("🧪 Model Prediction Backtesting")
                
                # Backtest strategy
                backtest_df = df.iloc[-len(y_test):].copy()
                backtest_df["Predicted"] = y_pred
                trades, portfolio_values = backtest_strategy(backtest_df, y_pred)
                
                # Display backtest results
                col1, col2, col3 = st.columns(3)
                initial_value = 10000
                final_value = portfolio_values[-1]
                returns = (final_value - initial_value) / initial_value * 100
                buy_hold = (backtest_df["Close"].iloc[-1] - backtest_df["Close"].iloc[0]) / backtest_df["Close"].iloc[0] * 100
                
                col1.metric("Initial Capital", f"${initial_value:,.2f}")
                col2.metric("Final Value", f"${final_value:,.2f}", f"{returns:.2f}%")
                col3.metric("Buy & Hold Return", f"{buy_hold:.2f}%")
                
                # Portfolio performance chart
                fig_portfolio = go.Figure()
                fig_portfolio.add_trace(go.Scatter(x=backtest_df["Date"], y=portfolio_values, name='Strategy', line=dict(color='#FF5252')))
                fig_portfolio.add_trace(go.Scatter(x=backtest_df["Date"], 
                                                 y=initial_value * (backtest_df["Close"] / backtest_df["Close"].iloc[0]),
                                                 name='Buy & Hold', line=dict(color='#3498DB', dash='dot')))
                fig_portfolio.update_layout(title='Portfolio Value Over Time',
                                          xaxis_title='Date',
                                          yaxis_title='Portfolio Value',
                                          template='plotly_dark',
                                          height=500)
                st.plotly_chart(fig_portfolio, use_container_width=True)
                
                # Trade history
                if trades:
                    st.subheader("Trade History")
                    trades_df = pd.DataFrame(trades)
                    st.dataframe(trades_df.style.format({"Price": "${:.2f}"}))
            else:
                st.warning("⚠️ Please train a model on the Dashboard before backtesting")
    else:
        st.warning("⚠️ Please load data in the sidebar to use Backtesting")

# Sentiment Analysis Tab
if st.session_state.current_tab == "sentiment":
    st.markdown('<div class="header-gradient"><h2>📰 Market Sentiment</h2></div>', unsafe_allow_html=True)
    
    # Sentiment analysis (placeholder for actual API integration)
    if st.button("Analyze Market Sentiment"):
        with st.spinner("Gathering market sentiment..."):
            # In a real implementation, this would connect to a news API
            sentiment_score, sentiment = analyze_sentiment(ticker if data_source == "Live Data" else "Stock")
            
            # Sentiment gauge
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=sentiment_score,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': 'Market Sentiment'},
                gauge={
                    'axis': {'range': [-1, 1]},
                    'bar': {'color': "darkred"},
                    'steps': [
                        {'range': [-1, -0.5], 'color': "red"},
                        {'range': [-0.5, 0.5], 'color': "gray"},
                        {'range': [0.5, 1], 'color': "green"}
                    ]
            }))
            fig_gauge.update_layout(height=400)
            st.plotly_chart(fig_gauge, use_container_width=True)
            
            # News headlines (simulated)
            st.subheader("Recent Market News")
            news = [
                {"title": "Tech Stocks Rally on Strong Earnings Reports", "sentiment": 0.8},
                {"title": "Fed Hints at Interest Rate Hikes, Markets Nervous", "sentiment": -0.6},
                {"title": "New Regulations Could Impact Tech Sector Growth", "sentiment": -0.4},
                {"title": "Company Announces Breakthrough Product Launch", "sentiment": 0.9},
                {"title": "Global Supply Chain Issues Persist, Stocks Volatile", "sentiment": -0.3}
            ]
            
            for item in news:
                emoji = "📈" if item["sentiment"] > 0.3 else "📉" if item["sentiment"] < -0.3 else "📰"
                st.markdown(f"<div class='feature-card'>{emoji} {item['title']}</div>", unsafe_allow_html=True)
    else:
        st.info("ℹ️ Click the button above to analyze market sentiment")

# Stock Screener Tab
if st.session_state.current_tab == "screener":
    st.markdown('<div class="header-gradient"><h2>🔎 Stock Screener</h2></div>', unsafe_allow_html=True)
    stock_screener()