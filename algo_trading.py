import schedule
import time
import logging
from datetime import datetime, timedelta
import os
import pandas as pd
import yfinance as yf
import ta
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler("algo_trading.log"), logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# Functions for multiple operations
def fetch_data(ticker, period="1y", interval="1d"):
    logger.info(f"Fetching data for {ticker}")
    df = yf.download(ticker, period=period, interval=interval, auto_adjust=True)
    df.reset_index(inplace=True)
    return df


def add_indicators(df):
    close_series = df['Close'].squeeze()
    df['RSI'] = ta.momentum.RSIIndicator(close=close_series).rsi()
    df['20DMA'] = close_series.rolling(20).mean()
    df['50DMA'] = close_series.rolling(50).mean()
    macd = ta.trend.MACD(close_series)
    df['MACD'] = macd.macd()
    df['MACD_Signal'] = macd.macd_signal()
    df.dropna(inplace=True)
    return df

def filter_last_six_months(df):
    last_date = df['Date'].max()
    six_months_ago = last_date - timedelta(days=182)
    df_6m = df[df['Date'] >= six_months_ago].copy()
    df_6m.reset_index(drop=True, inplace=True)
    return df_6m

def generate_signals(df):
    df['Signal'] = 0
    df.loc[(df['RSI'] < 30) & (df['20DMA'] > df['50DMA']), 'Signal'] = 1
    df.loc[(df['RSI'] > 70) & (df['20DMA'] < df['50DMA']), 'Signal'] = -1
    return df

def backtest_with_tradelog(df):
    position = 0
    buy_price = None
    buy_date = None
    trades = []

    for i in range(len(df)):
        sig = df['Signal'].iloc[i]
        date = df['Date'].iloc[i] if 'Date' in df.columns else df.index[i]
        price = df['Close'].iloc[i]

        if sig == 1 and position == 0:
            position = 1
            buy_price = price
            buy_date = date
        elif sig == -1 and position == 1:
            pnl = price - buy_price
            trades.append({
                "entry_date": str(buy_date.date() if hasattr(buy_date, 'date') else buy_date),
                "entry_price": float(buy_price),
                "exit_date": str(date.date() if hasattr(date, 'date') else date),
                "exit_price": float(price),
                "pnl": float(pnl)
            })
            position = 0
            buy_price = None
            buy_date = None

    total_profit = sum(t['pnl'] for t in trades)
    win_count = len([t for t in trades if t['pnl'] > 0])
    win_ratio = win_count / len(trades) if trades else 0
    return trades, total_profit, win_ratio
# ML model
def train_best_model(df):
    df['Target'] = (df['Close'].shift(-1) > df['Close']).astype(int)
    df['Volume'] = df['Volume'].astype(float)
    features = ['RSI', 'MACD', '20DMA', '50DMA', 'Volume']
    df.dropna(inplace=True)

    X = df[features]
    y = df['Target']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )

    lr = LogisticRegression(max_iter=1000)
    lr.fit(X_train, y_train)
    lr_acc = accuracy_score(y_test, lr.predict(X_test))

    dt = DecisionTreeClassifier(random_state=42)
    dt.fit(X_train, y_train)
    dt_acc = accuracy_score(y_test, dt.predict(X_test))

    if lr_acc >= dt_acc:
        return lr_acc, "Logistic Regression"
    else:
        return dt_acc, "Decision Tree"


# Logging data to csv files
def log_to_csv(ticker, trades, total_profit, win_ratio, ml_accuracy, model_name):
    os.makedirs("output", exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    trade_file = "output/TradeLog.csv"
    trade_df = pd.DataFrame(trades)
    if not trade_df.empty:
        trade_df.insert(0, "Ticker", ticker)
        trade_df.insert(1, "Model", model_name)
        trade_df["Last Updated"] = timestamp
        trade_df = trade_df[[
            "Ticker", "Model", "Last Updated",
            "entry_date", "entry_price", "exit_date", "exit_price", "pnl"
        ]]
        if not os.path.exists(trade_file):
            trade_df.to_csv(trade_file, index=False)
        else:
            trade_df.to_csv(trade_file, mode='a', header=False, index=False)

    summary_file = "output/Summary.csv"
    summary_row = pd.DataFrame([[ticker, model_name, total_profit, ml_accuracy, timestamp]],
                               columns=["Ticker", "Model", "Total Profit", "ML Accuracy", "Last Updated"])
    if not os.path.exists(summary_file):
        summary_row.to_csv(summary_file, index=False)
    else:
        summary_row.to_csv(summary_file, mode='a', header=False, index=False)

    win_file = "output/WinRatio.csv"
    win_row = pd.DataFrame([[ticker, model_name, win_ratio, timestamp]],
                           columns=["Ticker", "Model", "Win Ratio", "Last Updated"])
    if not os.path.exists(win_file):
        win_row.to_csv(win_file, index=False)
    else:
        win_row.to_csv(win_file, mode='a', header=False, index=False)

def run():
    tickers = [
        "ADANIPORTS.NS", "ASIANPAINT.NS", "AXISBANK.NS", "BAJAJ-AUTO.NS", "BAJFINANCE.NS",
        "BAJAJFINSV.NS", "BPCL.NS", "BHARTIARTL.NS", "BRITANNIA.NS", "CIPLA.NS",
        "COALINDIA.NS", "DIVISLAB.NS", "DRREDDY.NS", "EICHERMOT.NS", "GRASIM.NS",
        "HCLTECH.NS", "HDFCBANK.NS", "HDFCLIFE.NS", "HEROMOTOCO.NS", "HINDALCO.NS",
        "HINDUNILVR.NS", "ICICIBANK.NS", "ITC.NS", "INDUSINDBK.NS", "INFY.NS",
        "JSWSTEEL.NS", "KOTAKBANK.NS", "LT.NS", "M&M.NS", "MARUTI.NS",
        "NTPC.NS", "NESTLEIND.NS", "ONGC.NS", "POWERGRID.NS", "RELIANCE.NS",
        "SBILIFE.NS", "SBIN.NS", "SUNPHARMA.NS", "TCS.NS", "TATACONSUM.NS",
        "TATAMOTORS.NS", "TATASTEEL.NS", "TECHM.NS", "TITAN.NS", "ULTRACEMCO.NS",
        "UPL.NS", "WIPRO.NS", "LICHSGFIN.NS", "TATAELXSI.NS", "ADANIGREEN.NS"
    ]

    for ticker in tickers:
        df = fetch_data(ticker)
        df = add_indicators(df)
        df_6m = filter_last_six_months(df)
        df_6m = generate_signals(df_6m)
        trades, total_profit, win_ratio = backtest_with_tradelog(df_6m)
        accuracy, model_name = train_best_model(df_6m)
        logger.info(f"{ticker} | Model: {model_name} | Profit: {total_profit:.2f} | Win Ratio: {win_ratio:.2%} | ML Accuracy: {accuracy:.2%}")
        log_to_csv(ticker, trades, total_profit, win_ratio, accuracy, model_name)

# ---- Scheduling ----
# schedule.every().day.at("16:30").do(run)

if __name__ == "__main__":
    run()
    # logger.info("Scheduled trading bot started. Waiting for 16:30 each day...")
    # while True:
    #     schedule.run_pending()
    #     time.sleep(60)  # check every 60 sec
