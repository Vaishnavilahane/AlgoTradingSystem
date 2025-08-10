import yfinance as yf
import pandas as pd
import ta
import matplotlib.pyplot as plt


# =============================
# Fetch + Indicators
# =============================
def fetch_and_prepare(ticker):
    df = yf.download(ticker, period="6mo", interval="1d")

    # Ensure 'Close' is a Series, not a DataFrame
    close_series = df['Close'].squeeze()

    df['RSI'] = ta.momentum.RSIIndicator(close=close_series).rsi()
    df['20DMA'] = close_series.rolling(20).mean()
    df['50DMA'] = close_series.rolling(50).mean()
    macd = ta.trend.MACD(close_series)
    df['MACD'] = macd.macd()
    df['MACD_Signal'] = macd.macd_signal()

    df.dropna(inplace=True)

    # Generate Buy/Sell signals
    df['Signal'] = 0
    df.loc[(df['RSI'] < 30) & (df['20DMA'] > df['50DMA']), 'Signal'] = 1
    df.loc[(df['RSI'] > 70) & (df['20DMA'] < df['50DMA']), 'Signal'] = -1
    return df


# =============================
# Visualization
# =============================
def plot_indicators(df, ticker):
    fig, axs = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

    # Price with Moving Averages
    axs[0].plot(df.index, df['Close'], label="Close Price", color='blue')
    axs[0].plot(df.index, df['20DMA'], label="20DMA", color='orange')
    axs[0].plot(df.index, df['50DMA'], label="50DMA", color='magenta')
    axs[0].scatter(df.index[df['Signal'] == 1], df['Close'][df['Signal'] == 1], label='Buy Signal', marker='^', color='green')
    axs[0].scatter(df.index[df['Signal'] == -1], df['Close'][df['Signal'] == -1], label='Sell Signal', marker='v', color='red')
    axs[0].set_title(f"{ticker} Price & Moving Averages")
    axs[0].legend()

    # RSI
    axs[1].plot(df.index, df['RSI'], label='RSI', color='purple')
    axs[1].axhline(30, color='green', linestyle='--')
    axs[1].axhline(70, color='red', linestyle='--')
    axs[1].set_title("Relative Strength Index")
    axs[1].legend()

    # MACD
    axs[2].plot(df.index, df['MACD'], label='MACD', color='blue')
    axs[2].plot(df.index, df['MACD_Signal'], label='Signal Line', color='orange')
    axs[2].set_title("MACD")
    axs[2].legend()

    plt.tight_layout()
    plt.show()


# =============================
# Run
# =============================
if __name__ == "__main__":
    ticker = "TCS.NS"  # Change to study another stock
    df = fetch_and_prepare(ticker)
    plot_indicators(df, ticker)
    
