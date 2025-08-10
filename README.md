# Algo Trading System with ML & Automation

## 📌 Overview

This project is an **automated stock market analysis pipeline** that:

- Fetches historical stock data from Yahoo Finance.
- Calculates popular technical indicators (RSI, MACD, Moving Averages).
- Generates buy/sell signals based on strategy rules.
- Backtests trades to evaluate performance.
- Applies machine learning to select the most accurate prediction model.
- Logs results into structured CSV reports for further analysis.

It supports **multiple tickers** and is built for **scalable, repeatable analysis**.

---

## 🚀 Features

- **Data Fetching** – Retrieves historical market data for NSE-listed stocks.
- **Technical Indicators** – Calculates RSI, MACD, MACD Signal, 20-day & 50-day Moving Averages.
- **Signal Generation** – Issues BUY/SELL decisions based on indicator thresholds.
- **Backtesting** – Simulates trades over the last 6 months and calculates:
  - Total Profit
  - Win Ratio
- **Machine Learning Integration** – Compares Logistic Regression and Decision Tree to select the better-performing model.
- **Logging System** – Saves trade logs, performance summaries, and win ratios into CSV files.
- **Multi-Stock Support** – Runs analysis on a predefined list of tickers.

---

## 📂 Project Structure

```
project/
│── algo_trading.py              # Main execution script
│── algo_trading.log     # Log file
│── requirements.txt     # requirements file
│── visualize_data.py    # Data Visualization (Optional)
│── output/
│    ├── TradeLog.csv    # Detailed trade records
│    ├── Summary.csv     # Profit & ML accuracy summary
│    ├── WinRatio.csv    # Win ratio summary
│── README.md            # Project documentation
```

---

## ⚙️ Installation

```bash
pip install -r requirements.txt
```

---

## ▶️ Usage

Run the project with:

```bash
python algo_trading.py
```

The script will:

1. Fetch stock data.
2. Add indicators.
3. Generate signals.
4. Backtest performance.
5. Train & compare ML models.
6. Save results to `/output` folder.

---

## 📊 Output Files

- **TradeLog.csv** – Detailed trade entries & exits.
- **Summary.csv** – Profit and model accuracy summary per stock.
- **WinRatio.csv** – Win ratio summary per stock.

---

## 📈 Strategy Rules

- **Buy Signal:**
  - RSI < 30 (oversold)
  - 20DMA > 50DMA (short-term trend bullish)
- **Sell Signal:**
  - RSI > 70 (overbought)
  - 20DMA < 50DMA (short-term trend bearish)

---

## 🤖 Machine Learning Models

- **Logistic Regression**
- **Decision Tree Classifier**
- The script selects the model with the **highest accuracy**.

---
## 🤖 Automation & Scheduling

The system supports **automated daily execution** using Python’s `schedule` library. Once started, the script runs continuously and automatically triggers the trading analysis at **16:30** (4:30 PM) local time every day.

- Enables **hands-free daily updates** without manual intervention.
- Launch the script once; it manages daily runs automatically.
- Checks every 60 seconds for the scheduled time.


## 🛠 Difficulty Level

**Intermediate → Advanced**

- Involves data handling, technical analysis, backtesting, and ML integration.

---

## 📜 License

This project is licensed under the MIT License.

---


