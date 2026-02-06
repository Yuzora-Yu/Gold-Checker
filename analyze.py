import yfinance as yf
import pandas as pd
import json

# 設定
SYMBOL = "GC=F" # 金先物 (XAU/USDに相当)

def analyze():
    # 1. 価格データ取得
    gold = yf.Ticker(SYMBOL)
    df = gold.history(period="5d", interval="1h")
    
    # 2. テクニカル計算 (RSI)
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    latest_price = df['Close'].iloc[-1]
    latest_rsi = df['RSI'].iloc[-1]
    
    # 3. 結果をJSONに保存 (GitHub Pages用)
    result = {
        "price": round(latest_price, 2),
        "rsi": round(latest_rsi, 2),
        "update_time": pd.Timestamp.now(tz='Asia/Tokyo').strftime('%Y-%m-%d %H:%M')
    }
    with open('data.json', 'w') as f:
        json.dump(result, f)

if __name__ == "__main__":
    analyze()