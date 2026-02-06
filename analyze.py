import yfinance as yf
import pandas as pd
import json
import numpy as np
import sys

SYMBOL_GOLD = "GC=F"
SYMBOL_DOLLAR = "DX=F"

def analyze():
    # --- PHASE 1: APIデータ取得 ---
    try:
        print(f"--- Phase 1: Fetching data for {SYMBOL_GOLD} and {SYMBOL_DOLLAR} ---")
        gold = yf.Ticker(SYMBOL_GOLD)
        dollar = yf.Ticker(SYMBOL_DOLLAR)
        
        df_gold = gold.history(period="5d", interval="1h")
        df_dollar = dollar.history(period="5d", interval="1h")
        
        if df_gold.empty or df_dollar.empty:
            raise ValueError("取得したデータが空です。市場が休場かAPIの制限の可能性があります。")
        print("Data fetch successful.")
    except Exception as e:
        print(f"[ERROR] API取得ミス: {e}")
        sys.exit(1)

    # --- PHASE 2: テクニカル計算 ---
    try:
        print("--- Phase 2: Calculating technical indicators ---")
        # RSI(14)
        delta = df_gold['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df_gold['RSI'] = 100 - (100 / (1 + rs))
        
        # MA25 & Deviation
        df_gold['MA25'] = df_gold['Close'].rolling(window=25).mean()
        df_gold['Deviation'] = ((df_gold['Close'] - df_gold['MA25']) / df_gold['MA25']) * 100
        
        # ATR (Volatility)
        high_low = df_gold['High'] - df_gold['Low']
        high_close = np.abs(df_gold['High'] - df_gold['Close'].shift())
        low_close = np.abs(df_gold['Low'] - df_gold['Close'].shift())
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df_gold['ATR'] = tr.rolling(window=14).mean()
        
        correlation = df_gold['Close'].corr(df_dollar['Close'])
        
        latest_rsi = df_gold['RSI'].iloc[-1]
        latest_dev = df_gold['Deviation'].iloc[-1]
        atr_expanding = df_gold['ATR'].iloc[-1] > df_gold['ATR'].iloc[-2]
        
        print(f"Indicators: RSI={latest_rsi:.2f}, Dev={latest_dev:.2f}, Corr={correlation:.2f}")
    except Exception as e:
        print(f"[ERROR] 計算ミス (テクニカル指標): {e}")
        sys.exit(1)

    # --- PHASE 3: スコアリング & AI判定 ---
    try:
        print("--- Phase 3: Scoring and Decision making ---")
        score = (50 - latest_rsi) * 1.5 + (latest_dev * -15)
        if atr_expanding:
            score *= 1.2
        final_score = max(min(int(score), 100), -100)

        if final_score > 30:
            status, reason = "押し目買い", f"RSI{int(latest_rsi)}。反発の兆候あり。"
        elif final_score < -30:
            status, reason = "戻り売り", f"乖離率{latest_dev:.1f}%。過熱感あり。"
        else:
            status, reason = "静観", "シグナル混合。待機推奨。"
    except Exception as e:
        print(f"[ERROR] 計算ミス (判定ロジック): {e}")
        sys.exit(1)

    # --- PHASE 4: 結果保存 ---
    try:
        print("--- Phase 4: Saving results to data.json ---")
        result = {
            "price": round(df_gold['Close'].iloc[-1], 2),
            "rsi": round(latest_rsi, 2),
            "deviation": round(latest_dev, 2),
            "correlation": round(correlation, 2),
            "score": final_score,
            "status": status,
            "reason": reason,
            "buy_ratio": int(latest_rsi), # RSIを暫定的な買い圧として利用
            "update_time": pd.Timestamp.now(tz='Asia/Tokyo').strftime('%Y-%m-%d %H:%M')
        }
        with open('data.json', 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=4)
        print("Success: data.json updated.")
    except Exception as e:
        print(f"[ERROR] ファイル書き込みミス: {e}")
        sys.exit(1)

if __name__ == "__main__":
    analyze()