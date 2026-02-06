import yfinance as yf
import pandas as pd
import json
import numpy as np
import sys

SYMBOL_GOLD = "GC=F"
# ドル指数が取れない場合の予備を含めたリスト
SYMBOL_DOLLAR_LIST = ["DX=F", "UUP", "CL=F"] # UUPはドル指数のETF

def analyze():
    # --- Phase 1: データ取得 (Gold) ---
    try:
        print(f"--- Phase 1: Fetching Gold Data ({SYMBOL_GOLD}) ---")
        df_gold = yf.download(SYMBOL_GOLD, period="7d", interval="1h")
        if df_gold.empty:
            raise ValueError("金の価格データが空です。")
    except Exception as e:
        print(f"[ERROR] Gold Data Fetch Failed: {e}")
        sys.exit(1) # 金が取れない場合は続行不可

    # --- Phase 1.5: データ取得 (Dollar Index / 予備含む) ---
    df_dollar = pd.DataFrame()
    correlation = 0
    print("--- Phase 1.5: Fetching Dollar Data (Optional) ---")
    for sym in SYMBOL_DOLLAR_LIST:
        try:
            temp_df = yf.download(sym, period="7d", interval="1h")
            if not temp_df.empty:
                df_dollar = temp_df
                print(f"Success: Fetched dollar-related data from {sym}")
                break
        except:
            continue
    
    # 相関計算（ドルが取れた場合のみ）
    if not df_dollar.empty:
        try:
            combined = pd.concat([df_gold['Close'], df_dollar['Close']], axis=1).dropna()
            if len(combined) > 1:
                correlation = combined.corr().iloc[0, 1]
        except:
            correlation = 0

    # --- Phase 2: テクニカル計算 (Gold) ---
    try:
        print("--- Phase 2: Technical Analysis ---")
        close = df_gold['Close']
        
        # RSI(14)
        delta = close.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rsi = 100 - (100 / (1 + (gain / loss)))
        
        # MA25 & Deviation
        ma25 = close.rolling(window=25).mean()
        dev = ((close - ma25) / ma25) * 100
        
        # ATR
        high_low = df_gold['High'] - df_gold['Low']
        atr = high_low.rolling(window=14).mean()

        latest_price = float(close.iloc[-1])
        latest_rsi = float(rsi.iloc[-1])
        latest_dev = float(dev.iloc[-1])
        atr_expanding = bool(atr.iloc[-1] > atr.iloc[-2])
        
    except Exception as e:
        print(f"[ERROR] Calculations Failed: {e}")
        sys.exit(1)

    # --- Phase 3: スコアリング ---
    try:
        print("--- Phase 3: Scoring ---")
        # 簡易スコアロジック
        score = (50 - latest_rsi) * 1.5 + (latest_dev * -15)
        if atr_expanding: score *= 1.2
        final_score = int(max(min(score, 100), -100))

        if final_score > 30: status, reason = "押し目買い", f"RSI {latest_rsi:.1f}。反発の好機。"
        elif final_score < -30: status, reason = "戻り売り", f"乖離率 {latest_dev:.1f}%。過熱感あり。"
        else: status, reason = "静観", "シグナルなし。様子見。"
    except Exception as e:
        print(f"[ERROR] Scoring Failed: {e}")
        sys.exit(1)

    # --- Phase 4: JSON出力 ---
    try:
        print("--- Phase 4: Updating data.json ---")
        result = {
            "price": round(latest_price, 2),
            "rsi": round(latest_rsi, 2),
            "deviation": round(latest_dev, 2),
            "correlation": round(float(correlation), 2) if not np.isnan(correlation) else 0,
            "score": final_score,
            "status": status,
            "reason": reason,
            "buy_ratio": int(latest_rsi),
            "update_time": pd.Timestamp.now(tz='Asia/Tokyo').strftime('%Y-%m-%d %H:%M')
        }
        with open('data.json', 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=4)
        print("Update Completed Successfully.")
    except Exception as e:
        print(f"[ERROR] JSON Output Failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    analyze()