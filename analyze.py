import yfinance as yf
import pandas as pd
import json
import numpy as np
import sys

SYMBOL_GOLD = "GC=F"
SYMBOL_DOLLAR_LIST = ["DX=F", "UUP", "CL=F"]

def analyze():
    # --- Phase 1: Gold データ取得 ---
    try:
        print(f"--- Phase 1: Fetching Gold Data ({SYMBOL_GOLD}) ---")
        df_gold = yf.download(SYMBOL_GOLD, period="7d", interval="1h")
        if df_gold.empty:
            raise ValueError("金の価格データが空です。")
        
        # MultiIndex (二重列) の解消
        if isinstance(df_gold.columns, pd.MultiIndex):
            df_gold.columns = df_gold.columns.get_level_values(0)
            
        print("Gold data fetched and flattened.")
    except Exception as e:
        print(f"[ERROR] Gold Data Fetch Failed: {e}")
        sys.exit(1)

    # --- Phase 1.5: Dollar データ取得 ---
    df_dollar = pd.DataFrame()
    correlation = 0.0
    for sym in SYMBOL_DOLLAR_LIST:
        try:
            temp_df = yf.download(sym, period="7d", interval="1h")
            if not temp_df.empty:
                if isinstance(temp_df.columns, pd.MultiIndex):
                    temp_df.columns = temp_df.columns.get_level_values(0)
                df_dollar = temp_df
                break
        except:
            continue
    
    if not df_dollar.empty:
        try:
            # 終値同士を結合して相関計算
            combined = pd.concat([df_gold['Close'], df_dollar['Close']], axis=1).dropna()
            if len(combined) > 1:
                correlation = float(combined.corr().iloc[0, 1])
        except:
            correlation = 0.0

    # --- Phase 2: テクニカル計算 ---
    try:
        print("--- Phase 2: Technical Analysis ---")
        close = df_gold['Close']
        high = df_gold['High']
        low = df_gold['Low']
        
        # RSI(14) - スカラー値として抽出するために .item() を使用
        delta = close.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rsi_series = 100 - (100 / (1 + (gain / loss)))
        
        # MA25 & Deviation
        ma25_series = close.rolling(window=25).mean()
        dev_series = ((close - ma25_series) / ma25_series) * 100
        
        # ATR (簡易)
        atr_series = (high - low).rolling(window=14).mean()

        # 最新値の抽出（警告回避のため .iloc[-1] の後に .item() を付与）
        latest_price = float(close.iloc[-1].item())
        latest_rsi = float(rsi_series.iloc[-1].item())
        latest_dev = float(dev_series.iloc[-1].item())
        
        # ボラティリティ拡大判定
        # .item() を使うことで Series ではなく単一の bool 値として取得
        current_atr = atr_series.iloc[-1].item()
        prev_atr = atr_series.iloc[-2].item()
        atr_expanding = bool(current_atr > prev_atr)
        
        print(f"Latest Price: {latest_price}, RSI: {latest_rsi:.2f}")
    except Exception as e:
        print(f"[ERROR] Calculations Failed: {e}")
        import traceback
        traceback.print_exc() # 詳細なエラー箇所を出力
        sys.exit(1)

    # --- Phase 3: スコアリング ---
    try:
        print("--- Phase 3: Scoring ---")
        score = (50 - latest_rsi) * 1.5 + (latest_dev * -15)
        if atr_expanding: score *= 1.2
        final_score = int(max(min(score, 100), -100))

        if final_score > 30: status, reason = "押し目買い", f"RSI {latest_rsi:.1f}。反発の兆候あり。"
        elif final_score < -30: status, reason = "戻り売り", f"乖離率 {latest_dev:.1f}%。過熱感あり。"
        else: status, reason = "静観", "トレンド転換待ち。待機。"
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
            "correlation": round(correlation, 2) if not np.isnan(correlation) else 0.0,
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