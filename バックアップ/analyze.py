import yfinance as yf
import pandas as pd
import json
import numpy as np
import sys

SYMBOL_GOLD = "GC=F"
SYMBOL_DOLLAR_LIST = ["DX=F", "UUP", "CL=F"]

def analyze():
    # --- Phase 1: Gold データ取得 (1h & 4h) ---
    try:
        print(f"--- Phase 1: Fetching Gold Data ({SYMBOL_GOLD}) ---")
        # 短期判断用(1h)と長期トレンド用(4h)を分けて取得
        df_gold = yf.download(SYMBOL_GOLD, period="7d", interval="1h")
        df_gold_4h = yf.download(SYMBOL_GOLD, period="30d", interval="4h")
        
        if df_gold.empty or df_gold_4h.empty:
            raise ValueError("金の価格データが空です。")
        
        # MultiIndex (二重列) の解消
        for df in [df_gold, df_gold_4h]:
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            
        print("Gold data fetched (1h & 4h).")
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
            combined = pd.concat([df_gold['Close'], df_dollar['Close']], axis=1).dropna()
            if len(combined) > 1:
                correlation = float(combined.corr().iloc[0, 1])
        except:
            correlation = 0.0

    # --- Phase 2: テクニカル計算 (短期 & 長期 & S/R) ---
    try:
        print("--- Phase 2: Technical Analysis ---")
        # 短期(1h)データ
        close_1h = df_gold['Close']
        high_1h = df_gold['High']
        low_1h = df_gold['Low']
        
        # 1h RSI(14)
        delta = close_1h.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rsi_1h_series = 100 - (100 / (1 + (gain / loss)))
        
        # 1h MA25 & Deviation
        ma25_series = close_1h.rolling(window=25).mean()
        dev_series = ((close_1h - ma25_series) / ma25_series) * 100
        
        # ATR (1h)
        atr_series = (high_1h - low_1h).rolling(window=14).mean()

        # 長期(4h)トレンド計算 (エラー箇所修正)
        close_4h = df_gold_4h['Close']
        ma20_4h = close_4h.rolling(window=20).mean() # 4時間足20MA
        ma_long_val = ma20_4h.iloc[-1].item()
        trend_4h = "上昇" if close_4h.iloc[-1].item() > ma_long_val else "下落"
        
        # 4h RSI(14)
        delta_4h = close_4h.diff()
        gain_4h = (delta_4h.where(delta_4h > 0, 0)).rolling(window=14).mean()
        loss_4h = (-delta_4h.where(delta_4h < 0, 0)).rolling(window=14).mean()
        rsi_4h_val = float((100 - (100 / (1 + (gain_4h / loss_4h)))).iloc[-1].item())

        # レジスタンス・サポート算出 (直近48時間の最高値・最安値)
        resistance = float(high_1h.iloc[-48:].max().item())
        support = float(low_1h.iloc[-48:].min().item())

        # 最新値の抽出
        latest_price = float(close_1h.iloc[-1].item())
        latest_rsi = float(rsi_1h_series.iloc[-1].item())
        latest_dev = float(dev_series.iloc[-1].item())
        atr_expanding = bool(atr_series.iloc[-1].item() > atr_series.iloc[-2].item())
        
        print(f"Latest Price: {latest_price}, RSI(1h): {latest_rsi:.2f}, Trend(4h): {trend_4h}")
    except Exception as e:
        print(f"[ERROR] Calculations Failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # --- Phase 3: スコアリング ---
    try:
        print("--- Phase 3: Scoring ---")
        # 短期スコア (既存ロジックを維持)
        score_1h = (50 - latest_rsi) * 1.5 + (latest_dev * -15)
        if atr_expanding: score_1h *= 1.2
        final_score_1h = int(max(min(score_1h, 100), -100))

        # 長期スコア
        final_score_4h = int(max(min((50 - rsi_4h_val) * 2, 100), -100))

        # AIコメント
        if final_score_1h > 30:
            status = "押し目買い"
            reason = f"短期RSI {latest_rsi:.1f}。長期{trend_4h}トレンドの中での反発。"
        elif final_score_1h < -30:
            status = "戻り売り"
            reason = f"短期乖離 {latest_dev:.1f}%。長期{trend_4h}トレンド。過熱調整。"
        else:
            status = "静観"
            reason = f"長期は{trend_4h}中。短期シグナル待ち。"
            
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
            "score": final_score_1h,
            "score_1h": final_score_1h,
            "score_4h": final_score_4h,
            "trend_4h": trend_4h,
            "resistance": round(resistance, 2),
            "support": round(support, 2),
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