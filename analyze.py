import yfinance as yf
import pandas as pd
import json
import numpy as np
import sys

SYMBOL_GOLD = "GC=F"
SYMBOL_DOLLAR_LIST = ["DX=F", "UUP", "CL=F"]

def analyze():
    # --- Phase 1: Gold データ取得 (5m, 30m, 1h, 4h) ---
    try:
        print(f"--- Phase 1: Fetching Gold Data ({SYMBOL_GOLD}) ---")
        # 既存の1h/4hに加え、同期用の5m、解析用の30mを追加
        df_5m = yf.download(SYMBOL_GOLD, period="1d", interval="5m")
        df_30m = yf.download(SYMBOL_GOLD, period="5d", interval="30m")
        df_gold = yf.download(SYMBOL_GOLD, period="7d", interval="1h")
        df_gold_4h = yf.download(SYMBOL_GOLD, period="30d", interval="4h")
        
        if any(df.empty for df in [df_5m, df_30m, df_gold, df_gold_4h]):
            raise ValueError("一部の価格データが取得できませんでした。")
        
        # MultiIndex (二重列) の解消
        for df in [df_5m, df_30m, df_gold, df_gold_4h]:
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            
        print("Gold data fetched (5m, 30m, 1h, 4h).")
    except Exception as e:
        print(f"[ERROR] Gold Data Fetch Failed: {e}")
        sys.exit(1)

    # --- Phase 1.5: Dollar データ取得 & 相関計算 ---
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

    # --- Phase 2: テクニカル計算 (短期 & 長期 & S/R & 出来高 & プロ解析) ---
    try:
        print("--- Phase 2: Technical & Institutional Analysis ---")
        close_1h = df_gold['Close']
        high_1h = df_gold['High']
        low_1h = df_gold['Low']
        vol_1h = df_gold['Volume']
        
        # 短期 RSI(14) / 乖離率 / ATR
        delta = close_1h.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rsi_1h_series = 100 - (100 / (1 + (gain / loss)))
        ma25_series = close_1h.rolling(window=25).mean()
        dev_series = ((close_1h - ma25_series) / ma25_series) * 100
        atr_series = (high_1h - low_1h).rolling(window=14).mean()

        vol_sma = vol_1h.rolling(window=20).mean()
        vol_spike = bool(vol_1h.iloc[-1].item() > vol_sma.iloc[-1].item() * 1.5)

        # 長期(4h)トレンド計算
        close_4h = df_gold_4h['Close']
        ma20_4h = close_4h.rolling(window=20).mean()
        ma_long_val = ma20_4h.iloc[-1].item()
        trend_4h = "上昇" if close_4h.iloc[-1].item() > ma_long_val else "下落"
        
        delta_4h = close_4h.diff()
        gain_4h = (delta_4h.where(delta_4h > 0, 0)).rolling(window=14).mean()
        loss_4h = (-delta_4h.where(delta_4h < 0, 0)).rolling(window=14).mean()
        rsi_4h_val = float((100 - (100 / (1 + (gain_4h / loss_4h)))).iloc[-1].item())

        # レジスタンス・サポート (直近48時間)
        resistance = float(high_1h.iloc[-48:].max().item())
        support = float(low_1h.iloc[-48:].min().item())

        # CISD (需給ポイント) 解析
        change_30m = df_30m['Close'].diff()
        std_30m = change_30m.std()
        cisd_res = float(df_30m['High'][change_30m < -2 * std_30m].tail(1).item()) if not df_30m[change_30m < -2 * std_30m].empty else 0.0
        cisd_sup = float(df_30m['Low'][change_30m > 2 * std_30m].tail(1).item()) if not df_30m[change_30m > 2 * std_30m].empty else 0.0

        # 真空地帯 (空白エリア) 解析
        hist, bin_edges = np.histogram(df_30m['Close'], bins=15, weights=df_30m['Volume'])
        avg_hist = np.mean(hist)
        vacuum_zones = []
        for i in range(len(hist)):
            if hist[i] < avg_hist * 0.3:
                vacuum_zones.append({"from": round(bin_edges[i], 1), "to": round(bin_edges[i+1], 1)})

        # 時間軸同期 (5m & 30m)
        trend_5m = 1 if df_5m['Close'].iloc[-1].item() > df_5m['Close'].ewm(span=20).mean().iloc[-1].item() else -1
        trend_30m_fast = 1 if df_30m['Close'].iloc[-1].item() > df_30m['Close'].ewm(span=20).mean().iloc[-1].item() else -1
        is_synced = (trend_5m == trend_30m_fast)

        latest_price = float(close_1h.iloc[-1].item())
        latest_rsi = float(rsi_1h_series.iloc[-1].item())
        latest_dev = float(dev_series.iloc[-1].item())
        atr_expanding = bool(atr_series.iloc[-1].item() > atr_series.iloc[-2].item())
        
    except Exception as e:
        print(f"[ERROR] Calculations Failed: {e}")
        import traceback; traceback.print_exc(); sys.exit(1)

    # --- Phase 3: スコアリング & シグナル判定 ---
    try:
        score_1h = (50 - latest_rsi) * 1.5 + (latest_dev * -15)
        if atr_expanding: score_1h *= 1.2
        final_score_1h = int(max(min(score_1h, 100), -100))
        final_score_4h = int(max(min((50 - rsi_4h_val) * 2, 100), -100))

        is_golden = (final_score_1h > 30 and trend_4h == "上昇")
        is_death = (final_score_1h < -30 and trend_4h == "下落")

        if is_golden: status, reason = "✨ GOLDEN SIGN", f"短期・長期買い圧同調。需給サポート ${cisd_sup:.1f}。"
        elif is_death: status, reason = "💀 DEATH SIGN", f"短期・長期売り圧同調。需給レジスタンス ${cisd_res:.1f}。"
        elif is_synced: status, reason = "✨ 期待値最大化ポイント", f"5m/30m同期。{'上昇' if trend_5m==1 else '下落'}への加速注意。"
        elif final_score_1h > 30: status, reason = "押し目買い", f"RSI {latest_rsi:.1f}。長期{trend_4h}トレンド内。"
        elif final_score_1h < -30: status, reason = "戻り売り", f"乖離率 {latest_dev:.1f}%。長期{trend_4h}中。"
        else: status, reason = "静観", f"長期は{trend_4h}。明確なシグナル待ち。"
            
    except Exception as e:
        print(f"[ERROR] Scoring Failed: {e}"); sys.exit(1)

    # --- Phase 4: JSON出力 ---
    try:
        result = {
            "price": round(latest_price, 2),
            "rsi": round(latest_rsi, 2),
            "deviation": round(latest_dev, 2),
            "correlation": round(correlation, 2) if not np.isnan(correlation) else 0.0,
            "score": final_score_1h,
            "score_1h": final_score_1h,
            "score_4h": final_score_4h,
            "is_golden": is_golden,
            "is_death": is_death,
            "vol_spike": vol_spike,
            "trend_4h": trend_4h,
            "resistance": round(resistance, 2),
            "support": round(support, 2),
            "cisd_high": round(cisd_res, 2),
            "cisd_low": round(cisd_sup, 2),
            "vacuum_zones": vacuum_zones[:3],
            "is_synced": is_synced,
            "sync_dir": "上昇" if trend_5m == 1 else "下落",
            "status": status,
            "reason": reason,
            "buy_ratio": int(latest_rsi),
            "update_time": pd.Timestamp.now(tz='Asia/Tokyo').strftime('%Y-%m-%d %H:%M')
        }
        with open('data.json', 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=4)
        print("Update Completed Successfully.")
    except Exception as e:
        print(f"[ERROR] JSON Output Failed: {e}"); sys.exit(1)

if __name__ == "__main__":
    analyze()