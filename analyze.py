import yfinance as yf
import pandas as pd
import json
import numpy as np

# 設定
SYMBOL_GOLD = "GC=F"   # 金先物
SYMBOL_DOLLAR = "DX=F" # ドル指数先物

def analyze():
    # 1. データ取得（金とドル指数）
    gold = yf.Ticker(SYMBOL_GOLD)
    dollar = yf.Ticker(SYMBOL_DOLLAR)
    
    df_gold = gold.history(period="5d", interval="1h")
    df_dollar = dollar.history(period="5d", interval="1h")
    
    if df_gold.empty or df_dollar.empty:
        return

    # 2. テクニカル指標計算
    # RSI(14)
    delta = df_gold['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df_gold['RSI'] = 100 - (100 / (1 + rs))
    
    # 25期間移動平均線と乖離率
    df_gold['MA25'] = df_gold['Close'].rolling(window=25).mean()
    df_gold['Deviation'] = ((df_gold['Close'] - df_gold['MA25']) / df_gold['MA25']) * 100
    
    # ATR (14期間)
    high_low = df_gold['High'] - df_gold['Low']
    high_close = np.abs(df_gold['High'] - df_gold['Close'].shift())
    low_close = np.abs(df_gold['Low'] - df_gold['Close'].shift())
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df_gold['ATR'] = tr.rolling(window=14).mean()
    # ボラティリティ拡大判定（直近のATRが前時間を上回っているか）
    atr_expanding = df_gold['ATR'].iloc[-1] > df_gold['ATR'].iloc[-2]

    # 3. ドル相関分析 (金とドルの終値相関)
    correlation = df_gold['Close'].corr(df_dollar['Close'])
    
    # 4. 統合スコアリング (-100 〜 +100)
    latest_rsi = df_gold['RSI'].iloc[-1]
    latest_dev = df_gold['Deviation'].iloc[-1]
    
    score = 0
    # RSI(14) スコア: 50を基準に過熱感を算出
    score += (50 - latest_rsi) * 1.5 
    
    # MA25乖離スコア: 逆張りの圧力を算出
    score += (latest_dev * -15) 
    
    # ATRボラティリティ補正: 拡大時はスコアを1.2倍に強調（勢いがある）
    if atr_expanding:
        score *= 1.2
    
    final_score = max(min(int(score), 100), -100)

    # 5. AIコメント生成 (20文字程度)
    if final_score > 30:
        status = "押し目買い"
        reason = f"RSI{int(latest_rsi)}低下。MA乖離からの反発狙い。"
    elif final_score < -30:
        status = "戻り売り"
        reason = f"乖離率{latest_dev:.1f}%。過熱感による調整局面。"
    else:
        status = "静観"
        reason = "シグナル混合。明確なトレンド待ち。"

    # 6. 結果をJSONに保存
    result = {
        "price": round(df_gold['Close'].iloc[-1], 2),
        "rsi": round(latest_rsi, 2),
        "deviation": round(latest_dev, 2),
        "correlation": round(correlation, 2),
        "score": final_score,
        "status": status,
        "reason": reason,
        "update_time": pd.Timestamp.now(tz='Asia/Tokyo').strftime('%Y-%m-%d %H:%M')
    }
    
    with open('data.json', 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=4)

if __name__ == "__main__":
    analyze()