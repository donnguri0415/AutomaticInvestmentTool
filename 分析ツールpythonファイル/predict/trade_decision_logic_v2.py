import os
import pandas as pd
import joblib
import time

# 特徴量列（モデルと一致させる）
feature_cols = [
    'volatility', 'atr', 'adx', 'macd_diff',
    'bb_width', 'body', 'stoch_k', 'adx_neg',
    'adx_pos', 'mfi'
]
# エントリー確率のしきい値（固定）
THRESHOLD = 0.5

# グローバル変数でモデルインスタンスを保持
_MODEL = None


def init_model(model_path: str):
    """
    一度だけモデルをロードし、グローバル _MODEL に保持する
    """
    global _MODEL
    if _MODEL is None:
        _MODEL = joblib.load(model_path)
        # 推論時のワーカ数を1に固定
        _MODEL.set_params(n_jobs=1)
        print(f"[init_model] Loaded model from {model_path}")


def should_enter_trade(latest_row: pd.Series,
                       model_path: str = None,
                       threshold: float = THRESHOLD,
                       percentile: float = 75.0,
                       indir: str = 'data') -> dict:
    """
    最新の特徴量行をもとにモデルの予測確率を計算し、トレード判断を行う。
    model_pathは初回ロード時のみ使用。
    """
    global _MODEL
    # 初回またはリロード時にモデルパスが与えられればロード
    t0 = time.perf_counter() # debug
    if _MODEL is None:
        if model_path is None:
            raise RuntimeError("Model not initialized: call init_model() with model_path first.")
        init_model(model_path)
    t1 = time.perf_counter() # debug
    # モデルファイル名からsymbolとtimeframeを抽出（TP/SL計算用）
    fname = os.path.basename(model_path) if model_path else ''
    name, _ = os.path.splitext(fname)
    parts = name.split('_')
    if len(parts) < 2:
        raise ValueError(f"Cannot parse symbol/timeframe from model filename: {fname}")
    symbol, timeframe = parts[-2], parts[-1]

    # 特徴量配列を整形して予測
    X_latest = latest_row[feature_cols].values.reshape(1, -1)
    proba = _MODEL.predict_proba(X_latest)[0]
    print(f"[PROFILE] init_model={(t1-t0):.3f}s") #debug
    prob_buy, prob_sell = proba[1], proba[0]

    # エントリーロジック
    if prob_buy > threshold:
        return {"enter": True,  "direction": "buy",  "probability": prob_buy}
    elif prob_sell > threshold:
        return {"enter": True,  "direction": "sell", "probability": prob_sell}
    else:
        return {"enter": False, "direction": None,   "probability": max(prob_buy, prob_sell), "tp": None, "sl": None}


def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="最新の特徴量からトレード判断を行うスクリプト"
    )
    parser.add_argument('--symbol',    type=str, default='EURUSDm')
    parser.add_argument('--timeframe', type=str, default='M15')
    parser.add_argument('--percentile',type=float, default=75.0)
    parser.add_argument('--indir',     type=str, default='data')
    args = parser.parse_args()

    model_file = os.path.join('model', f"model_lgbm_best_{args.symbol}_{args.timeframe}.pkl")
    # 初回ロード
    init_model(model_file)

    df = pd.read_csv(
        os.path.join(args.indir, f"{args.symbol}_{args.timeframe}_features_v2.csv"),
        index_col='time', parse_dates=True
    )
    latest = df.iloc[-1]
    decision = should_enter_trade(latest)
    print(f"📈 トレード判断: {decision}")


if __name__ == '__main__':
    main()
