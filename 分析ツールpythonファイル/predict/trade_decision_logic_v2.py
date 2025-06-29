import pandas as pd
import joblib
import argparse
import os
import re
from tools.compute_tp_sl import get_tp_sl

# 特徴量列（モデルと一致させる）
feature_cols = [
    'volatility', 'atr', 'adx', 'macd_diff',
    'bb_width', 'body', 'stoch_k', 'adx_neg',
    'adx_pos', 'mfi'
]

# エントリー確率のしきい値（固定）
THRESHOLD = 0.5


def should_enter_trade(latest_row, model_path, threshold=THRESHOLD, percentile=75.0, indir='data'):
    """
    最新の特徴量行をもとにモデルの予測確率を計算し、トレード判断を行う
    モデルファイル名からsymbolとtimeframeを自動抽出し、TP/SLも自動設定

    :param latest_row: pandas.Series 最新の特徴量データ
    :param model_path: str 学習済みモデルのパス
    :param threshold: float エントリー判断しきい値
    :param percentile: float TP/SL算出に用いるパーセンタイル
    :param indir: str 特徴量CSVがあるディレクトリ
    :return: dict トレード判断結果
    """
    # モデルファイル名からsymbolとtimeframeを抽出
    fname = os.path.basename(model_path)
    name, _ = os.path.splitext(fname)
    parts = name.split('_')
    if len(parts) < 2:
        raise ValueError(f"モデルファイル名からsymbol/timeframeを解析できません: {fname}")
    symbol = parts[-2]
    timeframe = parts[-1]

    # 動的TP/SL計算
    tp, sl = get_tp_sl(symbol, timeframe, indir=indir, percentile=percentile)

    # モデル読み込み・予測
    model = joblib.load(model_path)
    X_latest = latest_row[feature_cols].values.reshape(1, -1)
    proba = model.predict_proba(X_latest)[0]
    prob_buy = proba[1]  # クラス1が買い
    prob_sell = proba[0] # クラス0が売り
    
    # Debug
    print(f"prob_buy；{prob_buy}")
    print(f"prob_sell；{prob_sell}")

    # エントリーロジック: 買い/売り両対応
    if prob_buy > threshold:
        return {
            "enter": True,
            "direction": "buy",
            "probability": prob_buy,
            "tp": tp,
            "sl": sl
        }
    elif prob_sell > threshold:
        return {
            "enter": True,
            "direction": "sell",
            "probability": prob_sell,
            "tp": tp,
            "sl": sl
        }
    else:
        return {
            "enter": False,
            "direction": None,
            "probability": max(prob_buy, prob_sell),
            "tp": None,
            "sl": None
        }


def main():
    parser = argparse.ArgumentParser(
        description="最新の特徴量からトレード判断を行うスクリプト"
    )
    parser.add_argument(
        '--symbol', type=str, default='EURUSDm',
        help='通貨ペアを指定（例: EURUSDm）'
    )
    parser.add_argument(
        '--timeframe', type=str, default='M15',
        choices=['M1','M5','M15','M30','H1','H4','D1','W1','MN1'],
        help='時間足を指定（例: M15, H1）'
    )
    parser.add_argument(
        '--percentile', type=float, default=75.0,
        help='TP/SL計算に使うパーセンタイル'
    )
    parser.add_argument(
        '--indir', type=str, default='data',
        help='特徴量CSVのディレクトリ'
    )
    args = parser.parse_args()

    symbol = args.symbol
    timeframe = args.timeframe.upper()

    # ファイルパス
    data_file = os.path.join(args.indir, f"{symbol}_{timeframe}_features_v2.csv")
    model_file = os.path.join('model', f"model_lgbm_best_{symbol}_{timeframe}.pkl")

    # 最新データ取得
    df = pd.read_csv(data_file, index_col='time', parse_dates=True)
    latest = df.iloc[-1]

    # トレード判断
    decision = should_enter_trade(latest, model_file,
                                  threshold=THRESHOLD,
                                  percentile=args.percentile,
                                  indir=args.indir)
    print(f"📈 トレード判断: {decision}")


if __name__ == '__main__':
    main()
