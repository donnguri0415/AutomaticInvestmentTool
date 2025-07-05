import os
import glob
import pandas as pd
import argparse
from importlib.machinery import SourceFileLoader
from mt5_fetch.get_price_CSV import fetch_price_csv
from feature.add_features import add_features

# コマンドライン引数の定義
parser = argparse.ArgumentParser(
    description="MT5からチャートデータを取得／特徴量生成／予測 or モデル再学習を行うスクリプト"
)
parser.add_argument(
    '--mode', choices=['predict','train'], default='predict',
    help='実行モードを指定: predict=予測, train=モデル再学習'
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
    '--bars', type=int, default=3000,
    help='予測モード時に取得するバー本数'
)
parser.add_argument(
    '--train-bars', type=int, default=100000,
    help='再学習モード時に取得するバー本数'
)
args = parser.parse_args()

symbol = args.symbol
frame = args.timeframe
target_bars = args.train_bars if args.mode == 'train' else args.bars

# trade_decision_logic_v2 をロード
trade_logic = SourceFileLoader(
    'trade_decision_logic_v2',
    os.path.join('predict','trade_decision_logic_v2.py')
).load_module()
should_enter_trade = trade_logic.should_enter_trade

# ステップ1: データ取得
print(f"▶ Fetching {target_bars} bars for {symbol}_{frame}...")
csv_path = fetch_price_csv(symbol, frame, target_bars, out_dir="data")
print(f"✅ Fetched: {csv_path}")

# ステップ2: 特徴量生成
print("▶ Generating features...")
csv_path = add_features(symbol, frame, in_dir="data", out_dir="data")
print(f"✅ Generated features at {csv_path}")

if args.mode == 'train':
    # ステップ3: モデル再学習
    print("▶ Training model...")
    os.system(
        f"python ./model/train_model.py --symbol {symbol} --timeframe {frame} --atr-multiplier 0.5 --train-bars {args.train_bars}"
    )
    init_model(model_path)
    print("✅ Model training completed")
else:
    # ステップ3: 最新データで予測判定
    print("▶ Predicting on latest bar...")
    df = pd.read_csv(
        f"./data/{symbol}_{frame}_features_v2.csv",
        index_col='time', parse_dates=True
    )
    latest = df.iloc[-1]
    decision = should_enter_trade(
        latest,
        model_path=f"./model/model_lgbm_best_{symbol}_{frame}.pkl"
    )
    print("\n📢 [最終トレード判断]")
    if decision.get("enter"):
        print(f"▶ エントリー: {decision['direction']} (確率: {decision['probability']:.2%})")
        # EA Files フォルダへの出力パスを自動検出
        base_dir = os.path.expanduser("~/.wine/drive_c/Users/$USER/AppData/Roaming/MetaQuotes/Terminal")
        # Windows 環境では ~ expands to C:/Users/<user>
        base_dir = os.path.expanduser("~/AppData/Roaming/MetaQuotes/Terminal")
        files_dirs = glob.glob(os.path.join(base_dir, '*', 'MQL5', 'Files'))
        if files_dirs:
            out_dir = files_dirs[0]
        else:
            # 直接パスがない場合は作成
            out_dir = os.path.join(base_dir, 'MQL5', 'Files')
            os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f"predict_result_batch_{symbol}.csv")
        signal = pd.DataFrame([[
            pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
            1,
            decision['direction'],
            decision['probability'],
            decision['tp'],
            decision['sl']
        ]])
        signal.to_csv(out_path, index=False, header=False)
        print(f"✅ Signal written to: {out_path}")
    else:
        print(f"▶ ノーエントリー (確率: {decision['probability']:.2%})")
