import os
import glob
import pandas as pd
import argparse
import yaml  # pyyaml required
from importlib.machinery import SourceFileLoader
from mt5_fetch.get_price_data import fetch_price_df
from feature.add_features_df import add_features_df
import time
t0 = time.perf_counter()

# --- コマンドライン引数の定義 ---
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
parser.add_argument(
    '--config', type=str, default='config.yaml',
    help='設定ファイルパス (YAML)'
)
args = parser.parse_args()

# --- 設定ファイル読み込み (YAML) ---
with open(args.config, 'r', encoding='utf-8') as f:
    cfg = yaml.safe_load(f)
TERMINAL_GUID = cfg.get('terminal_guid')  # MT5 Terminal GUID

# --- MT5 Files フォルダ検出 ---
appdata = os.path.expanduser(r'~\\AppData\\Roaming\\MetaQuotes\\Terminal')
if TERMINAL_GUID:
    base_dir = os.path.join(appdata, TERMINAL_GUID, 'MQL5', 'Files')
    if not os.path.isdir(base_dir):
        raise RuntimeError(f"指定されたターミナルGUIDのFilesフォルダが見つかりません: {base_dir}")
else:
    candidates = glob.glob(os.path.join(appdata, '*', 'MQL5', 'Files'))
    if not candidates:
        raise RuntimeError("Terminal/MQL5/Files フォルダが見つかりません。")
    base_dir = max(candidates, key=lambda p: os.path.getmtime(p))

# --- パラメータ設定 ---
symbol      = args.symbol
frame       = args.timeframe
target_bars = args.train_bars if args.mode == 'train' else args.bars

# --- トレードロジック読み込み ---
trade_logic = SourceFileLoader(
    'trade_decision_logic_v2',
    os.path.join('predict','trade_decision_logic_v2.py')
).load_module()
should_enter_trade = trade_logic.should_enter_trade
init_model = trade_logic.init_model

# モデルパス
model_path = f"model/model_lgbm_best_{symbol}_{frame}.pkl"

# --- ステップ1: データ取得 ---
print(f"▶ Fetching {target_bars} bars for {symbol}_{frame}...")
ohlcv_df   = fetch_price_df(symbol, frame, target_bars)
print("✅ Fetched")
t1 = time.perf_counter()

# --- ステップ2: 特徴量生成 ---
print("▶ Generating features...")
feat_df    = add_features_df(ohlcv_df)
print("✅ Generated features")
t2 = time.perf_counter()

if args.mode == 'train':
    # --- ステップ3: モデル再学習 ---
    print("▶ Training model...")
    os.system(
        f"python ./model/train_model.py --symbol {symbol} --timeframe {frame} --atr-multiplier 0.5 --train-bars {args.train_bars}"
    )
    init_model(model_path)
    print("✅ Model training completed")
else:
    # --- ステップ3: 最新データで予測判定 ---
    print("▶ Predicting on latest bar...")
    latest = feat_df.iloc[-1]
    t3     = time.perf_counter()
    # NOTE: Pass model_path to ensure parsing
    pred_start = time.perf_counter()
    decision = should_enter_trade(latest, model_path=model_path)
    pred_end   = time.perf_counter()

    print("\n📢 [最終トレード判断]")
    if decision.get('enter'):
        print(f"▶ エントリー: {decision['direction']} (確率: {decision['probability']:.2%})")
        out_path = os.path.join(base_dir, f"predict_result_batch_{symbol}.csv")
        sig_start = time.perf_counter()
        signal = pd.DataFrame([[
            pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
            1,
            decision['direction'],
            decision['probability'],
            decision['tp'],
            decision['sl']
        ]])
        signal.to_csv(out_path, index=False, header=False)
        sig_end = time.perf_counter()
        print(f"✅ Signal written to: {out_path}")
        print(
	        f"[TIMING] fetch={(t1-t0):.3f}s | feat={(t2-t1):.3f}s | load={(t3-t2):.3f}s "
	        f"| predict={(pred_end-pred_start):.3f}s "
	        f"| write={(sig_end-sig_start):.3f}s → {decision}"
)
    else:
        print(f"▶ ノーエントリー (確率: {decision['probability']:.2%})")
