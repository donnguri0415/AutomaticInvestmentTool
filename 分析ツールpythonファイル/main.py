import os
import pandas as pd
from importlib.machinery import SourceFileLoader
import argparse
from predict.trade_decision_logic import should_enter_trade

# コマンドライン引数の定義
parser = argparse.ArgumentParser(description="MT5からチャートデータを取得してCSVに保存するスクリプト")
parser.add_argument('--symbol',    type=str, default='EURUSDm',
                    help='通貨ペアを指定（例: EURUSDm）')
parser.add_argument('--timeframe', type=str, default='M15',
                    choices=['M1','M5','M15','M30','H1','H4','D1','W1','MN1'],
                    help='時間足を指定（例: M15, H1）')
parser.add_argument('--bars',      type=int, default=9000,
                    help='取得するバーの本数')
args = parser.parse_args()

# パラメータ各種
symbol = "ETHUSDm"
timeframe = "H1"
bars = 3000
#bars = 100000 # モデルの再学習用。週に1回実行をお願いします。

# trade_decision_logic.py を predict フォルダから読み込み
trade_logic = SourceFileLoader("trade_decision_logic_v2", "./predict/trade_decision_logic_v2.py").load_module()
should_enter_trade = trade_logic.should_enter_trade

# ステップ1: データ取得
os.system(f"python ./mt5_fetch/get_price_CSV.py --symbol {symbol} --timeframe {timeframe} --bars {bars}")

# ステップ2: 特徴量生成
os.system(f"python ./feature/add_features.py --symbol {symbol} --timeframe {timeframe}")

# ステップ3: モデル再学習（必要に応じて）
#os.system(f"python ./model/train_model.py --symbol {symbol} --timeframe {timeframe} --atr-multiplier 0.5")

# ステップ4: 最新データで判断
df = pd.read_csv(f"./data/{symbol}_{timeframe}_features_v2.csv", index_col="time", parse_dates=True)
latest = df.iloc[-1]
decision = should_enter_trade(latest, model_path=f"./model/model_lgbm_best_{symbol}_{timeframe}.pkl")

# ログ表示
print("\n📢 [最終トレード判断]")
if decision["enter"]:
    print(f"▶ エントリー: {decision['direction']}（確率: {decision['probability']:.2%}）")

    # === ここから追記 ===
    # 出力先：MT5の Files フォルダに配置される trade_signal.csv
    signal_df = pd.DataFrame([[
        pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),  # 今の時刻を記録（調整可能）
        1,
        decision["direction"],
        decision["probability"],
        decision["tp"],
        decision["sl"]
    ]])
    signal_df.to_csv(f"C:/Users/shang/AppData/Roaming/MetaQuotes/Terminal/A406065E6692A69B94B3E1F7E133A6B2/MQL5/Files/predict_result_batch_{symbol}.csv", index=False, header=False)
    print("✅ predict_result_batch.csv 出力完了")
else:
    print(f"▶ ノーエントリー（確率: {decision['probability']:.2%}）")

