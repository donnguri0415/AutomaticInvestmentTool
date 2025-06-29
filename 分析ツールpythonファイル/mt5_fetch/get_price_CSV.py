import MetaTrader5 as mt5
import pandas as pd
from datetime import datetime
import argparse
import os

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

symbol    = args.symbol
tf_str    = args.timeframe.upper()
bars      = args.bars


# 時間足文字列からMT5の定数へマッピング
tf_map = {
    'M1':  mt5.TIMEFRAME_M1,
    'M5':  mt5.TIMEFRAME_M5,
    'M15': mt5.TIMEFRAME_M15,
    'M30': mt5.TIMEFRAME_M30,
    'H1':  mt5.TIMEFRAME_H1,
    'H4':  mt5.TIMEFRAME_H4,
    'D1':  mt5.TIMEFRAME_D1,
    'W1':  mt5.TIMEFRAME_W1,
    'MN1': mt5.TIMEFRAME_MN1,
}
if tf_str not in tf_map:
    print(f"❌ 未サポートの時間足です: {tf_str}")
    exit(1)
timeframe = tf_map[tf_str]

# MT5初期化
if not mt5.initialize():
    print("MT5初期化失敗:", mt5.last_error())
    quit()

# データ取得
rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, bars)
if rates is None or len(rates) == 0:
    print("❌ データ取得失敗。チャートを左にスクロールして履歴を読み込んでから再試行してください。")
    mt5.shutdown()
    quit()
mt5.shutdown()

# DataFrame化とタイムスタンプ変換
df = pd.DataFrame(rates)
df['time'] = pd.to_datetime(df['time'], unit='s')

# 出力ディレクトリ確認
filename = f"{symbol}_{tf_str}_ohlcv.csv"
filepath = os.path.join("data/", filename)

# CSV保存
df.to_csv(filepath, index=False)
print(f"✅ データ取得成功: {filepath}")
