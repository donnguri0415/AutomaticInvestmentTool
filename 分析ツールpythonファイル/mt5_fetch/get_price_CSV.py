import MetaTrader5 as mt5
import pandas as pd
from datetime import datetime
import argparse
import os

# タイムフレームの文字列から MT5 定数へのマッピング
TF_MAP = {
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


def fetch_price_csv(symbol: str,
                    timeframe_str: str,
                    bars: int,
                    out_dir: str = 'data') -> str:
    """
    MT5 からチャートデータを取得し、CSV に保存する。戻り値は保存ファイルパス。
    """
    tf = timeframe_str.upper()
    if tf not in TF_MAP:
        raise ValueError(f"Unsupported timeframe: {tf}")
    timeframe = TF_MAP[tf]

    # MT5 初期化
    if not mt5.initialize():
        raise RuntimeError(f"MT5 initialize failed: {mt5.last_error()}")

    # データ取得
    rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, bars)
    mt5.shutdown()
    if rates is None or len(rates) == 0:
        raise RuntimeError("No data fetched. Ensure history is loaded in MT5.")

    # DataFrame 化
    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')

    # 出力先準備
    os.makedirs(out_dir, exist_ok=True)
    filename = f"{symbol}_{tf}_ohlcv.csv"
    filepath = os.path.join(out_dir, filename)

    # CSV 保存
    df.to_csv(filepath, index=False)
    return filepath


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="MT5からチャートデータを取得してCSVに保存する"
    )
    parser.add_argument('--symbol',    type=str, required=True,
                        help='通貨ペア（例: EURUSDm）')
    parser.add_argument('--timeframe', type=str, required=True,
                        choices=list(TF_MAP.keys()),
                        help='時間足（例: M15, H1）')
    parser.add_argument('--bars',      type=int, required=True,
                        help='取得バー本数')
    parser.add_argument('--out-dir',   type=str, default='data',
                        help='出力ディレクトリ')
    args = parser.parse_args()

    path = fetch_price_csv(
        symbol=args.symbol,
        timeframe_str=args.timeframe,
        bars=args.bars,
        out_dir=args.out_dir
    )
    print(f"✅ データ取得成功: {path}")
