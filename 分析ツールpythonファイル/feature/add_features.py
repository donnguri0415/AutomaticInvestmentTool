import pandas as pd
import ta  # テクニカル分析ライブラリ
import argparse
import os

def main():
    # コマンドライン引数の定義
    parser = argparse.ArgumentParser(description="OHLCV CSVから特徴量を生成してCSVに保存するスクリプト")
    parser.add_argument('--symbol',    type=str, default='EURUSDm',
                        help='通貨ペアを指定（例: EURUSDm）')
    parser.add_argument('--timeframe', type=str, default='M15',
                        choices=['M1','M5','M15','M30','H1','H4','D1','W1','MN1'],
                        help='時間足を指定（例: M15, H1）')
    args = parser.parse_args()

    symbol  = args.symbol
    tf_str  = args.timeframe.upper()

    # 入出力ファイルパス
    input_file  = os.path.join("data/", f"{symbol}_{tf_str}_ohlcv.csv")
    output_file = os.path.join("data/", f"{symbol}_{tf_str}_features_v2.csv")

    # データ読み込み
    df = pd.read_csv(input_file)
    df['time'] = pd.to_datetime(df['time'])
    df.set_index('time', inplace=True)

    # 基本指標
    df['rsi_14']     = ta.momentum.RSIIndicator(close=df['close'], window=14).rsi()
    df['sma_20']     = ta.trend.SMAIndicator(close=df['close'], window=20).sma_indicator()
    df['ema_20']     = ta.trend.EMAIndicator(close=df['close'], window=20).ema_indicator()
    df['volatility'] = df['high'] - df['low']
    df['body']       = (df['close'] - df['open']).abs()
    df['sar']        = ta.trend.PSARIndicator(high=df['high'], low=df['low'], close=df['close']).psar()
    bb = ta.volatility.BollingerBands(close=df['close'], window=20, window_dev=2)
    df['bb_upper']   = bb.bollinger_hband()
    df['bb_middle']  = bb.bollinger_mavg()
    df['bb_lower']   = bb.bollinger_lband()
    df['bb_width']   = df['bb_upper'] - df['bb_lower']
    df['mfi']        = ta.volume.MFIIndicator(high=df['high'], low=df['low'], close=df['close'], volume=df['tick_volume']).money_flow_index()

    # 追加指標
    df['cci']        = ta.trend.CCIIndicator(high=df['high'], low=df['low'], close=df['close']).cci()
    macd = ta.trend.MACD(close=df['close'])
    df['macd_diff']  = macd.macd_diff()
    adx = ta.trend.ADXIndicator(high=df['high'], low=df['low'], close=df['close'])
    df['adx']        = adx.adx()
    df['adx_pos']    = adx.adx_pos()
    df['adx_neg']    = adx.adx_neg()
    stoch = ta.momentum.StochasticOscillator(high=df['high'], low=df['low'], close=df['close'])
    df['stoch_k']    = stoch.stoch()
    df['stoch_d']    = stoch.stoch_signal()
    df['atr']        = ta.volatility.AverageTrueRange(high=df['high'], low=df['low'], close=df['close']).average_true_range()

    # 欠損値除去・CSV出力
    df.dropna(inplace=True)
    df.to_csv(output_file)

    print(f"✅ 特徴量追加完了：{output_file}")


if __name__ == '__main__':
    main()
