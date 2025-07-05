import os
import pandas as pd
import ta  # テクニカル分析ライブラリ


# 入力されたOHLCV CSVから特徴量を生成し、別のCSVとして出力する関数

def add_features(symbol: str,
                 timeframe: str,
                 in_dir: str = 'data',
                 out_dir: str = 'data') -> str:
    """
    Parameters:
        symbol: 通貨ペア名 (例: 'EURUSDm')
        timeframe: 時間足文字列 (例: 'M15', 'H1')
        in_dir: OHLCV CSVがあるディレクトリ
        out_dir: 特徴量CSVを保存するディレクトリ
    Returns:
        出力ファイルのパス
    """
    tf_str = timeframe.upper()
    input_file = os.path.join(in_dir, f"{symbol}_{tf_str}_ohlcv.csv")
    output_file = os.path.join(out_dir, f"{symbol}_{tf_str}_features_v2.csv")

    # CSV読み込み
    df = pd.read_csv(input_file)
    df['time'] = pd.to_datetime(df['time'])
    df.set_index('time', inplace=True)

    # 基本テクニカル指標
    df['rsi_14'] = ta.momentum.RSIIndicator(close=df['close'], window=14).rsi()
    df['sma_20'] = ta.trend.SMAIndicator(close=df['close'], window=20).sma_indicator()
    df['ema_20'] = ta.trend.EMAIndicator(close=df['close'], window=20).ema_indicator()
    df['volatility'] = df['high'] - df['low']
    df['body'] = (df['close'] - df['open']).abs()
    df['sar'] = ta.trend.PSARIndicator(high=df['high'], low=df['low'], close=df['close']).psar()
    bb = ta.volatility.BollingerBands(close=df['close'], window=20, window_dev=2)
    df['bb_upper'] = bb.bollinger_hband()
    df['bb_middle'] = bb.bollinger_mavg()
    df['bb_lower'] = bb.bollinger_lband()
    df['bb_width'] = df['bb_upper'] - df['bb_lower']
    df['mfi'] = ta.volume.MFIIndicator(
        high=df['high'], low=df['low'], close=df['close'], volume=df['tick_volume'], window=14
    ).money_flow_index()

    # 追加指標
    df['cci'] = ta.trend.CCIIndicator(high=df['high'], low=df['low'], close=df['close'], window=20).cci()
    macd = ta.trend.MACD(close=df['close'])
    df['macd_diff'] = macd.macd_diff()
    adx = ta.trend.ADXIndicator(high=df['high'], low=df['low'], close=df['close'], window=14)
    df['adx'] = adx.adx()
    df['adx_pos'] = adx.adx_pos()
    df['adx_neg'] = adx.adx_neg()
    stoch = ta.momentum.StochasticOscillator(high=df['high'], low=df['low'], close=df['close'], window=14)
    df['stoch_k'] = stoch.stoch()
    df['stoch_d'] = stoch.stoch_signal()
    df['atr'] = ta.volatility.AverageTrueRange(
        high=df['high'], low=df['low'], close=df['close'], window=14
    ).average_true_range()

    # 欠損値除去
    df.dropna(inplace=True)

    # 出力先準備
    os.makedirs(out_dir, exist_ok=True)
    df.to_csv(output_file)
    return output_file


# CLI 実行時のエントリーポイント
if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description="OHLCV CSVから特徴量を生成してCSVに保存する")
    parser.add_argument('--symbol', type=str, required=True, help='通貨ペア (例: EURUSDm)')
    parser.add_argument('--timeframe', type=str, required=True,
                        choices=['M1','M5','M15','M30','H1','H4','D1','W1','MN1'],
                        help='時間足 (例: M15)')
    parser.add_argument('--in-dir', type=str, default='data', help='入力ディレクトリ')
    parser.add_argument('--out-dir', type=str, default='data', help='出力ディレクトリ')
    args = parser.parse_args()

    output_path = add_features(
        symbol=args.symbol,
        timeframe=args.timeframe,
        in_dir=args.in_dir,
        out_dir=args.out_dir
    )
    print(f"✅ 特徴量追加完了: {output_path}")
