import os
import pandas as pd
import ta  # テクニカル分析ライブラリ


# 入力されたOHLCV CSVから特徴量を生成し、別のCSVとして出力する関数

def add_features_df(ohlcv: pd.DataFrame) -> pd.DataFrame:
    """
    Parameters:
        OHLCV DataFrame → 特徴量を付与した DataFrame を返す
    Returns:
        出力ファイルのパス
    """

    # CSV読み込み
    df = ohlcv.copy()

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

    return df
