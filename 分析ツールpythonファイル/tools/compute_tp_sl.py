import pandas as pd
import argparse
import os
import numpy as np

def get_pip_multiplier(symbol: str) -> float:
    """
    各シンボルに応じたピップ単位を返す
    FX通貨ペア: 0.0001
    JPYペア: 0.01
    XAUUSD: 0.001
    BTCUSD: 0.01
    その他はデフォルトで0.0001
    """
    s = symbol.upper()
    if 'JPY' in s:
        return 0.01
    if 'XAU' in s:
        return 0.001
    if 'BTC' in s:
        return 0.01
    return 0.0001


def get_tp_sl_from_df(feat_df: pd.DataFrame,
                      symbol: str,
                      percentile: float = 100.0) -> tuple[float, float]:
    """
    現在の特徴量 DataFrame を基に TP/SL を計算して返す

    :param feat_df: pandas.DataFrame 特徴量を含むデータフレーム（'close' 列必須）
    :param symbol: str 通貨ペア名など
    :param percentile: float パーセンタイル（0-100）
    :return: (tp, sl) 価格単位
    """
    # リターンを計算
    close = feat_df['close'].values
    pip_mult = get_pip_multiplier(symbol)
    # 次足との差分（最後の行は NaN）
    ret = np.empty_like(close)
    ret[:-1] = close[1:] - close[:-1]
    ret[-1] = np.nan

    ret_pips = ret / pip_mult
    # 正負に分ける
    pos = ret_pips[:-1][ret_pips[:-1] > 0]
    neg = np.abs(ret_pips[:-1][ret_pips[:-1] < 0])

    if pos.size == 0 or neg.size == 0:
        raise ValueError("データ不足: 上昇または下落のサンプルがありません。")

    # パーセンタイル計算
    tp_pips = np.percentile(pos, percentile)
    sl_pips = np.percentile(neg, percentile)

    # pips→価格単位
    return float(tp_pips * pip_mult), float(sl_pips * pip_mult)

