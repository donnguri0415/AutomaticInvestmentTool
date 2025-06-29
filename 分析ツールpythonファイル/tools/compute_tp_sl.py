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


def get_tp_sl(symbol: str, timeframe: str, indir: str = 'data', percentile: float = 75.0):
    """
    指定のCSVからヒストリカルリターン分布を計算し、TP/SLを返す関数
    :param symbol: str 通貨ペアまたは資産名（例: EURUSDm）
    :param timeframe: str 時間足（例: M15）
    :param indir: str CSV格納ディレクトリ
    :param percentile: float パーセンタイル
    :return: tuple (tp, sl) 価格単位
    """
    file_path = os.path.join(indir, f"{symbol}_{timeframe}_features_v2.csv")
    df = pd.read_csv(file_path, parse_dates=['time'], index_col='time')

    # リターン計算
    df['ret'] = df['close'].shift(-1) - df['close']
    df.dropna(subset=['ret'], inplace=True)

    pip_mult = get_pip_multiplier(symbol)
    df['ret_pips'] = df['ret'] / pip_mult

    pos = df.loc[df['ret_pips'] > 0, 'ret_pips']
    neg = df.loc[df['ret_pips'] < 0, 'ret_pips'].abs()

    if pos.empty or neg.empty:
        raise ValueError("データ不足: 上昇または下落のサンプルがありません。")

    tp_pips = np.percentile(pos, percentile)
    sl_pips = np.percentile(neg, percentile)

    tp = tp_pips * pip_mult
    sl = sl_pips * pip_mult
    return tp, sl


def main():
    parser = argparse.ArgumentParser(
        description="ヒストリカルリターン分布からTP/SLを自動設定するスクリプト"
    )
    parser.add_argument('--symbol', type=str, default='EURUSDm', help='通貨ペアまたは資産名')
    parser.add_argument('--timeframe', type=str, default='M15', help='時間足')
    parser.add_argument('--indir', type=str, default='data', help='特徴量CSVのディレクトリ')
    parser.add_argument('--percentile', type=float, default=75.0, help='パーセンタイル')
    args = parser.parse_args()

    tp, sl = get_tp_sl(args.symbol, args.timeframe.upper(), args.indir, args.percentile)
    print(f"✅ {args.symbol} {args.timeframe} の推奨TP: {tp:.5f}, 推奨SL: {sl:.5f}")
    print(f"(約 {tp/get_pip_multiplier(args.symbol):.2f} pips / {sl/get_pip_multiplier(args.symbol):.2f} pips)")

if __name__ == '__main__':
    main()
