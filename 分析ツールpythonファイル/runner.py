import os
import glob
import time
import yaml
import pandas as pd
from datetime import datetime

# 内部モジュール
from mt5_fetch.get_price_data import fetch_price_df
from feature.add_features_df import add_features_df
from predict.trade_decision_logic_v2 import init_model, should_enter_trade
from model.train_model import train
from tools.compute_tp_sl import get_tp_sl_from_df


def load_config(path: str = 'config.yaml') -> dict:
    """
    YAML 設定ファイルを読み込み、辞書で返す
    """
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def detect_base_dir(guid: str = None) -> str:
    """
    MT5 Terminal の Files フォルダを GUID 指定または最終更新日時で選択
    """
    appdata = os.path.expanduser(r'~\AppData\Roaming\MetaQuotes\Terminal')
    if guid:
        path = os.path.join(appdata, guid, 'MQL5', 'Files')
        if not os.path.isdir(path):
            raise RuntimeError(f"指定されたターミナル GUID が見つかりません: {path}")
        return path
    # 自動検出
    candidates = glob.glob(os.path.join(appdata, '*', 'MQL5', 'Files'))
    if not candidates:
        raise RuntimeError("Terminal/MQL5/Files フォルダが見つかりません。")
    return max(candidates, key=lambda p: os.path.getmtime(p))


def ensure_model(symbol: str, timeframe: str, train_bars: int) -> str:
    """
    モデルファイルがなければ初回学習を実行し、パスを返す
    """
    model_path = f"model/model_lgbm_best_{symbol}_{timeframe}.pkl"
    if not os.path.exists(model_path):
        print(f"▶ モデルが見つかりません: {model_path} -> 初回学習開始")
        train(symbol=symbol, timeframe=timeframe, bars=train_bars)
        print("✅ 初回学習完了")
    return model_path


def retrain_if_needed(now: datetime, last_date: datetime.date, symbol: str,
                      timeframe: str, train_bars: int, train_time: str) -> datetime.date:
    """
    指定時刻で毎日再学習。更新日を返す
    """
    hhmm = now.strftime('%H:%M')
    if hhmm == train_time and last_date != now.date():
        print(f"🔄 {now} - 再学習 ({train_bars} bars)")
        train(symbol=symbol, timeframe=timeframe, bars=train_bars)
        init_model(model_path)
        print(f"✅ {now} - 再学習完了")
        return now.date()
    return last_date


def run_cycle(symbol: str, timeframe: str, bars: int, base_dir: str):
    """
    1サイクル分の fetch->feature->predict->write を実行し、時間をログ出力
    """
    now = datetime.now()
    t0 = time.perf_counter()
    ohlcv = fetch_price_df(symbol, timeframe, bars)
    t1 = time.perf_counter()
    feats = add_features_df(ohlcv)
    t2 = time.perf_counter()
    latest = feats.iloc[-1]
    t3 = time.perf_counter()
    tp, sl = get_tp_sl_from_df(ohlcv, symbol)
    decision = should_enter_trade(latest, model_path)
    t4 = time.perf_counter()
    # シグナル出力
    df_sig = pd.DataFrame([[
        now.strftime('%Y-%m-%d %H:%M:%S'),
        int(decision['enter']),
        decision['direction'],
        decision['probability'],
        tp,
        sl
    ]])
    out_file = os.path.join(base_dir, f"predict_result_batch_{symbol}.csv")
    df_sig.to_csv(out_file, index=False, header=False)
    t5 = time.perf_counter()

    print(
        f"[{now}] timing: fetch={(t1-t0):.3f}s | feat={(t2-t1):.3f}s | "
        f"extract={(t3-t2):.3f}s | predict={(t4-t3):.3f}s | write={(t5-t4):.3f}s -> {decision}"
    )
    print(f"▶ エントリー: {decision['direction']} (確率: {decision['probability']:.2%})")


def main():
    cfg = load_config()
    symbol, timeframe = cfg['symbol'], cfg['timeframe']
    bars, train_bars = cfg['bars'], cfg['train_bars']
    interval, train_time = cfg['predict_interval'], cfg['train_time']
    guid = cfg.get('terminal_guid')

    base_dir = detect_base_dir(guid)
    print(f"▶ Files directory: {base_dir}")

    global model_path
    model_path = ensure_model(symbol, timeframe, train_bars)
    init_model(model_path)

    last_date = None
    print(f"▶ Start scheduler: every {interval}s, retrain at {train_time}")
    while True:
        now = datetime.now()
        last_date = retrain_if_needed(now, last_date,
                                      symbol, timeframe,
                                      train_bars, train_time)
        run_cycle(symbol, timeframe, bars, base_dir)
        time.sleep(interval)


if __name__ == '__main__':
    main()
