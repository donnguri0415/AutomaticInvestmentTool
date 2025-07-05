import os
import glob
import pandas as pd
import argparse
import yaml
import schedule
import time
import subprocess
from datetime import datetime

# 設定ファイル読み込み
parser = argparse.ArgumentParser(
    description="予測・再学習スケジューラ"
)
parser.add_argument('--config', type=str, default='config.yaml', help='設定ファイルパス')
args = parser.parse_args()

with open(args.config, 'r', encoding='utf-8') as f:
    cfg = yaml.safe_load(f)

SYMBOL        = cfg.get('symbol', 'EURUSDm')
TIMEFRAME     = cfg.get('timeframe', 'M15')
PREDICT_INT   = cfg.get('predict_interval', 1)     # 秒
TRAIN_TIME    = cfg.get('train_time', '06:00')
TERMINAL_GUID = cfg.get('terminal_guid', None)

# main.py 呼び出し用コマンドベース
def call_main(mode):
    cmd = [
        'python', 'main.py',
        '--mode', mode,
        '--symbol', SYMBOL,
        '--timeframe', TIMEFRAME
    ]
    if mode == 'predict':
        cmd += ['--bars', str(cfg.get('bars', 3000))]
    else:
        cmd += ['--train-bars', str(cfg.get('train_bars', 100000))]
    return cmd

# 予測ジョブ
def run_predict():
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"🔁 {now} - Predict start ({SYMBOL}, {TIMEFRAME})")
    try:
        subprocess.run(call_main('predict'), check=True)
        print(f"✅ {now} - Predict done\n")
    except subprocess.CalledProcessError as e:
        print(f"❌ {now} - Predict error: {e}\n")

# 再学習ジョブ
def run_train():
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"🔄 {now} - Train start ({SYMBOL}, {TIMEFRAME})")
    try:
        subprocess.run(call_main('train'), check=True)
        print(f"✅ {now} - Train done\n")
    except subprocess.CalledProcessError as e:
        print(f"❌ {now} - Train error: {e}\n")

# スケジュール設定
schedule.every(PREDICT_INT).seconds.do(run_predict)
schedule.every().day.at(TRAIN_TIME).do(run_train)

print(f"📌 Scheduler started: Predict every {PREDICT_INT}s for {SYMBOL}_{TIMEFRAME}, Train daily at {TRAIN_TIME}")

while True:
    schedule.run_pending()
    time.sleep(1)
