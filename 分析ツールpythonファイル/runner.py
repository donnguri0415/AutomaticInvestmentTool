import schedule
import time
import subprocess
from datetime import datetime

# --- 設定 ---
SYMBOL    = "ETHUSDm"  # ここで通貨ペアを設定
TIMEFRAME = "H1"       # ここで時間足を設定（例: M15, H1 など）

# 1秒ごとの予測と、毎日06:00のモデル再学習をスケジュール

def run_predict():
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"🔁 {now} - Predict start (symbol={SYMBOL}, timeframe={TIMEFRAME})")
    try:
        cmd = [
            "python", "main.py",
            "--mode", "predict",
            "--symbol", SYMBOL,
            "--timeframe", TIMEFRAME
        ]
        subprocess.run(cmd, check=True)
        print(f"✅ {now} - Predict done\n")
    except subprocess.CalledProcessError as e:
        print(f"❌ {now} - Predict error: {e}\n")


def run_train():
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"🔄 {now} - Train start")
    try:
        cmd = [
            "python", "main.py",
            "--mode", "train",
            "--symbol", SYMBOL,
            "--timeframe", TIMEFRAME
        ]
        subprocess.run(cmd, check=True)
        print(f"✅ {now} - Train done\n")
    except subprocess.CalledProcessError as e:
        print(f"❌ {now} - Train error: {e}\n")

# スケジュール設定
schedule.every(1).seconds.do(run_predict)
schedule.every().day.at("06:00").do(run_train)

print(f"📌 Scheduler started: Predict every 1s for {SYMBOL}_{TIMEFRAME}, Train daily at 06:00")

while True:
    schedule.run_pending()
    time.sleep(1)
