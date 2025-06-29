import schedule
import time
import subprocess
from datetime import datetime




def run_main():
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"🔁 {now} - 実行開始")
    try:
        # Pythonのmain.pyを実行
        subprocess.run(["python", "main.py"], check=True)
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"✅ {now} - 実行成功\n")
    except subprocess.CalledProcessError as e:
        print(f"❌ {now} - エラー発生: {e}\n")

# 毎分実行
schedule.every(1).seconds.do(run_main)

print("📌 スケジューラー起動中（毎分実行）...\n")

while True:
    schedule.run_pending()
    time.sleep(1)
