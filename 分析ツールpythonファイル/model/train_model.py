import pandas as pd
import numpy as np
import lightgbm as lgb
import joblib
import argparse
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt

# デフォルトのATRラベル閾値倍率
LABEL_ATR_MULTIPLIER = 0.5

# 特徴量カラムリスト
FEATURE_COLS = [
    'volatility', 'atr', 'adx', 'macd_diff',
    'bb_width', 'body', 'stoch_k', 'adx_neg',
    'adx_pos', 'mfi'
]

# 時間足ごとに固定閾値を残す場合は使えるが、動的ATR閾値利用を推奨
threshold_map = {
    'EURUSDm': 0.0003,
    'ETHUSDm': 5.0,
    'BTCUSDm': 50,
    'USDJPYm': 0.1
}

#----------------------------------------------------------------------------+
#| モデル学習スクリプト                                                      |
#----------------------------------------------------------------------------+
def main():
    parser = argparse.ArgumentParser(
        description="特徴量CSVからモデルを学習・保存するスクリプト"
    )
    parser.add_argument('--symbol',    type=str, default='EURUSDm',
                        help='通貨ペア（例: EURUSDm）')
    parser.add_argument('--timeframe', type=str, default='M15',
                        choices=['M1','M5','M15','M30','H1','H4','D1','W1','MN1'],
                        help='時間足（例: M15, H1）')
    parser.add_argument('--indir',     type=str, default='data',
                        help='特徴量CSVディレクトリ')
    parser.add_argument('--modeldir',  type=str, default='model',
                        help='モデル保存ディレクトリ')
    parser.add_argument('--test-size', type=float, default=0.2,
                        help='訓練/テスト分割比率')
    parser.add_argument('--random-state', type=int, default=42,
                        help='乱数シード')
    parser.add_argument('--atr-multiplier', type=float,
                        default=LABEL_ATR_MULTIPLIER,
                        help='ATR閾値倍率')
    args = parser.parse_args()

    symbol = args.symbol
    tf_str = args.timeframe.upper()
    indir = args.indir
    model_dir = args.modeldir
    os.makedirs(model_dir, exist_ok=True)

    # データ読み込み
    input_file = os.path.join(indir, f"{symbol}_{tf_str}_features_v2.csv")
    df = pd.read_csv(input_file, index_col='time', parse_dates=True)

    # ラベル付け：ATRベースの動的閾値
    threshold_px = df['atr'] * args.atr_multiplier
    df['future'] = df['close'].shift(-1) - df['close']
    # 买い: 1, 売り: 0, 中立: NaN
    df['target'] = np.where(
        df['future'] >  threshold_px, 1,
        np.where(df['future'] < -threshold_px, 0, np.nan)
    )
    df.dropna(subset=['target'], inplace=True)
    df['target'] = df['target'].astype(int)

    # 特徴量とターゲット
    X = df[FEATURE_COLS]
    y = df['target']

    # 訓練/テスト分割
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=args.test_size,
        random_state=args.random_state,
        stratify=y
    )

    # モデル学習
    model = lgb.LGBMClassifier(
        class_weight='balanced',
        learning_rate=0.1,
        max_depth=5,
        n_estimators=200,
        num_leaves=31,
        random_state=args.random_state,
        verbose=-1
    )
    model.fit(X_train, y_train)

    # モデル保存
    model_file = os.path.join(model_dir, f"model_lgbm_best_{symbol}_{tf_str}.pkl")
    joblib.dump(model, model_file)
    print(f"✅ モデル保存完了: {model_file}")

    # モデル評価
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"✅ モデル精度（Accuracy）: {acc:.3f}")
    print(classification_report(y_test, y_pred, digits=3))

    # 特徴量重要度の可視化
    importances = pd.Series(model.feature_importances_, index=FEATURE_COLS)
    plt.figure()
    importances.sort_values().plot(kind='barh', title='Feature Importances')
    plt.tight_layout()
    fi_file = os.path.join(model_dir, f"feature_importance_{symbol}_{tf_str}.png")
    plt.savefig(fi_file)
    print(f"📈 特徴量重要度グラフ保存: {fi_file}")

if __name__ == '__main__':
    main()
