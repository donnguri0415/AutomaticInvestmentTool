import os
import pandas as pd
import numpy as np
import lightgbm as lgb
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import argparse

# デフォルトのATRラベル閾値倍率
DEFAULT_ATR_MULTIPLIER = 0.5
# デフォルトの訓練/テスト分割
DEFAULT_TEST_SIZE = 0.2
DEFAULT_RANDOM_STATE = 42
# 特徴量カラムリスト
FEATURE_COLS = [
    'volatility', 'atr', 'adx', 'macd_diff',
    'bb_width', 'body', 'stoch_k', 'adx_neg',
    'adx_pos', 'mfi'
]


def train(symbol: str,
          timeframe: str,
          bars: int,
          indir: str = 'data',
          modeldir: str = 'model',
          atr_multiplier: float = DEFAULT_ATR_MULTIPLIER,
          test_size: float = DEFAULT_TEST_SIZE,
          random_state: int = DEFAULT_RANDOM_STATE) -> str:
    """
    モデルを学習し、ファイルへ保存する関数。

    :param symbol: 通貨ペア名
    :param timeframe: 時間足
    :param bars: 使用する特徴量CSVの件数（インディレクトリから最新bars行を使用）
    :param indir: 特徴量CSVが格納されたディレクトリパス
    :param modeldir: モデル保存先ディレクトリ
    :param atr_multiplier: ATRベースのラベル閾値倍率
    :param test_size: 訓練/テスト分割比率
    :param random_state: 乱数シード
    :return: 保存したモデルファイルパス
    """
    os.makedirs(modeldir, exist_ok=True)
    tf = timeframe.upper()
    # データ読み込み
    input_file = os.path.join(indir, f"{symbol}_{tf}_features_v2.csv")
    df = pd.read_csv(input_file, index_col='time', parse_dates=True)
    # 最新barsだけ使用
    if bars and bars < len(df):
        df = df.iloc[-bars:]

    # ラベル作成
    threshold_px = df['atr'] * atr_multiplier
    future = df['close'].shift(-1) - df['close']
    df['target'] = np.where(
        future > threshold_px, 1,
        np.where(future < -threshold_px, 0, np.nan)
    )
    df.dropna(subset=['target'], inplace=True)
    df['target'] = df['target'].astype(int)

    # 特徴量とターゲット
    X = df[FEATURE_COLS]
    y = df['target']

    # 訓練/テスト分割
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )

    # モデル学習
    model = lgb.LGBMClassifier(
        class_weight='balanced',
        learning_rate=0.1,
        max_depth=5,
        n_estimators=200,
        num_leaves=31,
        random_state=random_state,
        verbose=-1
    )
    model.fit(X_train, y_train)

    # モデル保存
    model_file = os.path.join(modeldir, f"model_lgbm_best_{symbol}_{tf}.pkl")
    joblib.dump(model, model_file)
    print(f"✅ モデル保存完了: {model_file}")

    # モデル評価
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"✅ モデル精度 (Accuracy): {acc:.3f}")
    print(classification_report(y_test, y_pred, digits=3))

    # 特徴量重要度可視化
    importances = pd.Series(model.feature_importances_, index=FEATURE_COLS)
    plt.figure()
    importances.sort_values().plot(kind='barh', title='Feature Importances')
    plt.tight_layout()
    fi_file = os.path.join(modeldir, f"feature_importance_{symbol}_{tf}.png")
    plt.savefig(fi_file)
    print(f"📈 特徴量重要度グラフ保存: {fi_file}")

    return model_file


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="特徴量CSVからモデルを学習・保存する"
    )
    parser.add_argument('--symbol', type=str, required=True, help='通貨ペア (例: EURUSDm)')
    parser.add_argument('--timeframe', type=str, required=True,
                        choices=['M1','M5','M15','M30','H1','H4','D1','W1','MN1'],
                        help='時間足 (例: M15)')
    parser.add_argument('--bars', type=int, default=None,
                        help='使用する最新のバー数 (デフォルト: 全件)')
    parser.add_argument('--indir', type=str, default='data', help='特徴量CSVディレクトリ')
    parser.add_argument('--modeldir', type=str, default='model', help='モデル保存ディレクトリ')
    parser.add_argument('--test-size', type=float, default=DEFAULT_TEST_SIZE,
                        help='訓練/テスト分割比率')
    parser.add_argument('--random-state', type=int, default=DEFAULT_RANDOM_STATE,
                        help='乱数シード')
    parser.add_argument('--atr-multiplier', type=float, default=DEFAULT_ATR_MULTIPLIER,
                        help='ATR閾値倍率')
    args = parser.parse_args()

    train(
        symbol=args.symbol,
        timeframe=args.timeframe,
        bars=args.bars,
        indir=args.indir,
        modeldir=args.modeldir,
        atr_multiplier=args.atr_multiplier,
        test_size=args.test_size,
        random_state=args.random_state
    )
