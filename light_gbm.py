import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def calculate_mape(predictions, actuals, alpha=1):
    """
    平均絶対誤差率(MAPE)を計算するヘルパー関数

    Args:
        predictions (list or array): モデルの予測値
        actuals (list or array): 実際の値
        alpha (float): 補正用パラメータ（デフォルトは1）

    Returns:
        float: MAPEの計算結果
    """
    deltas = [(actual - alpha * pred) / actual for pred, actual in zip(predictions, actuals)]
    return np.average(np.abs(deltas))


def train_lgb_model(train_df, target, parameters):
    """
    LightGBMモデルを学習し、学習済みモデルを返す関数

    Args:
        train_df (DataFrame): 学習データ
        target (str): 目的変数のカラム名
        parameters (list): 使用する特徴量のリスト

    Returns:
        model: 学習済みLightGBMモデル
        loss: 用いたloss関数
        df: trainデータ
        df: validationデータ
    """
    seed = 42  # 乱数シード

    # 学習データと検証データに分割
    train_df, val_df = train_test_split(train_df, train_size=0.8, random_state=seed)

    # 学習データの準備
    X_train = pd.get_dummies(train_df[parameters], drop_first=True)
    y_train = train_df[target].reset_index(drop=True)

    # 検証データの準備
    X_val = pd.get_dummies(val_df[parameters], drop_first=True)
    y_val = val_df[target].reset_index(drop=True)

    # Datasetオブジェクトとしてデータを定義
    train_data = lgb.Dataset(X_train, y_train)
    val_data = lgb.Dataset(X_val, y_val)

    # 学習パラメータ
    loss = 'rmse'
    params = {
        'objective': 'regression',  # 最小化させるべき損失関数
        'metric': loss,  # 評価指標
        'random_state': seed,  # 乱数シード
        'boosting_type': 'gbdt',
        'verbose': -1,  # エラー対応
    }

    # モデルの学習
    model = lgb.train(
        params,
        train_data,
        valid_sets=[val_data],  # early_stoppingの評価用データ
        valid_names=['valid'],
        num_boost_round=10000,
        callbacks=[lgb.early_stopping(stopping_rounds=10)],
    )

    return model, loss, train_df, val_df


def predict_and_evaluate(model, dataset, target, parameters):
    """
    モデルを使って予測し、MAPEを計算する関数

    Args:
        model: 学習済みモデル
        dataset (DataFrame): 予測するデータ
        target (str): 目的変数のカラム名
        parameters (list): 使用する特徴量のリスト

    Returns:
        float: MAPEの計算結果(%)
    """
    # テストデータの準備
    X = pd.get_dummies(dataset[parameters], drop_first=True)
    y = dataset[target].reset_index(drop=True)

    # 予測
    predictions = model.predict(X)

    # MAPEの計算
    mape = calculate_mape(predictions, y)

    return round(mape * 100, 5)
