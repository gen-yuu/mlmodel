import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def calculate_mape(predictions, actuals, alpha=1):
    """平均絶対誤差率(MAPE)を計算するヘルパー関数"""
    deltas = [(actual - alpha * pred) / actual for pred, actual in zip(predictions, actuals)]
    return np.average(np.abs(deltas))


def lgb_model(train_df, test_df, target, features):
    """
    LightGBMモデルを学習し、MAPEを計算して結果を返す関数

    Parameters:
        train_df (DataFrame): 学習データ
        test_df (DataFrame): テストデータ
        target (str): 目的変数のカラム名
        features (list): 特徴量のリスト

    Returns:
        dict: モデルの評価結果を含む辞書
    """
    seed = 42  # 乱数シード
    train_df, val_df = train_test_split(train_df, train_size=0.8, random_state=seed)

    # 学習データの準備
    X_train = pd.get_dummies(train_df[features], drop_first=True)
    y_train = train_df[target].reset_index(drop=True)

    # 検証データの準備
    X_val = pd.get_dummies(val_df[features], drop_first=True)
    y_val = val_df[target].reset_index(drop=True)

    # テストデータの準備
    X_test = pd.get_dummies(test_df[features], drop_first=True)
    y_test = test_df[target].reset_index(drop=True)

    # Datasetオブジェクトとしてデータを定義
    train_data = lgb.Dataset(X_train, y_train)
    val_data = lgb.Dataset(X_val, y_val)

    # 学習パラメータ
    params = {
        'objective': 'regression',  # 最小化させるべき損失関数
        'metric': 'rmse',  # 評価指標
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

    # 予測
    pred_train = model.predict(X_train)
    pred_val = model.predict(X_val)
    pred_test = model.predict(X_test)

    # MAPEの計算
    mape_train = calculate_mape(pred_train, y_train)
    mape_val = calculate_mape(pred_val, y_val)
    mape_test = calculate_mape(pred_test, y_test)

    return {
        'ML': 'lgb',
        'loss': 'l2(rmse)',
        'Input Num': len(features),
        'Input': features,
        'MAPE train (%)': round(mape_train * 100, 5),
        'MAPE val (%)': round(mape_val * 100, 5),
        'MAPE test (%)': round(mape_test * 100, 5)
    }
