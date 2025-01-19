import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def calculate_mape(predictions, actuals, alpha=1):
    """
    平均絶対誤差率(MAPE)を計算するヘルパー関数

    Parameters:
        predictions (list or array): モデルの予測値
        actuals (list or array): 実際の値
        alpha (float): 補正用パラメータ（デフォルトは1）

    Returns:
        float: MAPEの計算結果
    """
    deltas = [(actual - alpha * pred) / actual for pred, actual in zip(predictions, actuals)]
    return np.average(np.abs(deltas))


def lgb_model(train_df, test_df, target, const_parameters, variable_parameters):
    """
    LightGBMモデルを学習し、MAPEを計算して結果を返す関数

    Parameters:
        train_df (DataFrame): 学習データ
        test_df (DataFrame): テストデータ
        target (str): 目的変数のカラム名
        const_parameters (list): 定数パラメータ
        variable_parameters (list): 変動パラメータ

    Returns:
        dict: モデルの評価結果を含む辞書
    """

    parameters = const_parameters + variable_parameters
    seed = 42  # 乱数シード

    # 学習データと検証データに分割
    train_df, val_df = train_test_split(train_df, train_size=0.8, random_state=seed)

    # 学習データの準備
    X_train = pd.get_dummies(train_df[parameters], drop_first=True)
    y_train = train_df[target].reset_index(drop=True)

    # 検証データの準備
    X_val = pd.get_dummies(val_df[parameters], drop_first=True)
    y_val = val_df[target].reset_index(drop=True)

    # テストデータの準備
    X_test = pd.get_dummies(test_df[parameters], drop_first=True)
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

    # 結果を辞書形式で返す
    return {
        'ML': 'lgb',
        'loss': 'l2(rmse)',
        'Parameter Num': len(parameters),
        'Const Parameter': const_parameters,
        'Variable Parameter Num': len(variable_parameters),
        'Variable Parameter': variable_parameters,
        'MAPE train (%)': round(mape_train * 100, 5),
        'MAPE val (%)': round(mape_val * 100, 5),
        'MAPE test (%)': round(mape_test * 100, 5)
    }


def train_lgb_model(train_df, target, parameters):
    """
    LightGBMモデルを学習し、学習済みモデルを返す関数

    Parameters:
        train_df (DataFrame): 学習データ
        target (str): 目的変数のカラム名
        parameters (list): 使用する特徴量のリスト

    Returns:
        model: 学習済みLightGBMモデル
        dict: 検証用データセット
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

    return model, {'X_train': X_train, 'y_train': y_train}, {'X_val': X_val, 'y_val': y_val}


def predict_and_evaluate(model, dataset, target, parameters):
    """
    モデルを使って予測し、MAPEを計算する関数

    Parameters:
        model: 学習済みモデル
        dataset (DataFrame): テストデータ
        target (str): 目的変数のカラム名
        parameters (list): 使用する特徴量のリスト

    Returns:
        dict: MAPEの計算結果
    """
    # テストデータの準備
    X_test = pd.get_dummies(dataset[parameters], drop_first=True)
    y_test = dataset[target].reset_index(drop=True)

    # 予測
    predictions = model.predict(X_test)

    # MAPEの計算
    mape = calculate_mape(predictions, y_test)

    return {'MAPE (%)': round(mape * 100, 5)}
