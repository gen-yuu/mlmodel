import pandas as pd
import lightgbm as lgb
import numpy as np
from sklearn.model_selection import train_test_split


def lgb_model(train_df, test_df, target, features):
    seed = 42  # 乱数シード
    train_df, val_df = train_test_split(train_df, train_size=0.8, random_state=seed)
    # trainデータ
    X_train = train_df[features]
    X_train = pd.get_dummies(X_train, drop_first=True)
    y_train = train_df[target].reset_index(drop=True)
    # validationデータ
    X_val = val_df[features]
    X_val = pd.get_dummies(X_val, drop_first=True)
    y_val = val_df[target].reset_index(drop=True)
    # testデータ
    X_test = test_df[features]
    X_test = pd.get_dummies(X_test, drop_first=True)
    y_test = test_df[target].reset_index(drop=True)

    # Datasetオブジェクトとしてデータを定義
    train_data = lgb.Dataset(X_train, y_train)
    val_data = lgb.Dataset(X_val, y_val)
    test_data = lgb.Dataset(X_test, y_test)

    params = {
        'objective': 'regression',  # 最小化させるべき損失関数
        'metric': 'rmse',  # 学習時に使用する評価指標(early_stoppingの評価指標にも同じ値が使用される)
        'random_state': seed,  # 乱数シード
        'boosting_type': 'gbdt',
        'n_estimators': 1000,  # 最大学習サイクル数。early_stopping使用時は大きな値を入力
        'verbose': -1,  # エラー対応
        'early_stopping_round': 10  # ここでearly_stoppingを指定
    }

    model = lgb.train(
        params,
        train_data,
        valid_sets=[val_data],  # early_stoppingの評価用データ
        valid_names=['valid'],
        callbacks=[lgb.early_stopping(stopping_rounds=10, verbose=True)]  # early_stopping用コールバック関数
    )

    pred_train = model.predict(X_train)
    pred_test = model.predict(X_test)
    pred_val = model.predict(X_val)

    alpha = 1
    deltas = []
    # 平均絶対誤差率(訓練データ)
    for i in range(len(pred_train)):
        deltas.append((y_train[i] - alpha * pred_train[i]) / y_train[i])
    mape_train = np.average(np.abs(deltas))

    deltas = []
    zero_count = any((x <= 0 for x in pred_val))
    print(y_val)
    print(pred_val)
    # 平均絶対誤差(Valデータ)
    i = 0
    d_cnt_val = 0
    while True:
        if zero_count >= 1:
            break
        if i == len(pred_val):
            break
        delta = (y_val[i] - alpha * pred_val[i]) / y_val[i]
        if delta > 0:
            d_cnt_val += 1
        i += 1

    for i in range(len(pred_val)):
        deltas.append((y_val[i] - alpha * pred_val[i]) / y_val[i])
    mape_val = np.average(np.abs(deltas))
    d_rate_val = d_cnt_val / len(pred_val)

    # 平均絶対誤差(testデータ)
    deltas = []
    i = 0
    d_cnt_test = 0
    while True:
        if zero_count >= 1:
            break
        if i == len(pred_test):
            break
        delta = (y_test[i] - alpha * pred_test[i]) / y_test[i]
        if delta > 0:
            d_cnt_test += 1
        i += 1

    for i in range(len(pred_test)):
        deltas.append((y_test[i] - alpha * pred_test[i]) / y_test[i])
    mape_test = np.average(np.abs(deltas))
    d_rate_test = d_cnt_test / len(pred_test)

    if zero_count > 0:
        mape_train = 9999
        mape_val = 9999
        mape_test = 9999
        d_rate_test = 9999
        d_rate_val = 9999

    return {
        'ML': 'lgb',
        'loss': 'l2',
        'Input Num': len(features),
        'Input': features,
        'weight': 1,
        'MAPE train': mape_train,
        'MAPE test': mape_test,
        'Task Incomplete rate (TEST)': d_rate_test,
        'MAPE val': mape_val,
        'Tasl Incomplete rate (Val)': d_rate_val,
    }
