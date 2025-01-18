import os

import pandas as pd

data_dir = './data'


def format_data_loocv(target, data_path):
    """
    Leave-One-Out Cross-Validation (LOOCV) のためのデータを整形する関数。
    
    Parameters:
        target (str): Leave-one-outで除外するサーバー名
        data_path (str): CSVデータファイルのパス
    
    Returns:
        tuple: (train_df, test_df) 
            train_df: 除外されたサーバー以外のデータ
            test_df: 除外されたサーバーのデータ
    """
    # CSVファイルを読み込む
    df = pd.read_csv(data_path, index_col=0)

    # ユニークなサーバー名を取得
    server_list = df['Server Info'].unique()

    # targetサーバーが存在しない場合のエラーチェック
    if target not in server_list:
        print(f"{target} is not in serverlist")
        return -1

    # サーバー名でデータを分割
    train_df = df.query('`Server Info` != @target')  # target以外のサーバー
    test_df = df.query('`Server Info` == @target')  # targetサーバー

    return train_df, test_df
