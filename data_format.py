import os

import pandas as pd

data_dir = './data'


def format_data_loocv(target, data_path):
    df = pd.read_csv(data_path, index_col=0)
    server_list = df['Server Info'].unique()

    if target not in server_list:
        print(f"{target} is not in serverlist")
        return -1

    train_df = pd.DataFrame()
    test_df = pd.DataFrame()
    train_df = df.query('`Server Info` != @target')
    test_df = df.query('`Server Info` == @target')

    return train_df, test_df
