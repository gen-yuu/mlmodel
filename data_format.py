import os
import pandas as pd

data_dir = './data'
data_file = 'data.csv'


def one_leave_out(i):
    data_csv = os.path.join(data_dir, data_file)
    df = pd.read_csv(data_csv, index_col=0)
    server_list = df['Server Info'].unique()
    train_df = pd.DataFrame()
    test_df = pd.DataFrame()
    target = server_list[i]
    train_df = df.query('`Server Info` != @target')
    test_df = df.query('`Server Info` == @target')

    return train_df, test_df, target
