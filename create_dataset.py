import csv
import os
import pandas as pd
from sklearn.model_selection import train_test_split

data_dir = './data'
data_file = 'data.csv'


def main():
    data_csv = os.path.join(data_dir, data_file)
    df = pd.read_csv(data_csv, index_col=0)
    data_a(df, data_dir)
    data_b(df, data_dir)
    return


# 訓練に全サーバーを含むデータセット(8:2)
def data_a(df, parent_dir):
    output_dir = os.path.join(parent_dir, "data_a")
    os.makedirs(output_dir, exist_ok=True)
    server_list = df['Server Info'].unique()
    train_df = pd.DataFrame()
    test_df = pd.DataFrame()
    for server in server_list:
        server_df = df.query('`Server Info` == @server')
        server_train, server_test = train_test_split(
            server_df, test_size=0.8, random_state=0)
        train_df = pd.concat([train_df, server_train])
        test_df = pd.concat([test_df, server_test])
        pass
    train_df.to_csv(output_dir + '/train.csv', mode='w')
    test_df.to_csv(output_dir + '/test.csv', mode='w')
    return 0


'''
 ['13th corei5 - RTX3060 Ti' '13th corei7 - GTX1080'
 '13th corei5 - GTX1650' '1th Xeon Gold - RTX4070' '13th corei7 - RTX3050'
 '13th corei5 - GTX1080' '13th corei5 - RTX4070'
 '13th corei7 - RTX3060 Ti' '13th corei5 - RTX3050'
 '1th Xeon Gold - GTX1080' '9th corei7 - RTX2080 Ti'
 '13th corei7 - RTX4070']
'''


def data_b(df, parent_dir):
    output_dir = os.path.join(parent_dir, "data_b")
    os.makedirs(output_dir, exist_ok=True)
    server_list = df['Server Info'].unique()
    train_df = pd.DataFrame()
    test_df = pd.DataFrame()
    target = server_list[0]
    train_df = df.query('`Server Info` != @target')
    test_df = df.query('`Server Info` == @target')

    train_df.to_csv(output_dir + '/train.csv', mode='w')
    test_df.to_csv(output_dir + '/test.csv', mode='w')
    return 0


if __name__ == "__main__":
    main()
