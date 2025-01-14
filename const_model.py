import os
import pandas as pd
import light_gbm as lgb_reg
import itertools
from data_format import one_leave_out
"""
入力候補
['Directory Name', 'Total Frames', 'Width', 'Height', 'Pixels',
 'Directory Size (MB)', 'Server Info', 'Model Name', 'Params',
 'Categories', 'Inference Time (s)', 'transfer_all',
 'transfer_continuous', 'transfer_roundtrip', 'matrix_conv',
 'matrix_convloop', 'matrix_dot', 'matrix_dotloop', 'matrix_add',
 'matrix_addloop']
"""
"""
case 1 
動画
    総フレーム数, 動画サイズ, 高さ, 幅
モデル
    アーキテクチャ名,パラメータ数
サーバー

"""

"""
case 2 
動画
    総フレーム数
モデル
    パラメータ数
サーバー

"""
SERVER_NUM = 12
datas = ["data_a", "data_b"]
parent_dir = './data'
train_file = 'train.csv'
test_file = 'test.csv'
target = 'Inference Time (s)'

# const_features = ['Total Frames', 'Width', 'Height',
#                   'Directory Size (MB)', 'Model Name', 'Params']
const_features = ['Total Frames',
                  'Params']

server_features = [
    'transfer_all', 'transfer_continuous', 'transfer_roundtrip', 'matrix_conv', 'matrix_convloop', 'matrix_dot',
    'matrix_dotloop', 'matrix_add', 'matrix_addloop'
]

# top_features = [['Total Frames', 'Params', 'transfer_continuous', 'matrix_conv', 'matrix_dotloop'],
#                 ['Total Frames', 'Params', 'transfer_continuous',
#                     'matrix_conv', 'matrix_convloop', 'matrix_dotloop'],
#                 ['Total Frames', 'Params', 'transfer_all',
#                     'matrix_add', 'matrix_addloop'],
#                 ['Total Frames', 'Params', 'transfer_continuous',
#                  'transfer_roundtrip', 'matrix_convloop', 'matrix_dotloop'],
#                 ['Total Frames', 'Params', 'transfer_continuous', 'matrix_add', 'matrix_addloop']]
output_csv = "output_model_info.csv"


def main():
    model_info = []

    # # server特徴量で回す
    # serv_conb = []
    # for n in range(3, len(server_features)+1):
    #     for conb in itertools.combinations(server_features, n):
    #         serv_conb.append(list(conb))  # タプルをリスト型に変換

    # for conb_features in serv_conb:
    #     data_dir = os.path.join(parent_dir, "data_a")
    #     train_csv = os.path.join(data_dir, train_file)
    #     test_csv = os.path.join(data_dir, test_file)
    #     train_df = pd.read_csv(train_csv, index_col=0)
    #     test_df = pd.read_csv(test_csv, index_col=0)
    #     features = const_features + conb_features
    #     lgb_result = lgb_reg.lgb_model(train_df, test_df, target, features)
    #     model_info.append(lgb_result)

    # # dataで回す
    # for sub_dir in datas:
    #     data_dir = os.path.join(parent_dir, sub_dir)
    #     train_csv = os.path.join(data_dir, train_file)
    #     test_csv = os.path.join(data_dir, test_file)

    #     train_df = pd.read_csv(train_csv, index_col=0)
    #     test_df = pd.read_csv(test_csv, index_col=0)

    #     features = const_features + server_features

    #     lgb_result = lgb_reg.lgb_model(train_df, test_df, target, features)

    #     model_info.append(lgb_result)

    for i in range(SERVER_NUM):
        train_df, test_df, leave_server = one_leave_out(i)
        features = const_features + server_features
        lgb_result = lgb_reg.lgb_model(train_df, test_df, target, features)
        lgb_result["Leave One"] = leave_server
        model_info.append(lgb_result)

    if model_info:
        df = pd.DataFrame(model_info)
        df.to_csv(output_csv, index=True)
        print(f"Model information has been written to {output_csv}")
    else:
        print("No models.")

    return


if __name__ == "__main__":
    main()
