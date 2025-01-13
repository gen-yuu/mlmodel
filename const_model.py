import os
import pandas as pd
import light_gbm as lgb_reg
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
data_dir = './data/data_a'
train_file = 'train.csv'
test_file = 'test.csv'
target = 'Inference Time (s)'
const_features = ['Total Frames', 'Width', 'Height', 'Directory Size (MB)', 'Model Name', 'Params']

server_features = [
    'transfer_continuous', 'transfer_roundtrip', 'matrix_conv', 'matrix_convloop', 'matrix_dot',
    'matrix_dotloop', 'matrix_add', 'matrix_addloop'
]

output_csv = "output_model_info.csv"


def main():
    train_csv = os.path.join(data_dir, train_file)
    test_csv = os.path.join(data_dir, test_file)

    train_df = pd.read_csv(train_csv, index_col=0)
    test_df = pd.read_csv(test_csv, index_col=0)

    features = const_features + server_features
    model_info = []

    lgb_result = lgb_reg.lgb_model(train_df, test_df, target, features)

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
