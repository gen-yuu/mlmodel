import itertools
import os

import pandas as pd

import light_gbm as lgb_reg
from data_format import format_data_loocv
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
    総フレーム数, 
モデル
    パラメータ数
サーバー

"""

SERVER_LIST = [
    '13th corei5 - RTX3060 Ti', '13th corei7 - GTX1080', '13th corei5 - GTX1650',
    '1th Xeon Gold - RTX4070', '13th corei7 - RTX3050', '13th corei5 - GTX1080',
    '13th corei5 - RTX4070', '13th corei7 - RTX3060 Ti', '13th corei5 - RTX3050',
    '1th Xeon Gold - GTX1080', '9th corei7 - RTX2080 Ti', '13th corei7 - RTX4070'
]

output_dir = './ml_results'

data_dir = './data'
benchmark_data_file = 'data_benchmark.csv'
server_spec_data_file = 'data_server_spec.csv'

const_features = ['Total Frames', 'Params']

benchmark_features = [
    'transfer_all', 'transfer_continuous', 'transfer_roundtrip', 'matrix_conv', 'matrix_convloop',
    'matrix_dot', 'matrix_dotloop', 'matrix_add', 'matrix_addloop'
]

server_spec_features = [
    'cpu_core', 'cpu_boost_clock(GHz)', 'cpu_thread', 'cpu_cache(MB)', 'gpu_architecture',
    'gpu_core', 'gpu_boost_clock(GHz)', 'VRAM(GB)'
]

target = 'Inference Time (s)'


def main():

    # specの特徴量組み合わせ
    # features_conbs = get_features_conb(server_spec_features)
    # data_path = os.path.join(data_dir, server_spec_data_file)
    # model_info = search_features_conb(features_conbs, data_path)
    # output_csv = "spec_feature_search.csv"
    # output_results_to_csv(model_info, output_csv)

    # benchmarkの特徴量組み合わせ
    features_conbs = get_features_conb(benchmark_features)
    data_path = os.path.join(data_dir, benchmark_data_file)
    model_info = search_features_conb(features_conbs, data_path)
    output_csv = "benchmark_feature_search.csv"
    output_results_to_csv(model_info, output_csv)
    return


def search_features_conb(features_conbs, data_path):
    model_info = []
    for server_features in features_conbs:
        model_info += (loocv(const_features, server_features, data_path))
    return model_info


#特徴量の組み合わせ
def get_features_conb(features):
    min_size = 2
    conbs = []
    for n in range(min_size, len(features) + 1):
        for conb in itertools.combinations(features, n):
            conbs.append(list(conb))  # タプルをリスト型に変換
    return conbs


#leave-one-out法でmodel作成
def loocv(const_features, server_features, data_path):
    print(f"features : {server_features}")
    model_info = []
    for server in SERVER_LIST:
        train_df, test_df = format_data_loocv(server, data_path)
        features = const_features + server_features
        lgb_result = lgb_reg.lgb_model(train_df, test_df, target, features)
        lgb_result["Leave One"] = server
        model_info.append(lgb_result)
        print(f"Leave out server : {server}")
    return model_info


def output_results_to_csv(results, output_csv):
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, output_csv)
    if results:
        df = pd.DataFrame(results)
        df.to_csv(output_path, index=True)
        print(f"Model information has been written to {output_csv}")
    else:
        print("No models.")
    return 0


if __name__ == "__main__":
    main()
