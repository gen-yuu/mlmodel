import itertools
import os

import pandas as pd

import light_gbm as lgb_reg
from mldata_format import format_data_loocv
"""
入力候補
['Directory Name', 'Total Frames', 'Width', 'Height', 'Pixels',
 'Directory Size (MB)', 'Server Info', 'Model Name', 'Params',
 'Categories', 'Inference Time (s)', 'transfer_all',
 'transfer_continuous', 'transfer_roundtrip', 'matrix_conv',
 'matrix_convloop', 'matrix_dot', 'matrix_dotloop', 'matrix_add',
 'matrix_addloop']
"""
# サーバーリスト
SERVER_LIST = [
    '13th corei5 - RTX3060 Ti', '13th corei7 - GTX1080', '13th corei5 - GTX1650',
    '1th Xeon Gold - RTX4070', '13th corei7 - RTX3050', '13th corei5 - GTX1080',
    '13th corei5 - RTX4070', '13th corei7 - RTX3060 Ti', '13th corei5 - RTX3050',
    '1th Xeon Gold - GTX1080', '9th corei7 - RTX2080 Ti', '13th corei7 - RTX4070'
]

# データの出力先
output_dir = './ml_results'
data_dir = './data'

# データファイル
benchmark_data_file = 'data_benchmark.csv'
server_spec_data_file = 'data_server_spec.csv'

# 定数特徴量
const_parameters = ['Total Frames', 'Directory Size (MB)', 'Params']

# ベンチマーク特徴量
benchmark_parameters = [
    'transfer_all', 'transfer_continuous', 'transfer_roundtrip', 'matrix_conv', 'matrix_convloop',
    'matrix_dot', 'matrix_dotloop', 'matrix_add', 'matrix_addloop'
]
# スペック特徴量
server_spec_parameters = [
    'cpu_core', 'cpu_boost_clock(GHz)', 'cpu_thread', 'cpu_cache(MB)', 'gpu_architecture',
    'gpu_core', 'gpu_boost_clock(GHz)', 'VRAM(GB)'
]

# ターゲット変数
target = 'Inference Time (s)'


def main():
    """
    メイン処理: ベンチマークデータの特徴量の組み合わせを評価し、結果をCSVに保存する。
    """
    #specの特徴量組み合わせ
    parameters_conbs = get_parameters_conb(server_spec_parameters, min_size=2)
    data_path = os.path.join(data_dir, server_spec_data_file)
    model_info = search_parameters_conb(parameters_conbs, data_path)
    output_csv = "original_spec_parameter_loocv.csv"
    output_results_to_csv(model_info, output_csv)

    # ベンチマークデータの特徴量組み合わせ
    parameters_conbs = get_parameters_conb(benchmark_parameters, min_size=2)
    data_path = os.path.join(data_dir, benchmark_data_file)
    model_info = search_parameters_conb(parameters_conbs, data_path)
    output_csv = "original_benchmark_parameter_loocv.csv"
    output_results_to_csv(model_info, output_csv)


def search_parameters_conb(parameters_conbs, data_path):
    """
    特徴量の組み合わせごとに、leave-one-out交差検証を行う。
    
    Args:
        parameters_conbs (list): 特徴量の組み合わせリスト
        data_path (str): データのパス

    Returns:
        list: モデルの評価結果
    """
    model_info = []
    for server_parameters in parameters_conbs:
        model_info.extend(loocv(const_parameters, server_parameters, data_path))  # リストを展開して追加
    return model_info


def get_parameters_conb(parameters, min_size=2):
    """
    特徴量の組み合わせを生成する。
    
    Args:
        parameters (list): 特徴量のリスト
        min_size (int): 組み合わせの最小サイズ（デフォルトは2）
    
    Returns:
        list: 特徴量の組み合わせリスト
    """
    conbs = [
        list(conb)
        for n in range(min_size,
                       len(parameters) + 1)
        for conb in itertools.combinations(parameters, n)
    ]
    return conbs


def loocv(const_parameters, server_parameters, data_path):
    """
    Leave-One-Out交差検証を実行し、各サーバーについてモデルの評価結果を取得する。
    
    Args:
        const_parameters (list): 定数特徴量
        server_parameters (list): サーバーに関する特徴量
        data_path (str): データのパス

    Returns:
        list: サーバーごとのモデル評価結果
    """
    model_info = []
    for server in SERVER_LIST:
        print(f"parameters : {server_parameters}, Leave out server: {server}")
        train_df, test_df = format_data_loocv(server, data_path)
        parameters = const_parameters + server_parameters
        #lightGBM
        #訓練データが8:2でtrain:valに分割される
        lgb_model, loss, train_df, val_df = lgb_reg.train_lgb_model(train_df, target, parameters)
        mape_train = lgb_reg.predict_and_evaluate(lgb_model, train_df, target, parameters)
        mape_val = lgb_reg.predict_and_evaluate(lgb_model, val_df, target, parameters)
        mape_test = lgb_reg.predict_and_evaluate(lgb_model, test_df, target, parameters)
        lgb_result = {
            'ML': 'lgb',
            'loss': loss,
            'Parameter Num': len(parameters),
            'Const Parameter': const_parameters,
            'Variable Parameter Num': len(server_parameters),
            'Variable Parameter': server_parameters,
            'MAPE train (%)': mape_train,
            'MAPE val (%)': mape_val,
            'MAPE test (%)': mape_test,
            'Leave One': server
        }
        model_info.append(lgb_result)
    return model_info


def output_results_to_csv(results, output_csv):
    """
    モデルの評価結果をCSVファイルに出力する。
    
    Args:
        results (list): モデル評価結果のリスト
        output_csv (str): 出力するCSVファイル名
    """
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, output_csv)

    if results:
        df = pd.DataFrame(results)
        df.to_csv(output_path, index=True)
        print(f"Model information has been written to {output_csv}")
    else:
        print("No models found.")


if __name__ == "__main__":
    main()
