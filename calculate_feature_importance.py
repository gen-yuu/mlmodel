import ast
import itertools
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
# サーバーリスト
SERVER_LIST = [
    '13th corei5 - RTX3060 Ti', '13th corei7 - GTX1080', '13th corei5 - GTX1650',
    '1th Xeon Gold - RTX4070', '13th corei7 - RTX3050', '13th corei5 - GTX1080',
    '13th corei5 - RTX4070', '13th corei7 - RTX3060 Ti', '13th corei5 - RTX3050',
    '1th Xeon Gold - GTX1080', '9th corei7 - RTX2080 Ti', '13th corei7 - RTX4070'
]

# データの出力先
output_dir = './mldata_analyze'
data_dir = './mldata_analyze'

# データファイル
# transferとoperationが両方含まれる
csv_file = 'filtered_by_variable_parameters_mape_test.csv'

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
    csv_path = os.path.join(data_dir, csv_file)

    variable_parameter_list = get_variable_parameter_list(csv_path)
    print(variable_parameter_list)
    return 0


def get_variable_parameter_list(data_path):
    df = pd.read_csv(data_path)
    # Variable Parameterカラムからリストを取得し、各行をリストとして変換
    variable_parameter_list = df['Variable Parameter'].apply(ast.literal_eval).tolist()
    return variable_parameter_list


if __name__ == "__main__":
    main()
