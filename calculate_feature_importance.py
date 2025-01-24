import ast
import itertools
import os

import pandas as pd

import light_gbm as lgb_reg
import shap
from mldata_format import format_data_loocv

# サーバーリスト
SERVER_LIST = [
    '13th corei5 - RTX3060 Ti', '13th corei7 - GTX1080', '13th corei5 - GTX1650',
    '1th Xeon Gold - RTX4070', '13th corei7 - RTX3050', '13th corei5 - GTX1080',
    '13th corei5 - RTX4070', '13th corei7 - RTX3060 Ti', '13th corei5 - RTX3050',
    '1th Xeon Gold - GTX1080', '9th corei7 - RTX2080 Ti', '13th corei7 - RTX4070'
]

# データの出力先
output_dir = './mldata_analyze'
parameterdata_dir = './mldata_analyze'

# parameterデータファイル
# transferとoperationが両方含まれる
csv_file = 'filtered_by_variable_parameters_mape_test.csv'
sample_csv = 'soturon_shap_param_list.csv'
# mlデータファイル
mldata_dir = './data'
benchmark_data_file = 'data_benchmark.csv'
server_spec_data_file = 'data_server_spec.csv'

# 定数特徴量
const_parameters = ['Total Frames', 'Directory Size (MB)', 'Params']

# ターゲット変数
target = 'Inference Time (s)'

rename_list = {
    'T_SLT': 'transfer_all',
    'T_BST': 'transfer_continuous',
    'T_IST': 'transfer_roundtrip',
    'T_MCO': 'matrix_convloop',
    'T_SCO': 'matrix_conv',
    'T_MMO': 'matrix_dotloop',
    'T_SMO': 'matrix_dot',
    'T_MAO': 'matrix_addloop',
    'T_SAO': 'matrix_add'
}
reverse_rename_list = {v: k for k, v in rename_list.items()}


def main():
    csv_path = os.path.join(parameterdata_dir, sample_csv)

    variable_parameter_list = get_variable_parameter_list(csv_path)
    # 関数の呼び出し
    variable_parameter_list = renamed_variable_parameter(variable_parameter_list, rename_list)
    model_info = []
    mldata_path = os.path.join(mldata_dir, benchmark_data_file)
    for variable_parameter in variable_parameter_list:
        model_info.extend(loocv(const_parameters, variable_parameter, mldata_path))
        pass
    output_results_to_csv(model_info, 'soturon_shap_graph.csv')
    return 0


def get_variable_parameter_list(data_path, num_rows=None):
    """
    CSVファイルからVariable Parameterカラムを読み込み、リストとして取得する。

    Args:
        data_path (str): CSVファイルのパス
        num_rows (int, optional): 上から取得する行数を指定。Noneの場合はすべて取得。

    Returns:
        list of list of str: Variable Parameterカラムのリスト
    """
    # CSVファイルの読み込み
    df = pd.read_csv(data_path)

    # 指定された行数を取得
    if num_rows is not None:
        df = df.head(num_rows)

    # Variable Parameterカラムからリストを取得し、各行をリストとして変換
    variable_parameter_list = df['Variable Parameter'].apply(ast.literal_eval).tolist()
    return variable_parameter_list


# 関数の定義
def renamed_variable_parameter(variable_parameters, rename_mapping):
    """
    リスト内の変数名を指定された辞書で置き換える。

    Args:
        variable_parameters (list of list of str): 変数名のリスト
        rename_mapping (dict): リネームのマッピング辞書

    Returns:
        list of list of str: リネーム後の変数リスト
    """
    return [[rename_mapping.get(var, var) for var in sublist] for sublist in variable_parameters]


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
    reverse_server_parameters = [reverse_rename_list.get(p, p) for p in server_parameters]
    for server in SERVER_LIST:
        print(f"parameters : {server_parameters}, Leave out server: {server}")
        train_df, test_df = format_data_loocv(server, data_path)
        parameters = const_parameters + server_parameters
        #lightGBM
        #訓練データが8:2でtrain:valに分割される
        lgb_model, loss, train_df, val_df = lgb_reg.train_lgb_model(train_df, target, parameters)
        mape_test = lgb_reg.predict_and_evaluate(lgb_model, test_df, target, parameters)
        shap_values_dict = get_shap_values_for_parameters(lgb_model, test_df, server_parameters,
                                                          parameters)
        # SHAP値のリスト形式をリネーム
        shap_values_dict = [{
            'Parameter': reverse_rename_list.get(entry['Parameter'], entry['Parameter']),
            'Mean Absolute SHAP Value': entry['Mean Absolute SHAP Value'],
            'Sum Absolute SHAP Value': entry['Sum Absolute SHAP Value']
        } for entry in shap_values_dict]
        #ベンチマークパラメータを逆向きにリネーム
        lgb_result = {
            'ML': 'lgb',
            'loss': loss,
            'Parameter Num': len(parameters),
            'Const Parameter': const_parameters,
            'Variable Parameter Num': len(server_parameters),
            'Variable Parameter': reverse_server_parameters,
            'SHAP Value': shap_values_dict,
            'MAPE test (%)': mape_test,
            'Leave One': server
        }

        model_info.append(lgb_result)
        print(shap_values_dict)
    return model_info


def get_shap_values_for_parameters(model, test_df, target_parameters, parameters):
    """
    特定の変数に絞ったSHAP値を計算しリストとして返す関数。

    Args:
        model: 学習済みモデル (LightGBM, XGBoostなど)
        X_test: テストデータ (DataFrame形式)
        variable_parameters: SHAP値を計算する対象の変数名リスト

    Returns:
        dict: 特定の変数に対応するSHAP値の辞書形式リスト
              例: {"variable1": [value1, value2, ...], "variable2": [value1, ...]}
    """
    # テストデータの準備
    X_test = pd.get_dummies(test_df[parameters], drop_first=True)
    # SHAPのTreeExplainerを使用してSHAP値を計算
    explainer = shap.TreeExplainer(model)
    X_test_shap = X_test.copy().reset_index(drop=True)
    shap_values = explainer.shap_values(X=X_test_shap)
    # 結果を格納する辞書
    shap_values_dict = {}
    # 指定された変数に絞ってSHAP値を取得
    for parameter in target_parameters:
        if parameter in X_test_shap.columns:
            # 対象変数のSHAP値を取得
            index = X_test_shap.columns.get_loc(parameter)
            shap_values_dict[parameter] = shap_values[:, index].tolist()
        else:
            raise ValueError(f"Variable '{parameter}' is not found in X_test columns.")

    # 分析実行
    importance_data = analyze_shap_importance(shap_values_dict)

    return importance_data


def analyze_shap_importance(shap_values_dict):
    """
    SHAP値から各変数の予測への寄与度を分析し、重要度をランキング形式で出力する。

    Args:
        shap_values_dict: 特定の変数に対応するSHAP値の辞書形式リスト
                          例: {"variable1": [value1, value2, ...], "variable2": [value1, ...]}

    Returns:
        list of dict: 各変数の重要度ランキング
                      Columns: ["Parameter", "Mean Absolute SHAP Value", "Sum Absolute SHAP Value"]
    """

    # 各変数のSHAP値を絶対値にして、平均と合計を計算
    importance_data = []
    for parameter, shap_values in shap_values_dict.items():
        abs_shap_values = [abs(val) for val in shap_values]
        mean_abs_shap = sum(abs_shap_values) / len(abs_shap_values)
        sum_abs_shap = sum(abs_shap_values)
        importance_data.append({
            "Parameter": parameter,
            "Mean Absolute SHAP Value": mean_abs_shap,
            "Sum Absolute SHAP Value": sum_abs_shap
        })

    return importance_data


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
