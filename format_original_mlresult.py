import ast  # 文字列をリストに変換するために使用
import os

import pandas as pd

# 置き換えリスト
rename_list = {
    "transfer_all": "T_SLET",
    "transfer_continuous": "T_CSET",
    "transfer_roundtrip": "T_ISET",
    "matrix_convloop": "T_CSCO",
    "matrix_conv": "T_SLCO",
    "matrix_dotloop": "T_CSMO",
    "matrix_dot": "T_SLMO",
    "matrix_addloop": "T_CSAO",
    "matrix_add": "T_SLAO"
}

# benchmarkのTime Cost(s)
# 実行ループ回数
M = 100
weights = {
    "T_SLET": 0.247507 * M,
    "T_CSET": 0.146412 * M,
    "T_ISET": 0.190528 * M,
    "T_SLCO": 0.747215 * M,
    "T_CSCO": 1.947857 * M,
    "T_SLMO": 9.083457 * M,
    "T_CSMO": 1.033667 * M,
    "T_SLAO": 0.008916 * M,
    "T_CSAO": 0.886261 * M,
}


# メイン処理をまとめる関数
def main():
    data_dir = './ml_results'
    # 'spec_parameter_search.csv' の場合、Time Cost計算をスキップ
    process_csv(data_dir,
                'original_benchmark_parameter_loocv.csv',
                'benchmark_parameter_loocv.csv',
                calculate_time_cost=True)
    process_csv(data_dir,
                'original_spec_parameter_loocv.csv',
                'spec_parameter_loocv.csv',
                calculate_time_cost=False)


# CSVを読み込み、必要な操作を実行する関数
def process_csv(data_dir, data_file, output_file, calculate_time_cost=True):
    data_path = os.path.join(data_dir, data_file)
    df = pd.read_csv(data_path, index_col=0)

    # 文字列を置き換える
    for old_string, new_string in rename_list.items():
        df = df.replace(old_string, new_string, regex=True)

    # Time Cost (s)を計算して新しい列を追加（必要な場合）
    if calculate_time_cost:

        # コスト計算関数
        def calculate_time_cost(inputs):
            # 文字列をリストに変換
            input_list = ast.literal_eval(inputs)
            cost = sum(weights.get(parameter, 0) for parameter in input_list)
            # 小数点以下5桁に丸める
            return round(cost, 5)

        df["Time Cost (s)"] = df['Variable Parameter'].apply(calculate_time_cost)

    # 列の並び替え
    columns = df.columns.tolist()  # 現在の列順を取得
    if "Time Cost (s)" in columns:
        columns.remove("Time Cost (s)")  # 一旦 "Time Cost (s)" を除去
        columns.insert(columns.index('Variable Parameter') + 1,
                       "Time Cost (s)")  # "Input" の次に "Time Cost (s)" を挿入
    columns.remove("Leave One")  # 一旦 "Leave One" を除去
    columns.insert(columns.index("MAPE train (%)"),
                   "Leave One")  # "MAPE train(%)" の直前に "Leave One" を挿入
    df = df[columns]  # 列の順序を再設定

    # 結果を保存
    output_path = os.path.join(data_dir, output_file)
    df.to_csv(output_path)
    print(f"置き換え後のCSVを '{output_path}' に保存しました。")


# スクリプトが直接実行された場合に実行される処理
if __name__ == "__main__":
    main()
