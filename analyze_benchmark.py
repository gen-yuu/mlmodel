import os

import pandas as pd

# 名前のリスト（元の文字列を置き換える）
NAME_LISTS = {
    "matrix_conv": "$T_{SCO}$",
    "matrix_convloop": "$T_{MCO}$",
    "matrix_dot": "$T_{SMO}$",
    "matrix_dotloop": "$T_{MMO}$",
    "matrix_add": "$T_{SAO}$",
    "matrix_addloop": "$T_{MAO}$",
    "transfer_all": "$T_{SLT}$",
    "transfer_continuous": "$T_{BST}$",
    "transfer_roundtrip": "$T_{IST}$"
}

# サーバー名
SERVER_NAME = 'CPU - GPU'

# 行列計算に関するベンチマーク項目
MATRIX_BENCHMARKS = ["$T_{SCO}$", "$T_{MCO}$", "$T_{SMO}$", "$T_{MMO}$", "$T_{SAO}$", "$T_{MAO}$"]

# 転送に関するベンチマーク項目
TRANSFER_BENCHMARKS = ["$T_{SLT}$", "$T_{BST}$", "$T_{IST}$"]

# 出力先ディレクトリ
OUTPUT_DIR = './benchmark_analyze'

# データファイルのパスを設定
DATA_DIR = './data'
DATA_FILE = 'testbench_all.csv'
DATA_PATH = os.path.join(DATA_DIR, DATA_FILE)


def load_data(file_path):
    """CSVファイルを読み込み、必要な処理を行う"""
    df = pd.read_csv(file_path)
    # 最初の列名を 'CPU - GPU' に変更
    df = df.rename(columns={df.columns[0]: SERVER_NAME})
    # 特定の名前の置き換え
    df = df.rename(columns=NAME_LISTS)
    return df


def process_and_sort_data(df):
    """データフレームを並び替え、インデックスをリセット"""
    # 'CPU - GPU' 列でソート（インデックスリセット）
    df = df.sort_values(SERVER_NAME, ignore_index=True)
    return df


def save_benchmark_data(df, benchmark_columns, file_name):
    """指定したベンチマーク項目に基づいてデータを保存"""
    # ベンチマーク項目に 'CPU - GPU' を追加
    benchmark_columns.insert(0, SERVER_NAME)
    benchmark_df = df[benchmark_columns]
    benchmark_df.to_csv(os.path.join(OUTPUT_DIR, file_name), index=False)


def main():
    # データを読み込み、処理を実行
    df = load_data(DATA_PATH)
    df = process_and_sort_data(df)

    # 行列計算のデータを保存
    save_benchmark_data(df, MATRIX_BENCHMARKS, 'matrix_analyze.csv')

    # 転送のデータを保存
    save_benchmark_data(df, TRANSFER_BENCHMARKS, 'transfer_analyze.csv')


if __name__ == "__main__":
    main()
