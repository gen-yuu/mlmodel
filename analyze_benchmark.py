import os

import pandas as pd

# 名前のリスト（元の文字列を置き換える）
name_lists = {
    "matrix_conv": "$T_{LCOE}$",
    "matrix_convloop": "$T_{SCRE}$",
    "matrix_dot": "$T_{LMOE}$",
    "matrix_dotloop": "$T_{SMRE}$",
    "matrix_add": "$T_{LAOE}$",
    "matrix_addloop": "$T_{SARE}$",
    "transfer_all": "$T_{SLET}$",
    "transfer_continuous": "$T_{CSET}$",
    "transfer_roundtrip": "$T_{ISET}$"
}

# サーバー名
server = 'CPU - GPU'

# 行列計算に関するベンチマーク項目
matrix_benchmarks = [
    "$T_{LCOE}$", "$T_{SCRE}$", "$T_{LMOE}$", "$T_{SMRE}$", "$T_{LAOE}$", "$T_{SARE}$"
]

# 転送に関するベンチマーク項目
transfer_benchmarks = ["$T_{SLET}$", "$T_{CSET}$", "$T_{ISET}$"]

# 出力先ディレクトリ
output_dir = './benchmark_analyze'


def main():
    # データファイルのパスを設定
    data_dir = './data'
    data_file = 'testbench_all.csv'
    data_path = os.path.join(data_dir, data_file)

    # CSVファイルを読み込み
    df = pd.read_csv(data_path)

    # 最初の列名を 'CPU - GPU' に変更
    df = df.rename(columns={df.columns[0]: 'CPU - GPU'})

    # 特定の名前の置き換え
    df = df.rename(columns=name_lists)

    # 'CPU - GPU' 列でソート（インデックスリセット）
    df = df.sort_values('CPU - GPU', ignore_index=True)

    # ベンチマーク項目に 'CPU - GPU' を追加
    matrix_benchmarks.insert(0, server)
    transfer_benchmarks.insert(0, server)

    # 行列計算のデータフレーム
    matrix_df = df[matrix_benchmarks]

    # 転送のデータフレーム
    transfer_df = df[transfer_benchmarks]

    # インデックスなしでCSVに保存
    matrix_df.to_csv(f"{output_dir}/matrix_analyze.csv", index=False)
    transfer_df.to_csv(f"{output_dir}/transfer_analyze.csv", index=False)


if __name__ == "__main__":
    main()
