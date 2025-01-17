import os

import pandas as pd

#命名例
#Large_Matrix_One_Execution  Small_Matrix_Repeated_Executions
#通信パターン（単一回転送、連続転送、反復転送）
# Single_Large_Element_Transfer Iterative_Small_Element_Transfer Continuous_Small_Element_Transfer
name_lists = {
    "matrix_conv": "$T_{LCOE}$",
    "matrix_convloop": "$T_{SCRE}$",
    "matrix_dot": "$T_{SMOE}$",
    "matrix_dotloop": "$T_{SMRE}$",
    "matrix_add": "$T_{LAOE}$",
    "matrix_addloop": "$T_{SAOE}$",
    "transfer_all": "$T_{SLET}$",
    "transfer_continuous": "$T_{CSET}$",
    "transfer_roundtrip": "$T_{ISET}$"
}

#print(df.mean(numeric_only=True))
# # 与えられたデータと重み
# data = {
#     "T_{SLET}": 0.247507,
#     "T_{CSET}": 0.146412,
#     "T_{ISET}": 0.190528,
#     "T_{LCOE}": 0.747215,
#     "T_{SCRE}": 1.947857,
#     "T_{SMOE}": 9.083457,
#     "T_{SMRE}": 1.033667,
#     "T_{LAOE}": 0.008916,
#     "T_{SAOE}": 0.886261,
# }

server = 'CPU - GPU'

matrix_benchmarks = [
    "$T_{LCOE}$", "$T_{SCRE}$", "$T_{SMOE}$", "$T_{SMRE}$", "$T_{LAOE}$", "$T_{SAOE}$"
]
transfer_benchmarks = ["$T_{SLET}$", "$T_{CSET}$", "$T_{ISET}$"]

output_dir = './benchmark_analyze'


# GPU parameters as measured with micro-benchmarks.
def main():
    data_dir = './data'
    data_file = 'testbench_all.csv'
    data_path = os.path.join(data_dir, data_file)
    df = pd.read_csv(data_path)
    df = df.rename(columns={df.columns[0]: 'CPU - GPU'})
    df = df.rename(columns=name_lists)
    df = df.sort_values('CPU - GPU', ignore_index=True)
    matrix_benchmarks.insert(0, server)
    transfer_benchmarks.insert(0, server)
    matrix_df = df[matrix_benchmarks]
    transfer_df = df[transfer_benchmarks]
    # インデックスなしでCSVに保存
    matrix_df.to_csv(f"{output_dir}/matrix_analyze.csv", index=False)
    transfer_df.to_csv(f"{output_dir}/transfer_analyze.csv", index=False)
    return


if __name__ == "__main__":
    main()
