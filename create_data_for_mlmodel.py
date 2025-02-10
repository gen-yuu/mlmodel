import os

import pandas as pd
from get_server_spec import get_server_spec

# データディレクトリとファイルパスの設定
data_dir = './data'
testbench_file = 'testbench_all.csv'
results_file = 'results_all.csv'


def main():
    create_data_benchmark_csv()
    create_data_server_spec_csv()


def create_data_benchmark_csv():
    """testbenchとresultsのデータを統合し、data_benchmark.csvを作成"""
    testbench_csv = os.path.join(data_dir, testbench_file)
    results_csv = os.path.join(data_dir, results_file)

    testbench_df = pd.read_csv(testbench_csv, index_col=0)
    results_df = pd.read_csv(results_csv)

    # サーバーごとにベンチマーク情報を統合
    for server in results_df['Server Info'].unique():
        benchmark = testbench_df.loc[server]
        for name in benchmark.index:
            results_df.loc[results_df['Server Info'] == server, name] = benchmark[name]

    # 結果をCSVに書き出し
    output_path = os.path.join(data_dir, 'data_benchmark.csv')
    results_df.to_csv(output_path, mode='w', index=False)


def create_data_server_spec_csv():
    """resultsのサーバー情報に基づいて、サーバースペックを取得し、data_server_spec.csvを作成"""
    results_csv = os.path.join(data_dir, results_file)
    results_df = pd.read_csv(results_csv)

    # サーバーごとにスペック情報を追加
    for server in results_df['Server Info'].unique():
        spec = get_server_spec(server)
        for key, value in spec.items():
            results_df.loc[results_df['Server Info'] == server, key] = value

    # 結果をCSVに書き出し
    output_path = os.path.join(data_dir, 'data_server_spec.csv')
    results_df.to_csv(output_path, mode='w', index=False)


# サーバー情報に基づき、CPUとGPUのスペックを取得する関数
def get_server_spec(server):
    cpu_info = {
        "13th corei5": {
            "cpu_core": 14,
            "cpu_boost_clock(GHz)": 5.1,
            "cpu_thread": 20,
            "cpu_cache(MB)": 20
        },
        "13th corei7": {
            "cpu_core": 16,
            "cpu_boost_clock(GHz)": 5.2,
            "cpu_thread": 24,
            "cpu_cache(MB)": 30
        },
        "9th corei7": {
            "cpu_core": 8,
            "cpu_boost_clock(GHz)": 4.7,
            "cpu_thread": 8,
            "cpu_cache(MB)": 12
        },
        "1st Xeon Gold": {
            "cpu_core": 4,
            "cpu_boost_clock(GHz)": 3.7,
            "cpu_thread": 8,
            "cpu_cache(MB)": 16.5
        }
    }

    gpu_info = {
        "RTX4070": {
            "gpu_architecture": 0,
            "gpu_core": 5888,
            "gpu_boost_clock(GHz)": 2.48,
            "VRAM(GB)": 12,
        },
        "RTX3060 Ti": {
            "gpu_architecture": 1,
            "gpu_core": 4864,
            "gpu_boost_clock(GHz)": 1.665,
            "VRAM(GB)": 8,
        },
        "RTX3050": {
            "gpu_architecture": 1,
            "gpu_core": 2560,
            "gpu_boost_clock(GHz)": 1.777,
            "VRAM(GB)": 8,
        },
        "RTX2080 Ti": {
            "gpu_architecture": 2,
            "gpu_core": 4352,
            "gpu_boost_clock(GHz)": 1.640,
            "VRAM(GB)": 11,
        },
        "GTX1650": {
            "gpu_architecture": 2,
            "gpu_core": 896,
            "gpu_boost_clock(GHz)": 1.665,
            "VRAM(GB)": 4,
        },
        "GTX1080": {
            "gpu_architecture": 3,
            "gpu_core": 2560,
            "gpu_boost_clock(GHz)": 1.733,
            "VRAM(GB)": 8,
        }
    }

    # サーバー名からCPUとGPUを抽出して、対応するスペックを結合
    cpu, gpu = server.split(" - ")
    parameter = {**cpu_info[cpu], **gpu_info[gpu]}
    return parameter


if __name__ == "__main__":
    main()
