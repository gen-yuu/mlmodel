import csv
import os

import pandas as pd

from get_server_spec import get_server_spec

data_dir = './data'
testbench_file = 'testbench_all.csv'
results_file = 'results_all.csv'


def main():
    create_data_benchmark_csv()
    create_data_server_spec_csv()
    return


def create_data_benchmark_csv():
    output_file = 'data_benchmark.csv'

    testbench_csv = os.path.join(data_dir, testbench_file)
    results_csv = os.path.join(data_dir, results_file)
    testbench_df = pd.read_csv(testbench_csv, index_col=0)
    results_df = pd.read_csv(results_csv)

    server_list = results_df['Server Info'].unique()
    for server in server_list:
        benchmark = testbench_df.loc[server]
        for name in benchmark.index:
            results_df.loc[results_df['Server Info'] == server, name] = benchmark[name]
        pass

    output_path = os.path.join(data_dir, output_file)
    results_df.to_csv(output_path, mode='w')
    return 0


def create_data_server_spec_csv():
    output_file = 'data_server_spec.csv'
    results_csv = os.path.join(data_dir, results_file)
    results_df = pd.read_csv(results_csv)

    server_list = results_df['Server Info'].unique()
    for server in server_list:
        spec = get_server_spec(server)
        for key, value in spec.items():
            results_df.loc[results_df['Server Info'] == server, key] = value
        pass

    output_path = os.path.join(data_dir, output_file)
    results_df.to_csv(output_path, mode='w')
    return 0


if __name__ == "__main__":
    main()
