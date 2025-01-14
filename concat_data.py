import csv
import os
import pandas as pd

data_dir = './data'
testbench_file = 'testbench_all.csv'
results_file = 'results_all.csv'
output_file = 'data.csv'

testbench_csv = os.path.join(data_dir, testbench_file)
results_csv = os.path.join(data_dir, results_file)

testbench_df = pd.read_csv(testbench_csv, index_col=0)
results_df = pd.read_csv(results_csv)

server_list = results_df['Server Info'].unique()

for server in server_list:
    index = testbench_df.index.get_loc(server)
    benchmark = testbench_df.loc[server]
    target_df = results_df.query('`Server Info` == @server')
    for name in benchmark.index:
        results_df.loc[results_df['Server Info']
                       == server, name] = benchmark[name]
    print(results_df)
    pass


output_path = os.path.join(data_dir, output_file)

results_df.to_csv(output_path, mode='w')
