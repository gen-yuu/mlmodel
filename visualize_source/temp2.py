import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from config import PLT_FONT

plt.rcParams['font.family'] = PLT_FONT

output_dir = '../soturon_graph'  # 保存先のディレクトリ


def plot_benchmark_time(data, output_filename):
    """
    データを受け取り、ベンチマークの棒グラフを作成し、指定のファイル名で保存する関数。

    Parameters:
    - data (dict): GPU/CPUと各ベンチマークの実行時間を含む辞書形式のデータ
    - output_filename (str): 保存するファイル名
    """
    # DataFrameの作成
    df = pd.DataFrame(data)

    # データをlong formatに変換
    df_melted = df.melt(
        id_vars=df.columns[0],  # 最初の列（GPUやCPU名）
        value_vars=df.columns[1:],  # ベンチマークの列
        var_name='Benchmark',
        value_name='Time (s)')

    # LaTeX形式に変換する関数
    def convert_to_latex_format(parameter):
        return f"T$_{{\\text{{{parameter[2:]}}}}}$"

    # Benchmark列を変換
    df_melted["Benchmark"] = df_melted["Benchmark"].apply(convert_to_latex_format)

    # df の列数（最初の列はカテゴリ列として除外）
    num_benchmarks = len(df.columns) - 1  # 例えば "CPU" や "GPU" の列を除外
    if num_benchmarks == 3:
        palette = ["#1f77b4", "#ff7f0e", "#d62728"]  # 青・オレンジ・赤
    else:
        palette = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#edc949"]
    plt.figure(figsize=(5, 3))
    ax = plt.gca()
    ax.grid(True, zorder=1)  # グリッドを zorder=1 に設定
    sns.barplot(x=df.columns[0],
                y='Time (s)',
                hue='Benchmark',
                data=df_melted,
                palette=palette,
                width=0.6,
                zorder=2)
    ax.legend().set_title('')
    plt.xlabel(df.columns[0], fontsize=12)  # CPUやGPU名に応じたラベル
    plt.ylabel('Time (s, log scale)', fontsize=12)
    plt.xticks(fontsize=10)
    plt.yscale("log")
    plt.tight_layout()
    plt.subplots_adjust(right=0.85)  # 下部の余白を確保

    # 保存
    save_path = os.path.join(output_dir, output_filename)
    plt.savefig(save_path)
    plt.close()


# # 使用例
# data1 = {
#     'CPU': ['13th Core i7', '13th Core i5', '1st Xeon Gold'],
#     'T_SLT': [0.18259, 0.19787, 0.39629],
#     'T_BST': [0.11185, 0.11958, 0.17592],
#     'T_IST': [0.14290, 0.15332, 0.24017]
# }

# plot_benchmark_time(data1, 'transfer_rtx4070_cpu.png')

# data2 = {
#     'GPU': ['RTX4070', 'RTX3060 Ti', 'RTX3050', 'GTX1650', 'GTX1080'],
#     'T_SLT': [0.19787, 0.19683, 0.22359, 0.22539, 0.22363],
#     'T_BST': [0.11958, 0.11924, 0.13752, 0.13858, 0.13802],
#     'T_IST': [0.15332, 0.15200, 0.18115, 0.18422, 0.18245]
# }

# plot_benchmark_time(data2, 'transfer_13th_Core_i5_gpu.png')

# データの設定
data3 = {
    'GPU': ['RTX4070', 'GTX1080'],
    'T_SLT': [0.39629, 0.39282],
    'T_BST': [0.17592, 0.17521],
    'T_IST': [0.24017, 0.24391]
}
plot_benchmark_time(data3, 'transfer_1st_Xeon_Gold_gpu.png')

# # データの設定
# data4 = {
#     'GPU': ['RTX4070', 'RTX3060 Ti', 'RTX3050', 'GTX1650', 'GTX1080'],
#     'T_SCO': [0.36207, 0.61767, 1.32907, 1.76343, 0.56396],
#     'T_MCO': [0.80885, 1.38042, 3.10884, 3.95939, 2.36742],
#     'T_SMO': [4.44514, 7.77619, 16.50043, 19.24658, 7.47047],
#     'T_MMO': [0.49520, 0.83275, 1.79453, 2.17332, 0.97811],
#     'T_SAO': [0.00517, 0.00579, 0.01566, 0.01432, 0.00994],
#     'T_MAO': [0.51385, 0.57337, 1.55930, 1.42609, 0.98987]
# }

# plot_benchmark_time(data4, 'matrix_13th_Core_i5_gpu.png')

# data5 = {
#     "CPU": ["13th Core i7", "13th Core i5", "1st Xeon Gold"],
#     "T_SCO": [0.55801, 0.56396, 0.56004],
#     "T_MCO": [2.28216, 2.36742, 2.29248],
#     "T_SMO": [7.44310, 7.47047, 7.49736],
#     "T_MMO": [0.96404, 0.97811, 0.97532],
#     "T_SAO": [0.00994, 0.00994, 0.00998],
#     "T_MAO": [0.98969, 0.98987, 0.98950],
# }
# plot_benchmark_time(data5, 'matrix_gtx1080_cpu.png')

# # データの設定
# data6 = {
#     "CPU": ["13th Core i7", "13th Core i5", "1st Xeon Gold"],
#     "T_SCO": [0.36004, 0.36207, 0.40328],
#     "T_MCO": [0.80333, 0.80885, 0.84915],
#     "T_SMO": [4.42400, 4.44514, 4.49111],
#     "T_MMO": [0.48996, 0.49520, 0.49006],
#     "T_SAO": [0.00516, 0.00517, 0.00522],
#     "T_MAO": [0.51386, 0.51385, 0.51435],
# }
# plot_benchmark_time(data6, 'matrix_rtx4070_cpu.png')
