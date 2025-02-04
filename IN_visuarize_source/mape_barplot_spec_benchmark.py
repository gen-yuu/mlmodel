import os
import sys

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# サーバーの順序
SERVER_ORDER = [
    "13th Core i5 - GTX1080", "13th Core i5 - GTX1650", "13th Core i5 - RTX3050",
    "13th Core i5 - RTX3060 Ti", "13th Core i5 - RTX4070", "13th Core i7 - GTX1080",
    "13th Core i7 - RTX3050", "13th Core i7 - RTX3060 Ti", "13th Core i7 - RTX4070",
    "1th Xeon Gold - GTX1080", "1th Xeon Gold - RTX4070", "9th Core i7 - RTX2080 Ti"
]


def main():
    spec_parameter = [
        'cpu_core', 'cpu_boost_clock(GHz)', 'cpu_cache(MB)', 'gpu_architecture', 'gpu_core',
        'VRAM(GB)'
    ]
    benchmark_parameter = ["T_MCO", "T_SMO", "T_MAO"]

    data_dir = "../ml_results"
    spec_csv = "format_spec_parameter_loocv.csv"
    benchmark_csv = "format_benchmark_parameter_loocv.csv"

    # データの読み込み
    spec_df = load_data(data_dir, spec_csv)
    benchmark_df = load_data(data_dir, benchmark_csv)

    # Variable Parameter をリスト形式に変換
    spec_df['Variable Parameter'] = spec_df['Variable Parameter'].apply(eval)
    benchmark_df['Variable Parameter'] = benchmark_df['Variable Parameter'].apply(eval)

    # spec_df の Variable Parameter が spec_parameter と順不同で一致する行を抽出
    spec_filtered_df = spec_df[spec_df['Variable Parameter'].apply(
        lambda x: set(x) == set(spec_parameter))]

    # benchmark_df の Variable Parameter が benchmark_parameter と順不同で一致する行を抽出
    benchmark_filtered_df = benchmark_df[benchmark_df['Variable Parameter'].apply(
        lambda x: set(x) == set(benchmark_parameter))]

    # `Model Type` 列を追加して、どのデータが何のパラメータに対応するかを明示
    spec_filtered_df['Model Type'] = 'Hardware spec Model'
    benchmark_filtered_df['Model Type'] = 'Benchmark Model'

    hue_order = ['Benchmark Model', 'Hardware spec Model']
    # データフレームを結合
    combined_df = pd.concat([benchmark_filtered_df, spec_filtered_df])

    # 棒グラフの作成
    plt.figure(figsize=(7, 5))
    ax = plt.gca()
    ax.grid(axis='y', linestyle='--', zorder=1)  # グリッドをzorder=1で描画
    palette = ['#1f77b4', '#ff7f0e']
    # 棒グラフのプロット
    sns.barplot(
        x='Leave One',
        y='MAPE test (%)',
        hue='Model Type',
        data=combined_df,
        palette=palette,
        order=SERVER_ORDER,
        hue_order=hue_order,
        zorder=2  # 棒グラフをzorder=2で描画
    )

    # Model Type ごとの平均 MAPE を計算
    avg_mape = combined_df.groupby('Model Type')['MAPE test (%)'].mean().reset_index()

    # 平均MAPEを直線でY軸に描画
    for model_type in hue_order:
        # 指定したModel Typeのデータを抽出
        model_data = avg_mape[avg_mape['Model Type'] == model_type]

        # 直線の色を設定
        if model_type == 'Benchmark Model':
            line_color = '#1f77b4'  # 青色
        elif model_type == 'Hardware spec Model':
            line_color = '#ff7f0e'  # オレンジ色

        # Y軸に平均MAPEの直線を引く
        ax.axhline(y=model_data['MAPE test (%)'].values[0],
                   color=line_color,
                   linestyle='--',
                   linewidth=1,
                   label=f'{model_type} average MAPE')

    # タイトルとラベル設定
    # 凡例を設定
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.xlabel('Test Server', fontsize=12)
    plt.ylabel('MAPE (%)', fontsize=12)
    plt.xticks(rotation=45, ha='right', fontsize=10)
    plt.tight_layout()
    plt.subplots_adjust(left=0.15, right=0.9, bottom=0.28, top=0.99)  # 余白調整

    # 出力ディレクトリの設定
    output_dir = '../IN_graph'
    os.makedirs(output_dir, exist_ok=True)  # ディレクトリがない場合は作成

    # ファイルパスを設定して保存
    output_path = os.path.join(output_dir, 'mape_comparison_spec_barplot.png')
    plt.savefig(output_path, format='png')

    # 結果を表示
    print(f"図は {output_path} に保存されました。")

    plt.show()


def load_data(data_dir, data_file):
    """
    データを読み込み、欠損値をチェックする関数。

    Args:
        data_dir (str): データディレクトリのパス
        data_file (str): データファイル名

    Returns:
        pd.DataFrame: 読み込んだデータフレーム
    """
    try:
        data_path = os.path.join(data_dir, data_file)
        df = pd.read_csv(data_path)
        if df.isna().any().any():
            raise ValueError("データに欠損値（NaN）が含まれています。")
        return df
    except FileNotFoundError:
        print(f"ファイル {data_file} が見つかりませんでした。")
        sys.exit(1)
    except ValueError as e:
        print(e)
        sys.exit(1)


if __name__ == "__main__":
    main()
