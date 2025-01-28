import sys
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def main():
    spec_parameter = [
        'cpu_core', 'cpu_boost_clock(GHz)', 'cpu_cache(MB)', 'gpu_architecture', 'gpu_core',
        'VRAM(GB)'
    ]
    benchmark_parameter = ["T_MCO", "T_SMO", "T_MAO"]

    data_dir = "./ml_results"
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

    # `Category` 列を追加して、どのデータが何のパラメータに対応するかを明示
    spec_filtered_df['Category'] = 'Hardware spec Model'
    benchmark_filtered_df['Category'] = 'Benchmark Model'

    # データフレームを結合
    combined_df = pd.concat([benchmark_filtered_df, spec_filtered_df])

    # 箱ひげ図の作成
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='Category',
                y='MAPE test (%)',
                data=combined_df,
                showmeans=True,
                palette='Set3',
                width=0.5,
                whis=2.0)

    # タイトルとラベル設定
    plt.xlabel('Model Type', fontsize=12)
    plt.ylabel('MAPE (%)', fontsize=12)
    plt.xticks(fontsize=9)
    plt.tight_layout()

    # 出力ディレクトリの設定
    output_dir = './soturon_graph_data'
    os.makedirs(output_dir, exist_ok=True)  # ディレクトリがない場合は作成

    # ファイルパスを設定して保存
    output_path = os.path.join(output_dir, 'mape_comparison_spec_boxplot.png')
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
