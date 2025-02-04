import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from config import SERVER_ORDER, PLT_FONT

plt.rcParams['font.family'] = PLT_FONT
# パラメータリスト
PARAMETER_LISTS = [['T_MCO', 'T_SMO', 'T_MAO'], ['T_BST', 'T_MCO', 'T_SAO']]


def main():
    data_dir = '../ml_results'
    csv_file = 'format_benchmark_parameter_loocv.csv'
    output_dir = '../soturon_graph'

    # データ読み込み
    df = load_data(data_dir, csv_file)

    # データフィルタリング
    df_filtered = filter_by_parameter(df)

    # データ準備
    df_filtered = prepare_data(df_filtered)

    # グラフ作成と保存
    plot_mape_comparison(df_filtered, output_dir)


def load_data(data_dir, csv_file):
    """
    データをCSVから読み込む関数。

    Args:
        data_dir (str): CSVファイルが格納されているディレクトリ。
        csv_file (str): CSVファイル名。

    Returns:
        pd.DataFrame: 読み込んだデータフレーム。
    """
    csv_path = os.path.join(data_dir, csv_file)
    df = pd.read_csv(csv_path)
    # Variable Parameter列をリスト形式に変換
    df['Variable Parameter'] = df['Variable Parameter'].apply(eval)
    return df


def filter_by_parameter(df):
    """
    指定されたパラメータリストでフィルタリングする関数。

    Args:
        df (pd.DataFrame): 入力データフレーム。

    Returns:
        pd.DataFrame: フィルタリング後のデータフレーム。
    """

    def match_parameter(param):
        for parameter_list in PARAMETER_LISTS:
            if set(param) == set(parameter_list):
                return True
        return False

    return df[df['Variable Parameter'].apply(match_parameter)]


def to_latex_subscript(parameter):
    """
    LaTeX形式で変数を下付き文字として変換する
    """
    return f"T$_{{\\text{{{parameter[2:]}}}}}$"  # 'T_'の後ろを下付き文字として変換


def get_label(param_list):
    """
    パラメータリストをLaTeX形式のラベルに変換。

    Args:
        param_list (list): パラメータリスト。

    Returns:
        str: LaTeX形式のラベル。
    """
    return f"[{','.join([to_latex_subscript(param) for param in param_list])}]"


def prepare_data(df):
    """
    データフレームにBenchmark Parameter Combinations列を追加。

    Args:
        df (pd.DataFrame): 入力データフレーム。

    Returns:
        pd.DataFrame: 変換後のデータフレーム。
    """
    df['Benchmark Combinations'] = df['Variable Parameter'].apply(lambda x: get_label(x))
    return df


def plot_mape_comparison(df_filtered, output_dir):
    """
    MAPE比較の棒グラフをプロットして保存。

    Args:
        df_filtered (pd.DataFrame): フィルタリングされたデータフレーム。
        output_dir (str): グラフの保存先ディレクトリ。
    """
    plt.figure(figsize=(7, 6))
    ax = plt.gca()
    ax.grid(axis='y', linestyle='--', zorder=1)  # グリッドをzorder=1で描画

    # Benchmark Parameter CombinationsをPARAMETER_LISTSの順番にソート
    hue_order = [get_label(param_list) for param_list in PARAMETER_LISTS]
    # 色の設定: 青とオレンジ
    palette = ["#1f77b4", "#ff7f0e"]  # 青とオレンジ
    # 棒グラフのプロット
    sns.barplot(
        x='Leave One',
        y='MAPE test (%)',
        hue='Benchmark Combinations',
        data=df_filtered,
        palette=palette,
        order=SERVER_ORDER,
        hue_order=hue_order,
        zorder=2  # 棒グラフをzorder=2で描画
    )

    # タイトルとラベルの設定
    #plt.title('MAPE Comparison for Different Parameter Lists Across Servers', fontsize=16)
    plt.xlabel('Test Server', fontsize=12)
    plt.ylabel('MAPE (%)', fontsize=12)
    plt.xticks(rotation=45, ha='right', fontsize=10)
    plt.tight_layout()
    plt.subplots_adjust(left=0.16, right=0.9, bottom=0.30, top=0.99)  # 余白調整

    # グラフを保存
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'mape_comparison_operation_transfer.png')
    plt.savefig(output_path, bbox_inches=None, format='png')
    plt.close()


if __name__ == "__main__":
    main()
