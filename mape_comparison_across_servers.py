import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# パラメータリスト
PARAMETER_LISTS = [['T_MCO', 'T_SMO', 'T_MAO'], ['T_BST', 'T_MCO', 'T_SAO']]

# サーバーの順序
SERVER_ORDER = [
    "13th corei5 - GTX1080", "13th corei5 - GTX1650", "13th corei5 - RTX3050",
    "13th corei5 - RTX3060 Ti", "13th corei5 - RTX4070", "13th corei7 - GTX1080",
    "13th corei7 - RTX3050", "13th corei7 - RTX3060 Ti", "13th corei7 - RTX4070",
    "1th Xeon Gold - GTX1080", "1th Xeon Gold - RTX4070", "9th corei7 - RTX2080 Ti"
]


def main():
    data_dir = './ml_results'
    csv_file = 'format_benchmark_parameter_loocv.csv'
    output_dir = './soturon_graph_data'

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
    パラメータをLaTeX形式の下付き文字に変換。

    Args:
        parameter (str): パラメータ名。

    Returns:
        str: LaTeX形式の下付き文字。
    """
    return f"$T_{{{parameter[2:]}}}$"


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
    df['Benchmark Parameter Combinations'] = df['Variable Parameter'].apply(lambda x: get_label(x))
    return df


def plot_mape_comparison(df_filtered, output_dir):
    """
    MAPE比較の棒グラフをプロットして保存。

    Args:
        df_filtered (pd.DataFrame): フィルタリングされたデータフレーム。
        output_dir (str): グラフの保存先ディレクトリ。
    """
    plt.figure(figsize=(14, 7))
    ax = plt.gca()
    ax.grid(axis='y', linestyle='--', zorder=1)  # グリッドをzorder=1で描画

    # 棒グラフのプロット
    sns.barplot(
        x='Leave One',
        y='MAPE test (%)',
        hue='Benchmark Parameter Combinations',
        data=df_filtered,
        palette='Set1',
        order=SERVER_ORDER,
        zorder=2  # 棒グラフをzorder=2で描画
    )

    # タイトルとラベルの設定
    plt.title('MAPE Comparison for Different Parameter Lists Across Servers', fontsize=16)
    plt.xlabel('Test Server', fontsize=12)
    plt.ylabel('MAPE (%)', fontsize=12)
    plt.xticks(rotation=30, ha='right', fontsize=9)
    plt.tight_layout()

    # グラフを保存
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'mape_comparison_operation_transfer.png')
    plt.savefig(output_path, bbox_inches='tight', format='png')
    plt.close()


if __name__ == "__main__":
    main()
