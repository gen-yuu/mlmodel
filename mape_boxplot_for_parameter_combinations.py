import os
import sys

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# LOO方式のMAPEを箱ひげ図で評価
PARAMETER_LISTS = [["T_MCO", "T_SMO", "T_MAO"], ["T_MCO", "T_MAO"], ["T_MCO"],
                   ["T_BST", "T_MCO", "T_SAO"], ["T_BST", "T_MCO", "T_MMO", "T_SAO", "T_MAO"],
                   ["T_SLT", "T_BST", "T_SCO", "T_MCO", "T_SMO"],
                   ["T_SLT", "T_IST", "T_MCO", "T_SMO", "T_MMO", "T_SAO"], ["T_BST", "T_SCO"],
                   ["T_IST", "T_SCO"]]


def main():
    data_dir = './ml_results'
    output_dir = './soturon_graph_data'  # 保存先のディレクトリ
    os.makedirs(output_dir, exist_ok=True)  # 保存先フォルダを作成

    data_file = 'format_benchmark_parameter_loocv.csv'

    # データの読み込み
    df = load_data(data_dir, data_file)

    # Variable Parameter をリスト形式に変換
    df['Variable Parameter'] = df['Variable Parameter'].apply(eval)

    # PARAMETER_LISTSの要素に一致する行をフィルタリング
    def filter_by_parameter(param):
        # paramがリストであり、PARAMETER_LISTSの任意のリストと一致するかを確認
        for parameter_list in PARAMETER_LISTS:
            # paramがparameter_listと完全一致するかをチェック
            if set(param) == set(parameter_list):
                return True
        return False

    df_filtered = df[df['Variable Parameter'].apply(filter_by_parameter)]

    # モデルごとのMAPEを箱ひげ図で横並びに表示
    plot_combined_boxplot(df_filtered, output_dir)

    return 0


# MAPE test (%) に対する箱ひげ図を1つのグラフにまとめて描画する関数
def plot_combined_boxplot(df, output_dir):
    """
    すべてのモデルに対するMAPE（Mean Absolute Percentage Error）テスト結果の箱ひげ図を1つのグラフにまとめて描画する。

    この関数は、各モデルのパラメータリスト（例えば ["T_MCO", "T_SMO", "T_MAO"]）に対して、MAPEの値を箱ひげ図として表示する。
    モデルごとにMAPEの分布を視覚的に比較することができます。

    箱ひげ図では、各モデルごとのMAPEの中央値、四分位範囲（IQR）、外れ値を確認できます。

    引数:
    df (pd.DataFrame): 'Variable Parameter' 列にモデルのパラメータリスト、'MAPE test (%)' 列にMAPEの値が含まれるデータフレーム。
    """

    # 各モデルにラベルを付けるためにデータフレームを加工
    def get_label(param_list):
        # LaTeX形式でリストとして表示
        return f"[{','.join([to_latex_subscript(param) for param in param_list])}]"

    df['Model Label'] = df['Variable Parameter'].apply(lambda x: get_label(x))

    # 'Model Label' の順序を PARAMETER_LISTS に従わせる
    df['Model Label'] = pd.Categorical(
        df['Model Label'],
        categories=[
            f"[{','.join([to_latex_subscript(param) for param in param_list])}]"
            for param_list in PARAMETER_LISTS
        ],
        ordered=True)

    # 箱ひげ図を作成
    plt.figure(figsize=(7, 6))  # 横幅を広げるために figsize を調整
    sns.boxplot(
        x='Model Label',
        y='MAPE test (%)',
        data=df,
        palette="Set3",
        showmeans=True,
        # 中央値非表示
        #medianprops={'visible': False},
        width=0.7,
        whis=10.0)

    # ラベルとタイトルの設定
    #plt.title('MAPE (%) Distribution Across Models', fontsize=16)
    plt.xlabel('Benchmark Combinations', fontsize=12)
    plt.ylabel('MAPE (%)', fontsize=12)

    # X軸ラベルを回転しつつフォントサイズを小さく設定
    plt.xticks(rotation=45, ha='right', fontsize=10)
    plt.grid(axis='y', linestyle='--')

    # グラフを保存
    output_file = os.path.join(output_dir, 'mape_boxplot.png')
    plt.tight_layout()
    plt.subplots_adjust(left=0.09, right=0.94, bottom=0.33, top=0.99)  # 余白調整
    plt.savefig(output_file, bbox_inches=None, format='png')  # png形式で保存
    print(f"Graph saved to {output_file}")

    # グラフを表示
    #plt.show()


def to_latex_subscript(parameter):
    """
    LaTeX形式で変数を下付き文字として変換する
    """
    return f"$T_{{{parameter[2:]}}}$"  # 'T_'の後ろを下付き文字として変換


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
