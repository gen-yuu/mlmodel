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
    data_dir = '../ml_results'
    output_dir = '../IN_graph'  # 保存先のディレクトリ
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


import matplotlib.patches as mpatches  # 色付き凡例用


def plot_combined_boxplot(df, output_dir):
    """
    すべてのモデルに対するMAPE（Mean Absolute Percentage Error）テスト結果の箱ひげ図を1つのグラフにまとめて描画する。

    各モデルのベンチマーク組み合わせごとに異なる色を設定し、凡例を3列3行のグリッドとしてグラフ下部に追加する。

    引数:
    df (pd.DataFrame): 'Variable Parameter' 列にモデルのパラメータリスト、'MAPE test (%)' 列にMAPEの値が含まれるデータフレーム。
    """

    # 色のリスト（Set3 の色を手動設定）
    color_palette = sns.color_palette("Set3", n_colors=len(PARAMETER_LISTS))

    # 各モデルにラベルを付けるためにデータフレームを加工
    def get_label(param_list):
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
    plt.figure(figsize=(8, 5))
    ax = sns.boxplot(
        x='Model Label',
        y='MAPE test (%)',
        data=df,
        palette=color_palette,  # カラーを設定
        showmeans=True,
        width=0.7,
        whis=10.0)

    # 軸ラベルの設定
    plt.ylabel('MAPE (%)', fontsize=12)
    plt.xlabel('')
    plt.xticks([])
    plt.grid(axis='y', linestyle='--')

    # =======================
    # 凡例の追加（3列×3行のグリッド）
    # =======================
    row_count = 3  # 列数
    row_spacing = 0.07  # 行間隔
    col_spacing = 0.37  # 列間隔

    legend_x_start = 0.5 - (row_count * col_spacing) / 2  # X位置を中央寄せ
    legend_y_start = -0.1  # Y位置を調整（図の真ん中下部）

    for i, (param_list, color) in enumerate(zip(PARAMETER_LISTS, color_palette)):
        col = i // row_count  # 列の決定（先に列を増やす）
        row = i % row_count  # 行の決定（行を折り返す）
        x_pos = legend_x_start + col * col_spacing
        y_pos = legend_y_start - row * row_spacing

        # 色付きの四角形
        plt.gca().add_patch(
            mpatches.Rectangle(
                (x_pos, y_pos),
                0.05,
                0.05,
                facecolor=color,
                edgecolor="gray",  # 枠線の色
                linewidth=1.5,  # 枠線の太さ
                transform=plt.gca().transAxes,
                clip_on=False))

        # ラベル（黒色）
        plt.text(
            x_pos + 0.06,
            y_pos + 0.02,
            f"[{','.join([to_latex_subscript(param) for param in param_list])}]",
            fontsize=10,
            ha='left',
            va='center',
            color='black',
            transform=plt.gca().transAxes,
        )

    # =======================
    # グラフの保存
    # =======================
    output_file = os.path.join(output_dir, 'mape_boxplot.png')
    plt.subplots_adjust(left=0.05, right=0.8, bottom=0.30, top=0.99)  # 下部の余白を確保
    plt.savefig(output_file, bbox_inches="tight", format='png')

    print(f"Graph saved to {output_file}")


def to_latex_subscript(parameter):
    """
    LaTeX形式で変数を下付き文字として変換する
    """
    return f"T$_{{\\text{{{parameter[2:]}}}}}$"  # 'T_'の後ろを下付き文字として変換


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
