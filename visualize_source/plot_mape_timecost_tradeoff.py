import ast

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

max_mape_list = ['T_MCO', 'T_SMO', 'T_MAO']
trade_off_list = ['T_MCO']
cost_on_list = ['T_SCO']


def main():
    # ディレクトリとファイル名の設定
    data_dir = '../mlresults_analyze'
    output_dir = '../soturon_graph'

    data_file = 'benchmark_parameter_stats_results_with_metadata.csv'

    # データの読み込み
    df = load_data(data_dir, data_file)
    df['Variable Parameter'] = df['Variable Parameter'].apply(lambda x: ast.literal_eval(x)
                                                              if isinstance(x, str) else x)

    # 条件: 'Variable Parameter' が max_mape_list と一致する場合
    condition_1 = df['Variable Parameter'].apply(lambda x: set(x) == set(max_mape_list))
    condition_2 = df['Variable Parameter'].apply(lambda x: set(x) == set(trade_off_list))
    condition_3 = df['Variable Parameter'].apply(lambda x: set(x) == set(cost_on_list))
    condition_other = ~condition_1 & ~condition_2 & ~condition_3  # その他は青

    # # グラフの描画
    fig, ax = plt.subplots(figsize=(7, 5))

    # 青の点（それ以外）
    ax.scatter(df['Time Cost (s)'][condition_other],
               df['average MAPE test (%)'][condition_other],
               color='lightblue',
               label='Other')

    # 各モデルにラベルを付けるためにデータフレームを加工
    def get_label(param_list):
        return f"[{','.join([to_latex_subscript(param) for param in param_list])}]"

    # 赤の点 max
    scatter_max = ax.scatter(df['Time Cost (s)'][condition_1],
                             df['average MAPE test (%)'][condition_1],
                             color='red',
                             label=get_label(max_mape_list))

    # 青の点 trade_off
    scatter_trade_off = ax.scatter(df['Time Cost (s)'][condition_2],
                                   df['average MAPE test (%)'][condition_2],
                                   color='blue',
                                   label=get_label(trade_off_list))
    #オレンジcletの点
    scatter_cost_on = ax.scatter(df['Time Cost (s)'][condition_3],
                                 df['average MAPE test (%)'][condition_3],
                                 color='orange',
                                 label=get_label(cost_on_list))

    # 各点の上に 'average MAPE test (%)' の値を表示（lightblueを除く）
    def adjust_text_position(x, y, color):
        """テキストが点と重ならないようにオフセットを調整する関数"""
        y_offset = 0.5  # y軸方向にオフセット
        if color == 'red':
            return x, y + y_offset
        elif color == 'blue':
            return x, y + y_offset
        elif color == 'orange':
            return x, y + y_offset
        return x, y  # lightblueはテキストを表示しないのでそのまま

    for scatter, condition, color in [(scatter_max, condition_1, 'red'),
                                      (scatter_trade_off, condition_2, 'blue'),
                                      (scatter_cost_on, condition_3, 'orange')]:
        for i in range(len(df)):
            if condition.iloc[i]:
                mape_value = df['average MAPE test (%)'].iloc[i]
                x, y = df['Time Cost (s)'].iloc[i], mape_value
                # テキスト位置調整
                x_adj, y_adj = adjust_text_position(x, y, color)
                ax.text(x_adj,
                        y_adj,
                        f'{mape_value:.2f}%',
                        color=color,
                        fontsize=12,
                        ha='left',
                        va='center')

    # グラフの設定
    #ax.set_title('MAPE vs Time Cost', fontsize=14)
    ax.set_xlabel('Time Cost (s)', fontsize=12)
    ax.set_ylabel('average MAPE test (%)', fontsize=12)
    ax.grid(True)

    # 軸の範囲調整
    ax.set_xlim(0, max(df['Time Cost (s)']) + 100)
    ax.set_ylim(min(df['average MAPE test (%)']) - 0.5, max(df['average MAPE test (%)']) + 0.5)
    ax.legend()
    plt.tight_layout()
    plt.subplots_adjust(right=0.93)  # 余白調整
    # 画像として保存
    save_plot(fig, output_dir, 'timecost_mape_tradeoff.png')
    # 表示
    plt.show()


def load_data(data_dir, data_file):
    """
    指定されたディレクトリからデータを読み込む関数。

    Args:
        data_dir (str): データディレクトリのパス
        data_file (str): データファイル名

    Returns:
        pd.DataFrame: 読み込んだデータフレーム
    """
    import os
    import sys
    try:
        data_path = os.path.join(data_dir, data_file)
        df = pd.read_csv(data_path)

        # 欠損値の確認
        if df.isna().any().any():
            raise ValueError("データに欠損値（NaN）が含まれています。")
        return df
    except FileNotFoundError:
        print(f"ファイル {data_file} が見つかりませんでした。")
        sys.exit(1)
    except ValueError as e:
        print(e)
        sys.exit(1)


def to_latex_subscript(parameter):
    """
    LaTeX形式で変数を下付き文字として変換する
    """
    return f"T$_{{\\text{{{parameter[2:]}}}}}$"  # 'T_'の後ろを下付き文字として変換


def save_plot(fig, output_dir, output_file):
    """
    グラフを指定されたディレクトリに保存する関数。
    ディレクトリが存在しない場合は自動的に作成される。

    Args:
        fig (matplotlib.figure.Figure): 保存するグラフ
        output_dir (str): 保存先ディレクトリ
        variable_parameter (str): グラフファイルの名前に使用する変数パラメータ
    """
    import os

    # 出力ディレクトリが存在しない場合は作成する
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    output_path = os.path.join(output_dir, output_file)

    # グラフを保存
    fig.savefig(output_path)
    print(f"グラフを保存しました: {output_path}")


if __name__ == "__main__":
    main()
