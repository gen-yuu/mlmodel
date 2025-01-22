import matplotlib.pyplot as plt
import pandas as pd

import matplotlib.pyplot as plt
import numpy as np
import ast

max_mape_list = ['T_CSCO', 'T_SLMO', 'T_CSAO']
trade_off_list = ['T_CSCO', 'T_CSAO']
cost_on_list = ['T_SLCO']


def main():
    # ディレクトリとファイル名の設定
    data_dir = './mldata_analyze'
    output_dir = './mape_timecost_tradeoff'

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
    fig, ax = plt.subplots(figsize=(10, 6))

    # 青の点（それ以外）
    ax.scatter(df['Time Cost (s)'][condition_other],
               df['average MAPE test (%)'][condition_other],
               color='lightblue',
               label='Other')

    # 赤の点 max
    ax.scatter(df['Time Cost (s)'][condition_1],
               df['average MAPE test (%)'][condition_1],
               color='red',
               label=f"{', '.join(max_mape_list)}")

    # 青の点 trade_off
    ax.scatter(df['Time Cost (s)'][condition_2],
               df['average MAPE test (%)'][condition_2],
               color='blue',
               label=f"{', '.join(trade_off_list)}")
    #緑の点
    ax.scatter(df['Time Cost (s)'][condition_3],
               df['average MAPE test (%)'][condition_3],
               color='orange',
               label=f"{', '.join(cost_on_list)}")

    # グラフの設定
    ax.set_title('MAPE vs Time Cost', fontsize=14)
    ax.set_xlabel('Time Cost (s)', fontsize=12)
    ax.set_ylabel('average MAPE test (%)', fontsize=12)
    ax.grid(True)

    # 軸の範囲調整
    ax.set_xlim(0, max(df['Time Cost (s)']) + 100)
    ax.set_ylim(min(df['average MAPE test (%)']) - 0.5, max(df['average MAPE test (%)']) + 0.5)
    ax.legend()
    # 画像として保存
    save_plot(fig, output_dir, 'mape_timecost_tradeoff_all_blue.png')
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
