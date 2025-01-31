import os
import sys

import matplotlib.pyplot as plt
import pandas as pd


def main():
    data_dir = '../mlresults_analyze'
    output_dir = '../IN_graph'  # 保存先のディレクトリ
    os.makedirs(output_dir, exist_ok=True)  # 保存先フォルダを作成

    data_file = 'benchmark_parameter_stats_results_with_metadata.csv'

    # データの読み込み
    df = load_data(data_dir, data_file)

    # Variable Parameter をリスト形式に変換し、LaTeX スタイルにフォーマット
    df['Variable Parameter'] = df['Variable Parameter'].apply(eval)
    df['Variable Parameter'] = df['Variable Parameter'].apply(
        lambda params: [to_latex_subscript(param) for param in params])

    # 高精度モデルの条件（例: MAPE < 15%）
    high_precision_models = df[df['average MAPE test (%)'] < 15]

    # 特徴量の出現頻度をプロット
    plot_feature_importance(high_precision_models, output_dir)

    # MAPE vs Time Cost の散布図をプロット
    plot_mape_vs_time_cost(df, high_precision_models, output_dir)


def to_latex_subscript(parameter):
    """
    LaTeX形式で変数を下付き文字として変換する
    
    Args:
        parameter (str): 入力パラメータ（例: 'T_MCO'）
        
    Returns:
        str: LaTeX形式の下付き文字（例: '$T_{MCO}$'）
    """
    return f"$T_{{{parameter[2:]}}}$"


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


def plot_feature_importance(high_precision_models, output_dir):
    """
    高精度モデルにおける特徴量の重要度を棒グラフでプロットする。

    Args:
        high_precision_models (pd.DataFrame): 高精度モデルのデータ
        output_dir (str): グラフ保存先のディレクトリ
    """
    feature_counts = pd.Series([
        feature for features in high_precision_models['Variable Parameter'] for feature in features
    ]).value_counts()

    plt.figure(figsize=(8, 6))
    ax = plt.gca()
    ax.grid(axis='y', linestyle='--', zorder=1)  # グリッドを zorder=1 に設定
    ax.bar(feature_counts.index, feature_counts.values, zorder=2)  # 棒グラフ
    #plt.title('Feature Importance in High-Precision Models', fontsize=14)
    plt.xlabel('Benchmark', fontsize=12)
    plt.ylabel('The Number of Benchmark in High-Precision Models', fontsize=12)
    plt.xticks(rotation=0, ha='center', fontsize=10)
    save_path = os.path.join(output_dir, 'high_precision_benchmark_models.png')
    plt.savefig(save_path, bbox_inches='tight')  # PNG形式で保存
    plt.close()  # グラフを閉じる


def plot_mape_vs_time_cost(df, high_precision_models, output_dir):
    """
    MAPE vs Time Cost の散布図をプロットする。

    Args:
        df (pd.DataFrame): 全データフレーム
        high_precision_models (pd.DataFrame): 高精度モデルのデータ
        output_dir (str): グラフ保存先のディレクトリ
    """
    plt.figure(figsize=(10, 4))
    ax = plt.gca()
    ax.grid(True, zorder=1)  # グリッドを zorder=1 に設定
    plt.scatter(df['Time Cost (s)'],
                df['average MAPE test (%)'],
                color='lightgray',
                alpha=0.5,
                label='All Models',
                zorder=2)  # 散布図を zorder=2 に設定
    plt.scatter(high_precision_models['Time Cost (s)'],
                high_precision_models['average MAPE test (%)'],
                color='red',
                label='High-Precision Models',
                zorder=3)  # 高精度モデルを zorder=3 に設定

    plt.title('MAPE vs Time Cost', fontsize=14)
    plt.xlabel('Time Cost (s)', fontsize=12)
    plt.ylabel('average MAPE test (%)', fontsize=12)
    plt.legend()
    save_path = os.path.join(output_dir, 'mape_vs_time_cost_high_precision_model.png')
    plt.savefig(save_path, bbox_inches='tight')  # PNG形式で保存
    plt.close()  # グラフを閉じる


if __name__ == "__main__":
    main()
