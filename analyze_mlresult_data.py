import os
import sys

import pandas as pd


def main():
    """
    メイン処理を行う関数。
    データの読み込み、統計量の計算、フィルタリング、結果保存を行う。
    """
    # ディレクトリとファイル名の設定
    data_dir = './ml_results'
    output_dir = './mldata_analyze'
    data_file = 'benchmark_parameter_loocv.csv'

    # データの読み込み
    df = load_data(data_dir, data_file)

    # Parameterごとの統計量の計算
    results_df = calculate_stats(df)

    # 結果を平均MAPEで昇順に並べ替え
    results_df = results_df.sort_values(by="average MAPE test (%)", ascending=True)

    # 結果を保存
    save_results(results_df, output_dir, "mape_test_stats_results_with_metadata.csv")

    # MAPEの範囲を指定（例：最小値から+1(%)の範囲）
    range_delta = 1

    # 指定したMAPE範囲でフィルタリング
    filtered_df = filter_by_mape_range(results_df, range_delta)

    # フィルタリングされた結果を保存
    save_results(filtered_df, output_dir, "mape_test_filtered_results.csv")

    # ベンチマーク項目の定義
    matrix_benchmarks = ["T_SLCO", "T_CSCO", "T_SLMO", "T_CSMO", "T_SLAO", "T_CSAO"]
    transfer_benchmarks = ["T_SLMT", "T_CSMT", "T_ISMT"]

    # Variable Parameterを条件にフィルタリング
    filtered_df = filter_variable_parameters(results_df, matrix_benchmarks, transfer_benchmarks)

    # MAPEの範囲を指定（例：最小値から+1(%)の範囲）
    range_delta = 1

    # 指定したMAPE範囲でフィルタリング
    filtered_df = filter_by_mape_range(filtered_df, range_delta)

    # フィルタリング結果を保存
    save_results(filtered_df, output_dir, "filtered_by_variable_parameters_mape_test.csv")


def load_data(data_dir, data_file):
    """
    指定されたディレクトリからデータを読み込む関数。

    Args:
        data_dir (str): データディレクトリのパス
        data_file (str): データファイル名

    Returns:
        pd.DataFrame: 読み込んだデータフレーム
    """
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


def calculate_stats(df):
    """
    各Variable Parameterの統計量を計算する関数。

    Args:
        df (pd.DataFrame): 入力データフレーム

    Returns:
        pd.DataFrame: 統計量を計算した結果
    """
    unique_inputs = df['Variable Parameter'].unique()
    results = []

    for input_value in unique_inputs:
        filtered_df = df[df['Variable Parameter'] == input_value]

        # 各種指標を計算
        avg_mape_test = round(filtered_df["MAPE test (%)"].mean(), 5)
        min_mape_test = filtered_df["MAPE test (%)"].min()
        max_mape_test = filtered_df["MAPE test (%)"].max()

        # 他の列の代表値を取得
        ml = filtered_df["ML"].iloc[0] if not filtered_df["ML"].empty else None
        loss = filtered_df["loss"].iloc[0] if not filtered_df["loss"].empty else None
        parameter_num = filtered_df['Parameter Num'].iloc[
            0] if not filtered_df['Parameter Num'].empty else None
        const_parameters = filtered_df['Const Parameter'].iloc[
            0] if not filtered_df['Const Parameter'].empty else None
        variable_parameter_num = filtered_df['Variable Parameter Num'].iloc[
            0] if not filtered_df['Variable Parameter Num'].empty else None
        variable_parameters = filtered_df['Variable Parameter'].iloc[
            0] if not filtered_df['Variable Parameter'].empty else None
        time_cost = round(filtered_df["Time Cost (s)"].iloc[0],
                          5) if not filtered_df["Time Cost (s)"].empty else None

        # 結果をリストに追加
        results.append({
            'ML': ml,
            'loss': loss,
            'Parameter Num': parameter_num,
            'Const Parameter': const_parameters,
            'Variable Parameter Num': variable_parameter_num,
            'Variable Parameter': variable_parameters,
            'Time Cost (s)': time_cost,
            "average MAPE test (%)": avg_mape_test,
            "min MAPE test (%)": min_mape_test,
            "max MAPE test (%)": max_mape_test
        })

    return pd.DataFrame(results)


def save_results(df, output_dir, output_file):
    """
    データフレームを指定されたディレクトリに保存する関数。

    Args:
        df (pd.DataFrame): 保存するデータフレーム
        output_dir (str): 保存先ディレクトリのパス
        output_file (str): 保存するファイル名
    """
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, output_file)
    df.to_csv(output_path, index=False)
    print(f"結果を {output_path} に保存しました。")


def filter_by_mape_range(df, range_delta):
    """
    平均MAPEの範囲を指定してフィルタリングする関数。

    Args:
        df (pd.DataFrame): 入力データフレーム
        range_delta (float): 平均MAPEの範囲を指定

    Returns:
        pd.DataFrame: フィルタリングされたデータフレーム
    """
    # 平均MAPEの最小値を取得
    min_mape_value = df['average MAPE test (%)'].min()

    # 最小値からの範囲の指定
    max_mape_value = min_mape_value + range_delta

    # 指定した範囲内でフィルタリング
    filtered_df = df[(df['average MAPE test (%)'] >= min_mape_value) &
                     (df['average MAPE test (%)'] <= max_mape_value)]
    # Time Cost (s)の小さい順に並べ替え
    #filtered_df = filtered_df.sort_values(by="Time Cost (s)", ascending=True)

    return filtered_df


def filter_variable_parameters(df, matrix_benchmarks, transfer_benchmarks):
    """
    Variable Parameterに行列計算と転送ベンチマーク項目を含む行を抽出する関数。

    Args:
        df (pd.DataFrame): 入力データフレーム
        matrix_benchmarks (list): 行列計算に関するベンチマーク項目のリスト
        transfer_benchmarks (list): 転送に関するベンチマーク項目のリスト

    Returns:
        pd.DataFrame: 条件を満たす行を抽出したデータフレーム
    """

    # ベンチマークが含まれているかを判定するヘルパー関数
    def contains_any(item_list, benchmarks):
        return any(benchmark in item_list for benchmark in benchmarks)

    # Variable Parameter列をリスト化
    df["Variable Parameter List"] = df["Variable Parameter"].apply(eval)

    # 行列計算ベンチマークと転送ベンチマークの両方を含む行をフィルタリング
    filtered_df = df[
        df["Variable Parameter List"].apply(lambda x: contains_any(x, matrix_benchmarks)) &
        df["Variable Parameter List"].apply(lambda x: contains_any(x, transfer_benchmarks))]

    # 不要な中間列を削除
    filtered_df = filtered_df.drop(columns=["Variable Parameter List"])
    return filtered_df


if __name__ == "__main__":
    main()
