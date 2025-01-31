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
    output_dir = './mlresults_analyze'

    data_file = 'format_benchmark_parameter_loocv.csv'

    # データの読み込み
    df = load_data(data_dir, data_file)

    # Parameterごとの統計量の計算
    results_df = calculate_stats(df)

    # 結果を保存
    save_results(results_df, output_dir, "benchmark_parameter_stats_results_with_metadata.csv")

    # ベンチマーク項目の定義
    matrix_benchmarks = ["T_SCO", "T_MCO", "T_SMO", "T_MMO", "T_SAO", "T_MAO"]
    transfer_benchmarks = ["T_SLT", "T_BST", "T_IST"]

    # Variable Parameterを制約条件にフィルタリング
    filtered_df = filter_variable_parameters(results_df, matrix_benchmarks, transfer_benchmarks)

    # フィルタリング結果を保存
    save_results(filtered_df, output_dir,
                 "benchmark_parameter_stats_results_filterd_by_variable_toc.csv")

    data_file = 'format_spec_parameter_loocv.csv'
    # データの読み込み
    df = load_data(data_dir, data_file)
    # Parameterごとの統計量の計算
    results_df = calculate_stats(df, has_time_cost=False)
    # 結果を保存
    save_results(results_df, output_dir, "spec_mape_test_stats_results_with_metadata.csv")


def calculate_stats(df, has_time_cost=True):
    """
    各Variable Parameterの統計量を計算する関数。

    Args:
        df (pd.DataFrame): 入力データフレーム
        has_time_cost (bool): データフレームに "Time Cost (s)" 列が含まれるかどうか

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
        low_mape_list, high_mape_list = filter_leave_one_by_mape(filtered_df)

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

        # "Time Cost (s)" を含む場合にのみ計算
        time_cost = (round(filtered_df["Time Cost (s)"].iloc[0], 5)
                     if has_time_cost and not filtered_df["Time Cost (s)"].empty else None)

        # 結果をリストに追加
        result = {
            'ML': ml,
            'loss': loss,
            'Parameter Num': parameter_num,
            'Const Parameter': const_parameters,
            'Variable Parameter Num': variable_parameter_num,
            'Variable Parameter': variable_parameters,
        }

        # "Time Cost (s)" を含む場合、Variable Parameter の右隣に追加
        if has_time_cost:
            result["Time Cost (s)"] = time_cost

        result.update({
            "average MAPE test (%)": avg_mape_test,
            "min MAPE test (%)": min_mape_test,
            "max MAPE test (%)": max_mape_test,
            "Low Mape Server": low_mape_list,
            "High Mape Server": high_mape_list
        })

        results.append(result)
        results_df = pd.DataFrame(results).sort_values(by="average MAPE test (%)", ascending=True)

    return results_df


def filter_leave_one_by_mape(df, low_threshold=10, high_threshold=20):
    """
    MAPE test (%) を基準に Leave One のリストを作成する関数。

    Args:
        df : データフレーム。
        low_threshold (float): MAPE test (%) の下限値。
        high_threshold (float): MAPE test (%) の上限値。

    Returns:
        tuple: (low_mape_list, high_mape_list)
            - low_mape_list: MAPE test (%) が low_threshold 以下の Leave One リスト。
            - high_mape_list: MAPE test (%) が high_threshold 以上の Leave One リスト。
    """
    # DataFrame の作成

    # 条件に基づくリスト作成
    low_mape_list = df[df["MAPE test (%)"] <= low_threshold]["Leave One"].tolist()
    high_mape_list = df[df["MAPE test (%)"] >= high_threshold]["Leave One"].tolist()

    return low_mape_list, high_mape_list


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


if __name__ == "__main__":
    main()
