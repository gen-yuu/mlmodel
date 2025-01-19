import pandas as pd
import os
import sys


# データの読み込み
def load_data(data_dir, data_file):
    data_path = os.path.join(data_dir, data_file)
    df = pd.read_csv(data_path)

    # 欠損値（NaN）があるか確認
    if df.isna().any().any():
        print("データに欠損値（NaN）が含まれています。プログラムを終了します。")
        sys.exit()  # 欠損値があればプログラムを終了

    return df


# 各Inputの統計量を計算
def calculate_stats(df):
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


# 結果を指定したファイルに保存
def save_results(df, output_dir, output_file):
    output_path = os.path.join(output_dir, output_file)
    df.to_csv(output_path, index=False)
    print(f"結果を {output_path} に保存しました。")


# MAPEの指定範囲で抽出
def filter_by_mape_range(df, range_delta):
    # MAPEの最小値を取得
    min_mape_value = df['average MAPE test (%)'].min()

    # max_mape_value = min_mape_value + range_delta を適用
    max_mape_value = min_mape_value + range_delta

    # 指定した範囲内でフィルタリング
    filtered_df = df[(df['average MAPE test (%)'] >= min_mape_value) &
                     (df['average MAPE test (%)'] <= max_mape_value)]
    # Time Cost (s)の小さい順に並べ替え
    filtered_df = filtered_df.sort_values(by="Time Cost (s)", ascending=True)

    return filtered_df


def main():
    data_dir = './ml_results'
    output_dir = './mldata_analyze'
    data_file = 'benchmark_parameter_loocv.csv'

    # データの読み込み
    df = load_data(data_dir, data_file)

    # 統計量の計算
    results_df = calculate_stats(df)

    # 結果を平均MAPEで昇順に並べ替え
    results_df = results_df.sort_values(by="average MAPE test (%)", ascending=True)

    # 結果を保存
    save_results(results_df, output_dir, "mape_test_stats_results_with_metadata.csv")

    # MAPEの範囲を指定（例：最小値から+1(%)の範囲）
    range_delta = 1

    # MAPE指定範囲でフィルタリング
    filtered_df = filter_by_mape_range(results_df, range_delta)

    # フィルタリングされた結果を保存
    save_results(filtered_df, output_dir, "mape_test_filtered_results.csv")


if __name__ == "__main__":
    main()
