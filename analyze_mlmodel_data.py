import pandas as pd
import os


# メイン処理をまとめる関数
def main():
    # データディレクトリと出力ディレクトリ
    data_dir = './ml_results'
    output_dir = './mldata_analyze'

    # データファイル名
    data_file = 'benchmark_feature_loocv.csv'  # 実際のファイルパスに置き換えてください
    data_path = os.path.join(data_dir, data_file)

    # CSVファイルを読み込む
    df = pd.read_csv(data_path)

    # ユニークなInputの値を取得
    unique_inputs = df["Input"].unique()

    # 結果を保存するリスト
    results = []

    # 各Input値に対して集計処理
    for input_value in unique_inputs:
        # 指定したInput値でフィルタリング
        filtered_df = df[df["Input"] == input_value]

        # 各種指標を計算
        avg_mape_test = round(filtered_df["MAPE test (%)"].mean(), 5)
        min_mape_test = filtered_df["MAPE test (%)"].min()
        max_mape_test = filtered_df["MAPE test (%)"].max()

        # 他の列の代表値を取得
        ml = filtered_df["ML"].iloc[0] if not filtered_df["ML"].empty else None
        loss = filtered_df["loss"].iloc[0] if not filtered_df["loss"].empty else None
        input_num = filtered_df["Input Num"].iloc[0] if not filtered_df["Input Num"].empty else None
        time_cost = round(filtered_df["Time Cost (s)"].mean(), 5)

        # 結果をリストに追加
        results.append({
            "ML": ml,
            "loss": loss,
            "Input Num": input_num,
            "Input": input_value,
            "Time Cost (s)": time_cost,
            "average MAPE test (%)": avg_mape_test,
            "min MAPE test (%)": min_mape_test,
            "max MAPE test (%)": max_mape_test
        })

    # 結果をDataFrameに変換
    results_df = pd.DataFrame(results)

    # average MAPE test (%) で昇順に並べ替え
    results_df = results_df.sort_values(by="average MAPE test (%)", ascending=True)

    # 結果を表示
    print(results_df)

    # 出力ファイルのパス
    output_file = "mape_test_stats_results_with_metadata.csv"
    output_path = os.path.join(output_dir, output_file)

    # 結果をCSVとして保存
    results_df.to_csv(output_path, index=False)
    print(f"結果を {output_path} に保存しました。")


if __name__ == "__main__":
    main()
