import ast
import os

import matplotlib.pyplot as plt
import pandas as pd

# 出力ディレクトリ
output_dir = './shap'


def extract_and_average_shap(shap_value_str):
    """
    SHAP値のJSON文字列を解析し、変数ごとの平均と合計を返す。
    """
    # SHAPのJSON文字列を辞書に変換
    shap_values = eval(shap_value_str)  # evalはセキュリティリスクがあるため、安全なデータでのみ使用すること

    # 変数ごとに平均を取る
    parameter_names = []
    mean_shap_values = []
    sum_shap_values = []

    for shap_entry in shap_values:
        parameter_names.append(shap_entry['Parameter'])
        mean_shap_values.append(shap_entry['Mean Absolute SHAP Value'])
        sum_shap_values.append(shap_entry['Sum Absolute SHAP Value'])

    return parameter_names, mean_shap_values, sum_shap_values


def visualize_shap_mean(shap_mean, variable_parameter):
    """
    SHAP値の平均を可視化する。
    """
    # 入力をリストとして評価
    inputs_list = ast.literal_eval(variable_parameter)
    print(inputs_list)

    # リスト内の要素を変換してLaTeX形式で表示
    latex_inputs = [to_latex_subscript(input_str) for input_str in inputs_list]

    # グラフの設定
    plt.figure(figsize=(10, 6))
    shap_mean.sort_values('Mean Absolute SHAP Value',
                          ascending=False).plot(kind='bar',
                                                y='Mean Absolute SHAP Value',
                                                legend=False,
                                                color='skyblue')
    plt.title(f"Mean Absolute SHAP Values for \n{latex_inputs}")
    plt.xlabel("Parameter")
    plt.ylabel("Mean Absolute SHAP Value")
    plt.xticks(rotation=0, ha='right')
    plt.tight_layout()

    # グラフを保存
    output_file = f"shap_mean_{variable_parameter}.png"
    output_path = os.path.join(output_dir, output_file)
    plt.savefig(output_path)

    print(f"Graph saved as {output_file}")


def to_latex_subscript(parameter):
    """
    LaTeX形式で変数を下付き文字として変換する
    """
    return f"$T_{{{parameter[2:]}}}$"  # 'T_'の後ろを下付き文字として変換


def main():
    csv_file = 'soturon_shap_graph.csv'
    data_dir = './mldata_analyze'

    # 入力ファイルのパスを設定
    csv_path = os.path.join(data_dir, csv_file)

    # データを読み込む
    df = pd.read_csv(csv_path)

    # ユニークなVariable Parameterの抽出
    unique_inputs = df['Variable Parameter'].unique()

    for inputs in unique_inputs:
        # Variable Parameterに基づいてフィルタリング
        filtered_df = df[df['Variable Parameter'] == inputs]

        # SHAP値の抽出と集計
        shap_values = filtered_df['SHAP Value'].apply(extract_and_average_shap)

        # 結果をまとめる
        mean_shap_values_all = []
        sum_shap_values_all = []
        parameter_names_all = []

        for parameter_names, mean_shap, sum_shap in shap_values:
            mean_shap_values_all.extend(mean_shap)
            sum_shap_values_all.extend(sum_shap)
            parameter_names_all.extend(parameter_names)

        # データフレームにまとめる
        shap_df = pd.DataFrame({
            'Parameter': parameter_names_all,
            'Mean Absolute SHAP Value': mean_shap_values_all,
            'Sum Absolute SHAP Value': sum_shap_values_all
        })

        # Parameter列をLaTeX形式に変換
        shap_df['Parameter'] = shap_df['Parameter'].apply(to_latex_subscript)

        # 平均値を計算
        shap_mean = shap_df.groupby('Parameter').mean()

        # 結果の確認
        print(f"Results for Variable Parameter: {inputs}")
        print(shap_mean)

        # 平均値の可視化
        visualize_shap_mean(shap_mean, inputs)


if __name__ == "__main__":
    main()
