import ast
import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.ticker import MaxNLocator  # MaxNLocatorをインポート
from config import SERVER_ORDER, PLT_FONT

plt.rcParams['font.family'] = PLT_FONT

csv_file = 'soturon_shap_graph.csv'
data_dir = '../mlresults_analyze'

# 入力ファイルのパスを設定
csv_path = os.path.join(data_dir, csv_file)

# データを読み込む
df = pd.read_csv(csv_path)

# ユニークなVariable Parameterの抽出
unique_inputs = df['Variable Parameter'].unique()

# "Leave One" 列の値を修正（正規表現を使用）
df["Leave One"] = df["Leave One"].replace(
    {
        r'corei5': 'Core i5',
        r'corei7': 'Core i7',
        r'corei9': 'Core i9'
    }, regex=True)


def to_latex_subscript(parameter):
    """
    LaTeX形式で変数を下付き文字として変換する
    """
    return f"T$_{{\\text{{{parameter[2:]}}}}}$"  # 'T_'の後ろを下付き文字として変換


# 色設定（関数を使ってキーを生成）
parameter_colors = {
    to_latex_subscript("T_MCO"): "#1f77b4",  # 青
    to_latex_subscript("T_SMO"): "#2ca02c",  # 緑
    to_latex_subscript("T_MAO"): "#9467bd",  # 紫
    to_latex_subscript("T_BST"): "#ff7f0e",  # オレンジ
    to_latex_subscript("T_SAO"): "#d62728",  # 赤
}
# グラフ保存ディレクトリの指定
output_dir = '../soturon_graph'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# 1週目と2週目のPNGを保存するためのカウント
roop_counter = 1

for inputs in unique_inputs:
    # Variable Parameterに基づいてフィルタリング
    filtered_df = df[df['Variable Parameter'] == inputs]

    # SHAPの値をリストから辞書として解析
    filtered_df['SHAP Value'] = filtered_df['SHAP Value'].apply(lambda x: ast.literal_eval(x))

    # 各行のMean Absolute SHAP Valueを抽出
    shap_data = []
    for index, row in filtered_df.iterrows():
        for shap_entry in row['SHAP Value']:
            shap_data.append({
                'Parameter': shap_entry['Parameter'],
                'Mean Absolute SHAP Value': shap_entry['Mean Absolute SHAP Value'],
                'Server': row['Leave One']
            })

    # SHAPデータを新しいDataFrameに変換
    shap_df = pd.DataFrame(shap_data)

    # Parameter列をLaTeX形式に変換
    shap_df['Parameter'] = shap_df['Parameter'].apply(to_latex_subscript)

    # グラフ作成: 各サーバーごとにパラメータのSHAP値を棒グラフで表示
    plt.figure(figsize=(8, 6))  # 各グラフごとにサイズを設定

    # 軸オブジェクトを取得
    ax = plt.gca()

    # グリッドを表示（zorder=0 で棒グラフの後ろ）
    ax.grid(axis='y', linestyle='--', zorder=1)

    # 棒グラフを表示（zorder=1 でグリッドの後ろ）
    sns.barplot(
        data=shap_df,
        x='Server',
        y='Mean Absolute SHAP Value',
        hue='Parameter',
        ci=None,
        palette=parameter_colors,  # ここで色を設定
        order=SERVER_ORDER,
        zorder=2)

    # y軸の補助線の間隔を固定（例として10間隔に設定）
    ax.yaxis.set_major_locator(MaxNLocator(integer=True, prune='lower', steps=[1, 5,
                                                                               10]))  # 10ごとの補助線
    # タイトルや設定
    #plt.title(f'Mean Absolute SHAP Values by Parameter and Server ({inputs})', fontsize=16)
    plt.xlabel('Test Server', fontsize=12)
    plt.ylabel('Mean Absolute SHAP Value', fontsize=12)
    plt.xticks(rotation=45, ha="right", fontsize=10)
    # 凡例の位置を右上に固定
    plt.legend(loc='upper left')

    plt.tight_layout()
    plt.subplots_adjust(left=0.14, right=0.89, bottom=0.29, top=0.99)  # 余白調整

    # 保存ファイル名の設定
    if roop_counter == 1:
        save_path = os.path.join(output_dir, "shap_graph_operation.png")
    else:
        save_path = os.path.join(output_dir, "shap_graph_transfer_and_operation.png")

    # 画像保存
    plt.savefig(save_path, bbox_inches=None)  # PNG形式で保存

    # 現在の図をリセット
    plt.clf()  # 図をクリア
    plt.close()  # 図を閉じる

    # 週数のカウントを進める
    roop_counter += 1
    if roop_counter > 2:
        roop_counter = 1  # 1週目と2週目でローテーション
