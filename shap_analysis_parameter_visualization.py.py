import pandas as pd
import ast
import matplotlib.pyplot as plt
import seaborn as sns
import os
from matplotlib.ticker import MaxNLocator  # MaxNLocatorをインポート

csv_file = 'soturon_shap_graph.csv'
data_dir = './mldata_analyze'

# 入力ファイルのパスを設定
csv_path = os.path.join(data_dir, csv_file)

# データを読み込む
df = pd.read_csv(csv_path)

# ユニークなVariable Parameterの抽出
unique_inputs = df['Variable Parameter'].unique()

# サーバー順序を指定
server_order = [
    "13th corei5 - GTX1080", "13th corei5 - GTX1650", "13th corei5 - RTX3050",
    "13th corei5 - RTX3060 Ti", "13th corei5 - RTX4070", "13th corei7 - GTX1080",
    "13th corei7 - RTX3050", "13th corei7 - RTX3060 Ti", "13th corei7 - RTX4070",
    "1th Xeon Gold - GTX1080", "1th Xeon Gold - RTX4070", "9th corei7 - RTX2080 Ti"
]


# LaTeX形式でのパラメータ名を変換する関数
def to_latex_subscript(parameter):
    """
    T_XXX 形式の文字列を $T_{XXX}$ 形式に変換
    """
    return f"$T_{{{parameter[2:]}}}$"


# 色設定
parameter_colors = {
    "$T_{MCO}$": "#1f77b4",  # 青
    "$T_{SMO}$": "#2ca02c",  # 緑
    "$T_{MAO}$": "#9467bd",  # 紫
    "$T_{BST}$": "#ff7f0e",  # オレンジ
    "$T_{SAO}$": "#d62728",  # 赤
}

# グラフ保存ディレクトリの指定
output_dir = './soturon_graph_data'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

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
    plt.figure(figsize=(16, 8))  # 各グラフごとにサイズを設定

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
        order=server_order,
        zorder=2)

    # y軸の補助線の間隔を固定（例として10間隔に設定）
    ax.yaxis.set_major_locator(MaxNLocator(integer=True, prune='lower', steps=[1, 5,
                                                                               10]))  # 10ごとの補助線
    # タイトルや設定
    #plt.title(f'Mean Absolute SHAP Values by Parameter and Server ({inputs})', fontsize=16)
    plt.xticks(rotation=30, ha="right", fontsize=9)
    plt.tight_layout()

    # 画像保存
    save_path = os.path.join(output_dir, f"shap_graph_{inputs}.png")
    plt.savefig(save_path, bbox_inches='tight')  # PNG形式で保存

    # 現在の図をリセット
    plt.clf()  # 図をクリア
    plt.close()  # 図を閉じる
