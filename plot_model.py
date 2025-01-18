import lightgbm as lgb
import matplotlib.pyplot as plt
import os


def plot_model(model):
    print(model.feature_importance())

    # 重要度としては「特徴量が分岐（ノード）の条件式で使用された回数」（＝デフォルト）
    lgb.plot_importance(model, figsize=(
        15, 4), max_num_features=15, importance_type='split')

    # 重要度としては「特徴量がある分岐（ノード）において目的関数の改善に寄与した度合い」
    lgb.plot_importance(model, figsize=(
        15, 4), max_num_features=15, importance_type='gain')
    output_dir = './importance_plt'
    os.makedirs(output_dir, exist_ok=True)
    i = sum(os.path.isfile(os.path.join(output_dir, name))
            for name in os.listdir(output_dir))
    output_file = f'test{i}.png'
    output = os.path.join(output_dir, output_file)
    plt.savefig(output)
    plt.clf()
    plt.close()
    return 0
