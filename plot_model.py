import lightgbm as lgb
import matplotlib.pyplot as plt


def plot_model(model):
    print(model.feature_importance())

    # 性能向上に寄与する度合いで重要度をプロット
    lgb.plot_importance(model, figsize=(8, 4), max_num_features=15, importance_type='gain')
    plt.show()
    return 0
