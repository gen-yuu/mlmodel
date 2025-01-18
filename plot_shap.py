import lightgbm as lgb
import matplotlib.pyplot as plt
import shap
import os

# SHAPの可視化


def plot_shap(model, X_test):
    """
    shap.TreeExplainer:決定木用(XGBoost、lightBGM等含む)
    shap.LinearExplainer :線形モデル用
    shap.DeepExplainer :Deeplearning用
    """
    # TreeExplainerは、決定木系のモデルのSHAP値を取得するもの。
    explainer = shap.TreeExplainer(model=model)
    X_test_shap = X_test.copy().reset_index(drop=True)
    shap_values = explainer.shap_values(X=X_test_shap)

    # shap.summary_plot(shap_values, X_test_shap)
    output_dir = './shap_plt'
    os.makedirs(output_dir, exist_ok=True)
    i = sum(os.path.isfile(os.path.join(output_dir, name))
            for name in os.listdir(output_dir))
    output_file = f'test{i}.png'
    output = os.path.join(output_dir, output_file)
    shap.summary_plot(shap_values, X_test_shap, plot_type='bar', show=False)
    plt.savefig(output)
    return 0
