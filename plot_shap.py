import lightgbm as lgb
import matplotlib.pyplot as plt
import shap


# SHAPの可視化
def plot_shap(model, X_test):
    """
    shap.TreeExplainer:決定木用(XGBoost、lightBGM等含む)
    shap.LinearExplainer :線形モデル用
    shap.DeepExplainer :Deeplearning用
    """
    #TreeExplainerは、決定木系のモデルのSHAP値を取得するもの。
    explainer = shap.TreeExplainer(model=model)
    X_test_shap = X_test.copy().reset_index(drop=True)
    shap_values = explainer.shap_values(X=X_test_shap)
    shap.summary_plot(shap_values, X_test_shap)  #左側の図
    plt.show()
    shap.summary_plot(shap_values, X_test_shap, plot_type='bar')
    plt.show()
    return 0
