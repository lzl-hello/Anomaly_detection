import lime
import lime.lime_tabular
import shap


def explain_with_lime(features, model, index, X_train, X_scaled):
    explainer = lime.lime_tabular.LimeTabularExplainer(
        training_data=X_train,
        feature_names=features.columns,
        class_names=['Normal', 'Anomaly'],
        discretize_continuous=True
    )

    # 解释某个特定实例
    exp = explainer.explain_instance(
        data_row=X_scaled[index],
        predict_fn=model.predict_proba,
        num_features=5
    )

    # 显示解释结果
    # exp.show_in_notebook(show_table=True, show_all=False)
    print(exp.as_list())


def explain_with_shap(features, model, X_test):
    # 生成SHAP解释器
    explainer = shap.TreeExplainer(model)

    # 计算SHAP值
    shap_values = explainer.shap_values(X_test)

    # 绘制SHAP解释图
    # shap.summary_plot(shap_values, X_test, feature_names=features.columns)
    shap.summary_plot(shap_values, X_test, feature_names=features.columns, max_display=features.shape[1])





