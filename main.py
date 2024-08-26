import seaborn as sns
import matplotlib.pyplot as plt
from data_extraction import extract_data, create_features
from anomaly_detection import detect_anomalies
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from model_explainability import explain_with_lime, explain_with_shap

# 步骤 1: 提取数据
df = extract_data()

# 步骤 2: 创建特征
features = create_features(df)
df = df.loc[features.index]  # 同步更新 df 的行数 feature中有nan数据

# 利用LOF算法得anomaly列数据
anomaly_scores, is_anomaly = detect_anomalies(features)
df['anomaly_score'] = anomaly_scores
df['is_anomaly'] = is_anomaly

# 步骤 3: 标准化和分割数据集
scaler = StandardScaler()
X_scaled = scaler.fit_transform(features)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, df['is_anomaly'], test_size=0.3, random_state=42)

# 训练一个随机森林分类器
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 解释模型
index = df[df['is_anomaly'] == 1].index[0]  # 选择第一个检测到的异常点
explain_with_lime(features, model, index, X_train, X_scaled)

explain_with_shap(features, model, X_test)

# 预测测试集的标签
y_pred = model.predict(X_test)

# 打印分类报告
print("Classification Report:")
print(classification_report(y_test, y_pred))

# 打印混淆矩阵
conf_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(conf_matrix)

# 绘制混淆矩阵
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Normal', 'Anomaly'], yticklabels=['Normal', 'Anomaly'])
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.show()
