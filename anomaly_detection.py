import matplotlib.pyplot as plt
import seaborn as sns
from pyod.models.lof import LOF
from sklearn.preprocessing import StandardScaler
from data_extraction import extract_data, create_features


def detect_anomalies(features):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(features)

    lof = LOF(n_neighbors=20, contamination=0.05)
    lof.fit(X_scaled)

    anomaly_scores = lof.decision_function(X_scaled)
    is_anomaly = lof.predict(X_scaled)
    return anomaly_scores, is_anomaly

if __name__ == '__main__':
    # 步骤 1: 提取数据
    df = extract_data()

    # 步骤 2: 创建特征
    features = create_features(df)
    df = df.loc[features.index]  # 同步更新 df 的行数 feature中有nan数据

    # 利用LOF算法得anomaly列数据
    anomaly_scores, is_anomaly = detect_anomalies(features)
    df['anomaly_score'] = anomaly_scores
    df['is_anomaly'] = is_anomaly

    # 绘制散点图
    plt.figure(figsize=(12, 8))
    sns.scatterplot(data=df, x='byte_ratio', y='hour', hue='is_anomaly', palette={0: 'blue', 1: 'red'}, alpha=0.6)
    plt.title('Scatter Plot of Anomaly Detection Results')
    plt.xlabel('Byte Ratio')
    plt.ylabel('Hour of Day')
    plt.show()
