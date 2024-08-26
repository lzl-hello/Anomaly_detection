import pandas as pd
from sqlalchemy import create_engine


def extract_data():
    engine = create_engine('postgresql+psycopg2://postgres:root@localhost/hit_log')

    query = "SELECT * FROM http_log;"
    df = pd.read_sql(query, engine)
    return df


def create_features(df):
    # 将 time_log 列先转换为字符串
    df['time_log'] = df['time_log'].astype(str)

    # 使用 errors='coerce' 来忽略无法转换的时间戳，并统一转换为 UTC
    df['time_log'] = pd.to_datetime(df['time_log'], errors='coerce', utc=True)

    # 生成特征
    df['byte_ratio'] = df['ctos'] / (df['stoc'] + 1)
    df['hour'] = df['time_log'].dt.hour
    df['day_of_week'] = df['time_log'].dt.dayofweek
    df['protocol_encoded'] = df['protocol'].astype('category').cat.codes
    features = df[['byte_ratio', 'Direction', 'hour', 'day_of_week', 'protocol_encoded']]
    # 提取特征后，处理 NaN 值
    features = features.dropna()

    return features


if __name__ == '__main__':
    df = extract_data()
    print(df.head())
    features = create_features(df)
    print(features.head(10))
    print(features.tail(10))
