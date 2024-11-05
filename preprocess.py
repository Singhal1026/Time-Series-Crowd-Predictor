import numpy as np

def create_features(df, label=None):
    df = df.copy()
    df['date'] = df['datetime']
    df['hour'] = df['date'].dt.hour
    df['minute'] = df['date'].dt.minute
    df['dayofweek'] = df['date'].dt.dayofweek
    df['month'] = df['date'].dt.month
    df['dayofyear'] = df['date'].dt.dayofyear
    df['dayofmonth'] = df['date'].dt.day
    df['weekofyear'] = df['date'].dt.isocalendar().week

    X = df[['dayofweek', 'month', 'dayofyear', 'dayofmonth', 'weekofyear', 'camera_location', 'hour', 'minute', 'crowd_count']]
    
    if label:
        y = df[label]
        return X, y
    return X


def cylindrical_encoding(df):
    df = df.copy()

    columns_info = {
        'dayofweek': 7,
        'month': 12,
        'minute' : 59,
        'hour' : 23
    }

    for column, max_value in columns_info.items():
        df[f'{column}_sin'] = np.sin(2 * np.pi * df[column] / max_value)
        df[f'{column}_cos'] = np.cos(2 * np.pi * df[column] / max_value)

    df = df.drop(list(columns_info.keys()), axis=1)

    return df
