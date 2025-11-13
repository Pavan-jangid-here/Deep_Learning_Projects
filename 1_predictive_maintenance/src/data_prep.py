import pandas as pd

def load_nasa_data(path):
    # NASA FD001, columns: unit, cycle, op, sensor_1 .. sensor_21
    col_names = ['unit', 'cycle'] + [f'op_{i}' for i in range(1,4)] + [f'sensor_{i}' for i in range(1,22)]
    df = pd.read_csv(path, sep="\s+", header=None)
    df.columns = col_names
    return df

def add_rul(df):
    max_cycle = df.groupby('unit')['cycle'].max().reset_index()
    max_cycle.columns = ['unit', 'max_cycle']
    df = df.merge(max_cycle, on='unit')
    df['RUL'] = df['max_cycle'] - df['cycle']
    df.drop('max_cycle', axis=1, inplace=True)
    return df

if __name__ == '__main__':
    df = load_nasa_data('data/train_FD001.txt')
    df = add_rul(df)
    df.to_csv('data/nasa_prepared.csv', index=False)
