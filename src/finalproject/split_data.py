import pandas as pd

def train_test_split(df, test_size):
    """
    Split the dataframe into train and test sets based on time.
    Assumes df is sorted by time.
    """
    split_index = int(len(df) * (1 - test_size))
    train_df = df.iloc[:split_index]
    test_df = df.iloc[split_index:]
    return train_df, test_df
