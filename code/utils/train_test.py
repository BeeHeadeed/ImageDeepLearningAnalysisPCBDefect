## Train Test Split

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit 


def split_dataset():
    """
    Split dataset into train, validation, and test sets.
    Ensures that images from the same file are not split across sets.
    """
    df = pd.read_csv('data/annotations.csv')

    # First split into train and test
    test_splitter = GroupShuffleSplit(test_size=0.2, n_splits=2, random_state = 7)
    test_split = test_splitter.split(df, groups=df['filename'])
    train_inds, test_inds = next(test_split)

    # Further split train into train and val
    val_splitter = GroupShuffleSplit(test_size=0.25, n_splits=2, random_state = 7)
    val_split = (val_splitter.split(df.iloc[train_inds], groups=df.iloc[train_inds]['filename']))
    train_inds,val_inds = next(val_split)

    # Create the final datasets
    train = df.iloc[train_inds]
    test = df.iloc[test_inds]
    val = df.iloc[val_inds]

    # Save to csv
    train.to_csv('data/train_annotations.csv', index=False)
    test.to_csv('data/test_annotations.csv', index=False)  
    val.to_csv('data/val_annotations.csv', index=False)

    print(f"Train set: {len(train)} annotations")
    print(f"Validation set: {len(val)} annotations")
    print(f"Test set: {len(test)} annotations")

if __name__ == "__main__":
    split_dataset()
    