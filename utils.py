# Required Header files imported in the program.
import os
import cv2
import numpy as np
from keras.models import load_model
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
from pandas.io.parsers import read_csv
#load_data function to load the data from CSV file.
def load_data(test=False):
    data_directory = 'data'
    FTRAIN = os.path.join(data_directory, 'training.csv')
    FTEST = os.path.join(data_directory, 'test.csv')
    fname = FTEST if test else FTRAIN
    df = read_csv(os.path.expanduser(fname))
    df['Image'] = df['Image'].apply(lambda im: np.fromstring(im, sep=' '))
    # drop all rows that having missing values.
    df = df.dropna()  
    # scale pixel values to [0, 1] (Normalizing)
    X = np.vstack(df['Image'].values) / 255.  
    # combine normalization and reshaping
    X = X.astype(np.float32).reshape(-1, 96, 96, 1)  
# only FTRAIN has target columns
    if not test:  
        y = (df[df.columns[:-1]].values - 48) / 48  # scale target coordinates to [-1, 1] (Normalizing)
        X, y = shuffle(X, y, random_state=42)  # shuffle train data
        y = y.astype(np.float32)
    else:
        y = None

    return X, y