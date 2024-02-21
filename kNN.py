import pyarrow as pa
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn as sk
from sklearn.base import BaseEstimator,ClassifierMixin
from scipy.spatial import distance

class kNN(BaseEstimator, ClassifierMixin):
    def __init__(self, n_neighbors:int = 3):
        self.n_neighbors = n_neighbors
        self.X = None
        self.y = None

    def fit(self, X, y):
        self.X = np.copy(X)
        self.y = np.copy(y)
        return self

    def predict(self, X):
        distances = distance.cdist(X, self.X, 'euclidean')
        k_nearest = np.argpartition(distances, self.n_neighbors)[:, :self.n_neighbors]
        predictions = np.sign(np.mean(self.y[k_nearest], axis=1))
        return predictions
