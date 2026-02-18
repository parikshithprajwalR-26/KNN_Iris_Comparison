import numpy as np
from collections import Counter

class KNN:
    def __init__(self, k=3):
        self.k = k
    def fit(self, X, y):
        self.train_x = X
        self.train_y = y
    def predict(self, x): #val_x
        distances = [self._euclidean_distance(x,train_x) for train_x in self.train_x]
        k_indices=np.argsort(distances)[:self.k]
        k_nearest_labels=[self.train_y[i] for i in k_indices]
        most_common=Counter(k_nearest_labels).most_common(1)
        return most_common[0][0]
    def _euclidean_distance(self,x1,x2):
        return np.sqrt(np.sum((x1-x2)**2))