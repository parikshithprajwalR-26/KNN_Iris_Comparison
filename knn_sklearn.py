from sklearn.neighbors import KNeighborsClassifier

def sklearn_knn(k=3):
    return KNeighborsClassifier(n_neighbors=k)
