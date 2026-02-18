import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split   
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedKFold

from knn_scratch import KNN
from knn_sklearn import sklearn_knn

#Load Dataset
data=load_iris()
X=data.data
y=data.target

#split dataset into training and testing
train_x,val_x,train_y,val_y=train_test_split(X,y,train_size=0.8,random_state=7)

#-----Scratch KNN model-----
scratch_model=KNN(k=5)
scratch_model.fit(train_x,train_y)

scratch_model_predictions=[scratch_model.predict(x) for x in val_x]
scratch_model_accuracy=accuracy_score(val_y,scratch_model_predictions)

#-----Sklearn KNN model-----
sk_model=sklearn_knn(k=5)
sk_model.fit(train_x,train_y)

sk_model_predictions=sk_model.predict(val_x)
sk_model_accuracy=accuracy_score(val_y,sk_model_predictions)

conf_matrix_1=confusion_matrix(val_y,scratch_model_predictions)
conf_matrix_2=confusion_matrix(val_y,sk_model_predictions)

print("\n===== Scratch KNN =====")
print("Accuracy:", scratch_model_accuracy)
print(conf_matrix_1)

print("\n===== Sklearn KNN =====")
print("Accuracy:", sk_model_accuracy)
print(conf_matrix_2)

