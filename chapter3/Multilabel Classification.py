from math import ceil

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.base import clone
from sklearn.model_selection import cross_val_score
from sklearn.base import BaseEstimator
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score,recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier


mnist=fetch_openml('mnist_784',version=1,as_frame=False)
X,y=mnist["data"],mnist["target"]

some_digit = X[0]
some_digit_image=some_digit.reshape(28,28)

plt.imshow(some_digit_image,cmap="binary")
plt.axis("off")

y=y.astype(np.uint8)

X_train,x_test,y_train,y_test=X[:60000],X[60000:],y[:60000],y[60000:]

y_train_large=(y_train>=7)
y_train_odd=(y_train%2==1)
y_multilabel=np.c_[y_train_large,y_train_odd]

knn_clf=KNeighborsClassifier()
knn_clf.fit(X_train,y_multilabel)
# print(knn_clf.predict([some_digit]))
y_train_nn_pred=cross_val_predict(knn_clf,X_train,y_multilabel,cv=3)
print(f1_score(y_multilabel,y_train_nn_pred,average="macro"))
# print(f1_score(y_multilabel,y_train_nn_pred,average="weighted")) # 根據權重決定料的重要性
