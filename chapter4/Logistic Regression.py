import numpy as np
from sklearn import datasets
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
# t = np.linspace(-10, 10, 100)
# sig = 1 / (1 + np.exp(-t))
# plt.figure(figsize=(9, 6))
# plt.plot([-10, 10], [0, 0], "k-")
# plt.plot([-10, 10], [0.5, 0.5], "k:")
# plt.plot([-10, 10], [1, 1], "k:")
# plt.plot([0, 0], [-1.1, 1.1], "k-")
# plt.plot(t, sig, "b-", linewidth=2, label=r"$\sigma(t) = \frac{1}{1 + e^{-t}}$")
# plt.xlabel("t")
# plt.legend(loc="upper left", fontsize=20)
# plt.axis([-10, 10, -0.1, 1.1])
# plt.show()

iris=datasets.load_iris()
# print(list(iris.keys()))
# print(iris.DESCR)
X=iris["data"][:,3:]    # petal width
y=(iris["target"]==2).astype(np.int)    # 1 if Iris virginica, else 0

log_reg=LogisticRegression(solver="lbfgs",random_state=42)
log_reg.fit(X,y)

X_new=np.linspace(0,3,1000).reshape(-1,1)
y_proba=log_reg.predict_proba(X_new)

plt.plot(X_new,y_proba[:,1],"g-",linewidth=2,label="Iris virginica")
plt.plot(X_new,y_proba[:,0],"b--",linewidth=2,label="Not Iris virginica")
plt.show()
