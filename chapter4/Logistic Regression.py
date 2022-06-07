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
log_reg.fit(X,y.ravel())

X_new=np.linspace(0,3,1000).reshape(-1,1)
y_proba=log_reg.predict_proba(X_new)

# plt.plot(X_new,y_proba[:,1],"g-",linewidth=2,label="Iris virginica")
# plt.plot(X_new,y_proba[:,0],"b--",linewidth=2,label="Not Iris virginica")
# plt.show()

decision_boundary=X_new[y_proba[:,1]>=0.5][0]

plt.figure(figsize=(8,5))
plt.plot(X[y==0],y[y==0],"bs")
plt.plot(X[y==1],y[y==1],"g^")
plt.plot([decision_boundary,decision_boundary],[-1,2],"k:",linewidth=2)
plt.plot(X_new,y_proba[:,1],"g-",linewidth=2,label="Iris virginica")
plt.plot(X_new,y_proba[:,0],"b--",linewidth=2,label="Not Iris virginica")
plt.text(decision_boundary+0.02,0.15,"Decision boundary",fontsize=14,color="k",ha="center")
plt.arrow(decision_boundary,0.08,-0.3,0,head_width=0.05,head_length=0.1,fc='b',ec='b')
plt.arrow(decision_boundary,0.92,0.3,0,head_width=0.05,head_length=0.1,fc='g',ec='g')
plt.xlabel("Petal width (cm)",fontsize=14)
plt.ylabel("Probability",fontsize=14)
plt.legend(loc="best",fontsize=14)
plt.axis([0,3,-0.02,1.02])
# plt.show()

print(decision_boundary)
print(log_reg.predict([[1.7],[1.5]]))