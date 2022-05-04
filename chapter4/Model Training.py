import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import SGDRegressor

X=2*np.random.rand(100,1)
y=4+3*X+np.random.randn(100,1)

# plt.plot(X,y,"b.")
# plt.xlabel("$x_1$",fontsize=18)
# plt.ylabel("$y$",rotation=0,fontsize=18)
# plt.axis([0,2,0,15])
# plt.show()

X_b=np.c_[np.ones((100,1)),X] # add x0 = 1 to each instance
theta_best=np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y) # Nornal Equation

# print(theta_best)

# use θ to predict
X_new=np.array([[0],[2]])
X_new_b=np.c_[np.ones((2,1)),X_new] # add x0 = 1 to each instance
y_predict=X_new_b.dot(theta_best)
# print(y_predict)

# plt.plot(X_new,y_predict,"r-",linewidth=2,label="prediction")
# plt.plot(X,y,"b.")
# plt.xlabel("$x_1$",fontsize=18)
# plt.ylabel("$y$",rotation=0,fontsize=18)
# plt.legend(loc="upper left",fontsize=14)
# plt.axis([0,2,0,15])
# plt.show()

# Linear Regression using sklearn
lin_reg=LinearRegression()
lin_reg.fit(X,y)
# print(lin_reg.intercept_)
# print(lin_reg.coef_)

theta_best_svd,residuals,rank,s=np.linalg.lstsq(X_b,y,rcond=1e-6) # least squares
# print(theta_best_svd)
# print(np.linalg.pinv(X_b).dot(y))


# Batch Gradient Descent
eta=0.1 # learning rate
n_iterations=1000
m=100

theta=np.random.randn(2,1) # random initialization

for iteration in range(n_iterations):
    gradients=2/m*X_b.T.dot(X_b.dot(theta)-y)
    theta=theta-eta*gradients

# print(theta)

theta_path_bgd = []

def plot_gradient_descent(theta, eta, theta_path=None):
    m = len(X_b)
    plt.plot(X, y, "b.")
    n_iterations = 1000
    for iteration in range(n_iterations):
        if iteration < 10:
            y_predict = X_new_b.dot(theta)
            style = "b-" if iteration > 0 else "r--"
            plt.plot(X_new, y_predict, style)
        gradients = 2/m * X_b.T.dot(X_b.dot(theta) - y)
        theta = theta - eta * gradients
        if theta_path is not None:
            theta_path.append(theta)
    plt.xlabel("$x_1$", fontsize=18)
    plt.axis([0, 2, 0, 15])
    plt.title(r"$\eta = {}$".format(eta), fontsize=16)


np.random.seed(42)
theta = np.random.randn(2,1)  # random initialization

# plt.figure(figsize=(10,4))
# plt.subplot(131); plot_gradient_descent(theta, eta=0.02)
# plt.ylabel("$y$", rotation=0, fontsize=18)
# plt.subplot(132); plot_gradient_descent(theta, eta=0.1, theta_path=theta_path_bgd)
# plt.subplot(133); plot_gradient_descent(theta, eta=0.5)
# plt.show()

# Stochastic Gradient Descent
theta_path_sgd = []
m = len(X_b)
np.random.seed(42)

n_epoch=50
t0,t1=5,50 # learning schedule hyperparameters

def learning_schedule(t):
    return t0/(t+t1)

theta=np.random.randn(2,1)  # random initialization

for epoch in range(n_epoch):
    for i in range(m):
        if epoch==0 and i<20:
            y_predict = X_new_b.dot(theta)
            style = "b-" if i > 0 else "r--"
            plt.plot(X_new, y_predict, style)
        random_index=np.random.randint(m)
        xi=X_b[random_index:random_index+1]
        yi=y[random_index:random_index+1]
        gradients = 2 * xi.T.dot(xi.dot(theta) - yi)
        eta=learning_schedule(epoch*m+i)
        theta=theta-eta*gradients
        theta_path_sgd.append(theta)

# plt.plot(X, y, "b.")
# plt.xlabel("$x_1$", fontsize=18)
# plt.ylabel("$y$", rotation=0, fontsize=18)
# plt.axis([0, 2, 0, 15])
# plt.show()

# print(theta)

sgd_reg=SGDRegressor(max_iter=1000,tol=1e-3,penalty=None,eta0=0.1,random_state=42)
print(sgd_reg.fit(X,y.ravel()))
print(sgd_reg.intercept_,sgd_reg.coef_)



