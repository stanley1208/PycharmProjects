import numpy as np
np.random.seed(1337)  # for reproducibility
from keras.models import Sequential
from keras.layers import Dense
from keras.models import load_model

X=np.linspace(-1,1,200)
np.random.shuffle(X)
Y=0.5*X+2+np.random.normal(0,0.05,(200,))
X_train,Y_train=X[:160],Y[:160]
X_test,Y_test=X[160:],Y[160:]
model=Sequential()
model.add(Dense(1,input_dim=1))
model.compile(loss='mse',optimizer='sgd')
for step in range(301):
    cost=model.train_on_batch(X_train,Y_train)

print('test before save: ',model.predict(X_test[0:2]))
model.save('my_model.h5')
del model

model=load_model('my_model.h5')
print('test after save: ',model.predict(X_test[0:2]))