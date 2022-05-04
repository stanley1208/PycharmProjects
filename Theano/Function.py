import numpy as np
import theano
import theano.tensor as T

# activation unction example
x=T.dmatrix('x')
s=1/(1+T.exp(-x)) # logistic or soft step
logistic=theano.function([x],s)
print(logistic([[0,1],[-2,-3]]))


# multiply outputs or a function
a,b=T.dmatrices('a','b')
diff=a-b
abs_diff=abs(diff)
diff_squared=diff**2
f=theano.function([a,b],[diff,abs_diff,diff_squared])
print(f(
    np.ones((2,2)),
    np.arange(4).reshape((2,2))
))


# name or a function
x,y,w=T.dscalars('x','y','w')
z=(x+y)*w
f=theano.function([x,
                   theano.In(y,value=1),
                   theano.In(w,value=2,name='weights')],
                  z)
print(f(23,2,weights=4))