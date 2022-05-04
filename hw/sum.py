import numpy as np
def sum(*a):
    total=np.sum(a)
    return total

n=int(input())

print(n*sum(3, 4, 8))

