import math


def area(v):
    area=v**2*math.pi
    return area

def round(v):
    round=2*math.pi*v
    return round

v=float(input())
print(4*area(v)+5*round(v))
