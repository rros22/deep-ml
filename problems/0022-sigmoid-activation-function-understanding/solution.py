import math

def sigmoid(z: float) -> float:
    #Your code here
    # Numerically stable version
    if z > 0:
        return round(1/(1+math.exp(-z)), 4)
    else:
        return round(math.exp(z)/(1+math.exp(z)), 4)