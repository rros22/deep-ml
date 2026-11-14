import math

def sigmoid(z: float) -> float:
    #Your code here
    # Numerically stable version
    if z <= -1000:
        return 0
    else:
        result = 1 / (1 + math.exp(-z))
        return round(result, 4)