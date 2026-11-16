import numpy as np

def adam_optimizer(f, grad, x0, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8, num_iterations=10):
    # Note: `f` is accepted only for interface parity with the PyTorch/Tinygrad
    # variants, which derive gradients from it via autograd. This version uses
    # `grad` only — the objective value `f` is never evaluated.
    # Your code here
    
    # 1. Initialisation
    t = 0
    m = 0
    v = 0
    theta = np.array(x0, dtype=float, copy=True)

    # 2. Training loop
    for it in range(num_iterations):
        t += 1
        g = grad(theta)
        m = beta1 * m + (1-beta1)*g
        v = beta2 * v + (1-beta2)*(g**2)
        # Bias correction
        mhat = m /(1 - beta1**t)
        vhat = v /(1 - beta2**t)

        theta -= learning_rate * mhat/(np.sqrt(vhat) + epsilon)
    
    return theta







