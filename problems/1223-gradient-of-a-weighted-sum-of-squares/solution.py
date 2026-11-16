import torch

def grad_wss(w_list, x_list):
    """Build w (requires_grad) and x from lists, compute
    loss = 0.5 * sum((w * x)**2), backward, return w.grad
    as a list of floats rounded to 4 decimals.
    """
    # TODO
    w_list = torch.tensor(w_list, dtype=torch.float, requires_grad=True)
    x_list = torch.tensor(x_list, dtype=torch.float, requires_grad=False)

    loss = 0.5 * ((w_list * x_list)**2).sum()
    loss.backward()

    return [round(item, 4) for item in w_list.grad.tolist()]