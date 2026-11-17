
import torch
import torch.nn.functional as F

def train_softmaxreg(X: torch.Tensor, y: torch.Tensor,
                     learning_rate: float, iterations: int) -> tuple[list[float], ...]:
    """
    Train softmax regression using gradient descent and
    sum-based cross-entropy loss.
    """

    # 1. Prepare data and determine the number of classes
    X = X.float()
    y = y.long()
    c = int(y.max().item()) + 1

    # 2. Initialize weights: one column per class,
    # one row per feature (plus the bias)
    theta = torch.zeros(
        (X.shape[1] + 1, c),
        dtype=X.dtype,
        device=X.device,
        requires_grad=True
    )

    # 3. Add a column of ones for the intercept term
    X = torch.cat(
        (torch.ones((X.shape[0], 1), dtype=X.dtype, device=X.device), X),
        dim=1
    )

    # 4. Define sum-based cross-entropy loss
    def loss():
        # Convert integer labels to one-hot vectors
        Y = F.one_hot(y, num_classes=c)

        # Compute class scores (logits)
        logits = X @ theta

        # Compute log probabilities using stable log-softmax
        log_probs = F.log_softmax(logits, dim=1)

        # Select true-class log probabilities using the one-hot mask
        return -torch.sum(Y * log_probs)

    # 5. Train using gradient descent
    losses = []

    for it in range(iterations):
        # Forward pass: compute loss
        L = loss()

        # Backward pass: compute dL/dtheta
        L.backward()

        # Record loss before updating parameters
        losses.append(round(L.item(), 4))

        # Update parameters without tracking gradients
        with torch.no_grad():
            theta -= learning_rate * theta.grad

        # Clear gradients to avoid accumulation
        theta.grad.zero_()

    # 6. Transpose: one row per class, and round coefficients
    coefficients = [
        [round(v, 4) for v in row]
        for row in theta.detach().T.tolist()
    ]

    return coefficients, losses
