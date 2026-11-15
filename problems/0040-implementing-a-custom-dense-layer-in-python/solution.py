
import numpy as np
import torch
import torch.nn.functional as F
import copy
from typing import Tuple

# DO NOT CHANGE SEED
np.random.seed(42)

def _to_tensor(x: torch.Tensor | np.ndarray) -> torch.Tensor:
    return x if isinstance(x, torch.Tensor) else torch.as_tensor(x, dtype=torch.float32)


# DO NOT CHANGE LAYER CLASS
class Layer:
    def set_input_shape(self, shape: Tuple[int, ...]):
        self.input_shape = shape

    def layer_name(self) -> str:
        return self.__class__.__name__

    def parameters(self) -> int:
        return 0

    def forward_pass(self, X, training: bool = True):
        raise NotImplementedError

    def backward_pass(self, accum_grad):
        raise NotImplementedError

    def output_shape(self) -> Tuple[int, ...]:
        raise NotImplementedError


class Dense(Layer):
    def __init__(self, n_units: int, input_shape: Tuple[int, ...] | None = None):
        self.layer_input = None
        self.input_shape = input_shape
        self.n_units = n_units
        self.trainable = True
        self.W = None
        self.b = None
        self._opt = None

    def initialize(self, optimizer) -> None:
        n_inputs = self.input_shape[0]
        limit = 1 / np.sqrt(n_inputs)

        # Initialize weights using NumPy for determinism
        W_np = np.random.uniform(
            -limit, limit, (n_inputs, self.n_units)
        )

        self.W = torch.tensor(
            W_np, dtype=torch.float32, requires_grad=True
        )

        # Initialize biases
        self.b = torch.zeros(
            (1, self.n_units), requires_grad=True
        )

        # Independent optimizers for weights and biases
        self._opt = (
            copy.deepcopy(optimizer),
            copy.deepcopy(optimizer)
        )

    def number_of_parameters(self) -> int:
        return self.W.numel() + self.b.numel()

    def parameters(self) -> int:
        return self.number_of_parameters()

    def forward_pass(self, X, training: bool = True):
        X = _to_tensor(X)
        self.layer_input = X

        # F.linear expects weights in (out, in) shape
        self.layer_output = F.linear(
            X, self.W.T, self.b.squeeze(0)
        )

        return self.layer_output

    def backward_pass(self, accum_grad):
        accum_grad = _to_tensor(accum_grad)

        # Gradient with respect to input, before updates
        grad_input = accum_grad @ self.W.detach().T

        if self.trainable:
            # Compute weight and bias gradients with autograd
            grad_W, grad_b = torch.autograd.grad(
                outputs=self.layer_output,
                inputs=(self.W, self.b),
                grad_outputs=accum_grad
            )

            # Update parameters without recording gradients
            with torch.no_grad():
                new_W = self._opt[0].update(self.W, grad_W)
                new_b = self._opt[1].update(self.b, grad_b)

            # Restore leaf tensors for subsequent forward passes
            self.W = new_W.detach().requires_grad_(True)
            self.b = new_b.detach().requires_grad_(True)

        return grad_input

    def output_shape(self) -> Tuple[int, ...]:
        return (self.n_units,)
