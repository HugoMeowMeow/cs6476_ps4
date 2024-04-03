"""Unit tests for optimizer.py"""
import torch

from proj4_code.classification.optimizer import compute_quadratic_loss, gradient_descent_step


def test_compute_quadratic_loss():
    w = torch.tensor([7.0])
    expected = torch.tensor([4.0])
    computed = compute_quadratic_loss(w)
    assert torch.allclose(computed, expected)


def test_gradient_descent_step():
    w = torch.tensor([2.0], requires_grad=True)
    print(w)
    L = 7 * torch.pow(w, 3)
    lr = 1e-2
    expected_w = torch.tensor([1.16])

    gradient_descent_step(w, L, lr)
    assert torch.allclose(w, expected_w)

