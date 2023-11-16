import torch
import math

def torchtest():
    if False:
        return 10
    dtype = torch.float
    device = torch.device('cuda:0')
    q = torch.linspace(-math.pi, math.pi, 5000000, device=device, dtype=dtype)
    x = torch.linspace(-math.pi, math.pi, 5000000, device=device, dtype=dtype)
    y = torch.sin(x)
    a = torch.randn((), device=device, dtype=dtype, requires_grad=True)
    b = torch.randn((), device=device, dtype=dtype, requires_grad=True)
    c = torch.randn((), device=device, dtype=dtype, requires_grad=True)
    d = torch.randn((), device=device, dtype=dtype, requires_grad=True)
    learning_rate = 1e-06
    for t in range(2000):
        y_pred = a + b * x + c * x ** 2 + d * x ** 3
        loss = (y_pred - y).sum()
        if t % 100 == 99:
            print(t, loss.item())
        loss.backward()
        with torch.no_grad():
            a -= learning_rate * a.grad
            b -= learning_rate * b.grad
            c -= learning_rate * c.grad
            d -= learning_rate * d.grad
            a.grad = None
            b.grad = None
            c.grad = None
            d.grad = None
    print(f'Result: y = {a.item()} + {b.item()} x + {c.item()} x^2 + {d.item()} x^3')
torchtest()