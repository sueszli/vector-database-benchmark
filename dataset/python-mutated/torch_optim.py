import torch

def foo(opt: torch.optim.Optimizer) -> None:
    if False:
        while True:
            i = 10
    opt.zero_grad()
opt_adagrad = torch.optim.Adagrad([torch.tensor(0.0)])
reveal_type(opt_adagrad)
foo(opt_adagrad)
opt_adam = torch.optim.Adam([torch.tensor(0.0)], lr=0.01, eps=1e-06)
reveal_type(opt_adam)
foo(opt_adam)