"""
Create Pytorch Native Evaluator
===============================

If you are using a native pytorch training loop to train your model, this example could help you getting start quickly.
In this example, you will learn how to create a pytorch native evaluator step by step.

Prepare ``training_func``
-------------------------

``training_func`` has six required parameters.
Maybe you don't need some parameters such as ``lr_scheduler``, but you still need to reflect the complete six parameters on the interface.

For some reason, ``dataloader`` is not exposed on ``training_func`` as part of the interface,
so it is necessary to directly create or reference an dataloader in ``training_func`` inner.

Here is an simple ``training_func``.
"""
from typing import Any, Callable
import torch
from examples.compression.models import prepare_dataloader

def training_func(model: torch.nn.Module, optimizer: torch.optim.Optimizer, training_step: Callable[[Any, torch.nn.Module], torch.Tensor], lr_scheduler: torch.optim.lr_scheduler._LRScheduler, max_steps: int, max_epochs: int):
    if False:
        print('Hello World!')
    (train_dataloader, test_dataloader) = prepare_dataloader()
    assert max_steps is not None or max_epochs is not None
    total_steps = max_steps if max_steps else max_epochs * len(train_dataloader)
    total_epochs = total_steps // len(train_dataloader) + (0 if total_steps % len(train_dataloader) == 0 else 1)
    current_step = 0
    for _ in range(total_epochs):
        for batch in train_dataloader:
            loss = training_step(batch, model)
            loss.backward()
            optimizer.step()
            current_step = current_step + 1
            if current_step >= total_steps:
                return
        lr_scheduler.step()
import nni
from examples.compression.models import build_resnet18
model = build_resnet18()
optimizer = nni.trace(torch.optim.Adam)(model.parameters(), lr=0.001)
lr_scheduler = nni.trace(torch.optim.lr_scheduler.LambdaLR)(optimizer, lr_lambda=lambda epoch: 1 / epoch)
import torch.nn.functional as F

def training_step(batch: Any, model: torch.nn.Module, *args, **kwargs):
    if False:
        i = 10
        return i + 15
    output = model(batch[0])
    loss = F.cross_entropy(output, batch[1])
    return loss
from nni.compression import TorchEvaluator
evaluator = TorchEvaluator(training_func, optimizer, training_step, lr_scheduler)