from .base import Callback
import torch
from bigdl.dllib.utils.log4Error import invalidInputError

def make_only_mainCallback(callbacks: list):
    if False:
        while True:
            i = 10
    _num_MCB = 0
    for i in range(len(callbacks)):
        if isinstance(callbacks[i], MainCallback):
            _num_MCB += 1
            (callbacks[0], callbacks[i]) = (callbacks[i], callbacks[0])
    if _num_MCB == 0:
        callbacks.insert(0, MainCallback())
    elif _num_MCB > 1:
        invalidInputError(False, f'Expect at most one MainCallbackinstance to be passed to torch estimator, got {{_num_MCB}} MainCallback instances.')

class MainCallback(Callback):
    """
    MainCallback is a one-of-a-kind callback that contains hook functions:
        - `on_iter_forward`
        - `on_iter_backward`
        - `on_lr_adjust`

    These methods are somewhat special, because only one special MainCallback
    should be allowed to implement these methods among all callbacks, otherwise
    there will propagate forward and backward twice.
    """

    def on_iter_forward(self, runner):
        if False:
            return 10
        '\n        If `on_train_forward` and `on_val_forward` are not overridden,\n        this will be called during forward when training and validating.\n        Any behavior inconsistent with the default forward behavior should be overridden here.\n        '
        (*features, target) = runner.batch
        runner.output = runner.model(*features)
        targetL = [target] if not isinstance(target, (list, tuple)) else target
        outputL = [runner.output] if not isinstance(runner.output, (list, tuple)) else runner.output
        runner.loss = runner.criterion(*outputL, *targetL)
        runner.target = target

    def on_iter_backward(self, runner):
        if False:
            i = 10
            return i + 15
        '\n        this will be called during backward when training.\n        Any behavior inconsistent with the default backward behavior should be overridden here.\n        '
        runner.optimizer.zero_grad()
        runner.loss.backward()
        runner.optimizer.step()

    def on_lr_adjust(self, runner):
        if False:
            for i in range(10):
                print('nop')
        '\n        this will be called during adjusting scheduler when each epoch ends.\n        By default, this will step scheduler if there is scheduler in runner.\n        Any behavior inconsistent with the default behavior should be overridden here.\n        '
        if runner.scheduler:
            runner.scheduler.step()

    def on_train_forward(self, runner):
        if False:
            for i in range(10):
                print('nop')
        '\n        Called during training.\n        Any behavior inconsistent with the default training behavior should be overridden here.\n        '
        self.on_iter_forward(runner)

    def on_val_forward(self, runner):
        if False:
            while True:
                i = 10
        '\n        Called during validate.\n        Any behavior inconsistent with the default training behavior should be overridden here.\n        '
        self.on_iter_forward(runner)

    def on_pred_forward(self, runner):
        if False:
            print('Hello World!')
        '\n        Called during prediction.\n        Any behavior inconsistent with the default prediction behavior should be overridden here.\n        '
        output = runner.model(*runner.batch)
        if len(output.size()) > 1:
            for i in reversed(range(1, len(output.size()))):
                output = torch.squeeze(output, i)
        runner.output = output.detach().numpy()