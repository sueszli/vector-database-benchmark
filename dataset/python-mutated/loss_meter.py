"""
This module implements the orignal loss tracker for Hidden Trigger Backdoor attack on Neural Networkxss.
"""

class LossMeter:
    """
    Computes and stores the average and current loss value
    """

    def __init__(self):
        if False:
            print('Hello World!')
        '\n        Create loss tracker\n        '
        self.reset()

    def reset(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Reset loss tracker\n        '
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val: float, n: int=1):
        if False:
            i = 10
            return i + 15
        '\n        Update loss tracker\n        :param val: Loss value to add to tracker\n        :param n: Number of elements contributing to val\n        '
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count