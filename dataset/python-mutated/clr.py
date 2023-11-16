from math import floor

def cycle(iteration, stepsize):
    if False:
        while True:
            i = 10
    return floor(1 + iteration / (2 * stepsize))

def abs_pos(cycle_num, iteration, stepsize):
    if False:
        while True:
            i = 10
    return abs(iteration / stepsize - 2 * cycle_num + 1)

def rel_pos(iteration, stepsize):
    if False:
        print('Hello World!')
    return max(0, 1 - abs_pos(cycle(iteration, stepsize), iteration, stepsize))

def cyclic_learning_rate(min_lr, max_lr, stepsize):
    if False:
        return 10
    return lambda iteration: min_lr + (max_lr - min_lr) * rel_pos(iteration, stepsize)