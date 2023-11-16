from __future__ import annotations
import numpy as np
from manimlib.utils.bezier import bezier
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from typing import Callable

def linear(t: float) -> float:
    if False:
        while True:
            i = 10
    return t

def smooth(t: float) -> float:
    if False:
        print('Hello World!')
    s = 1 - t
    return t ** 3 * (10 * s * s + 5 * s * t + t * t)

def rush_into(t: float) -> float:
    if False:
        i = 10
        return i + 15
    return 2 * smooth(0.5 * t)

def rush_from(t: float) -> float:
    if False:
        i = 10
        return i + 15
    return 2 * smooth(0.5 * (t + 1)) - 1

def slow_into(t: float) -> float:
    if False:
        print('Hello World!')
    return np.sqrt(1 - (1 - t) * (1 - t))

def double_smooth(t: float) -> float:
    if False:
        return 10
    if t < 0.5:
        return 0.5 * smooth(2 * t)
    else:
        return 0.5 * (1 + smooth(2 * t - 1))

def there_and_back(t: float) -> float:
    if False:
        for i in range(10):
            print('nop')
    new_t = 2 * t if t < 0.5 else 2 * (1 - t)
    return smooth(new_t)

def there_and_back_with_pause(t: float, pause_ratio: float=1.0 / 3) -> float:
    if False:
        while True:
            i = 10
    a = 1.0 / pause_ratio
    if t < 0.5 - pause_ratio / 2:
        return smooth(a * t)
    elif t < 0.5 + pause_ratio / 2:
        return 1
    else:
        return smooth(a - a * t)

def running_start(t: float, pull_factor: float=-0.5) -> float:
    if False:
        print('Hello World!')
    return bezier([0, 0, pull_factor, pull_factor, 1, 1, 1])(t)

def overshoot(t: float, pull_factor: float=1.5) -> float:
    if False:
        for i in range(10):
            print('nop')
    return bezier([0, 0, pull_factor, pull_factor, 1, 1])(t)

def not_quite_there(func: Callable[[float], float]=smooth, proportion: float=0.7) -> Callable[[float], float]:
    if False:
        i = 10
        return i + 15

    def result(t):
        if False:
            print('Hello World!')
        return proportion * func(t)
    return result

def wiggle(t: float, wiggles: float=2) -> float:
    if False:
        while True:
            i = 10
    return there_and_back(t) * np.sin(wiggles * np.pi * t)

def squish_rate_func(func: Callable[[float], float], a: float=0.4, b: float=0.6) -> Callable[[float], float]:
    if False:
        while True:
            i = 10

    def result(t):
        if False:
            for i in range(10):
                print('nop')
        if a == b:
            return a
        elif t < a:
            return func(0)
        elif t > b:
            return func(1)
        else:
            return func((t - a) / (b - a))
    return result

def lingering(t: float) -> float:
    if False:
        while True:
            i = 10
    return squish_rate_func(lambda t: t, 0, 0.8)(t)

def exponential_decay(t: float, half_life: float=0.1) -> float:
    if False:
        for i in range(10):
            print('nop')
    return 1 - np.exp(-t / half_life)