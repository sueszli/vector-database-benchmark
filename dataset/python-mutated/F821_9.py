"""Test: match statements."""
from dataclasses import dataclass

@dataclass
class Car:
    make: str
    model: str

def f():
    if False:
        i = 10
        return i + 15
    match Car('Toyota', 'Corolla'):
        case Car('Toyota', model):
            print(model)
        case Car(make, 'Corolla'):
            print(make)

def f(provided: int) -> int:
    if False:
        return 10
    match provided:
        case True:
            return captured

def f(provided: int) -> int:
    if False:
        while True:
            i = 10
    match provided:
        case captured:
            return captured

def f(provided: int) -> int:
    if False:
        while True:
            i = 10
    match provided:
        case [captured, *_]:
            return captured

def f(provided: int) -> int:
    if False:
        while True:
            i = 10
    match provided:
        case [*captured]:
            return captured

def f(provided: int) -> int:
    if False:
        for i in range(10):
            print('nop')
    match provided:
        case {**captured}:
            return captured