"""
*What is this pattern about?
Define a family of algorithms, encapsulate each one, and make them interchangeable.
Strategy lets the algorithm vary independently from clients that use it.

*TL;DR
Enables selecting an algorithm at runtime.
"""
from __future__ import annotations
from typing import Callable

class DiscountStrategyValidator:

    @staticmethod
    def validate(obj: Order, value: Callable) -> bool:
        if False:
            i = 10
            return i + 15
        try:
            if obj.price - value(obj) < 0:
                raise ValueError(f'Discount cannot be applied due to negative price resulting. {value.__name__}')
        except ValueError as ex:
            print(str(ex))
            return False
        else:
            return True

    def __set_name__(self, owner, name: str) -> None:
        if False:
            i = 10
            return i + 15
        self.private_name = f'_{name}'

    def __set__(self, obj: Order, value: Callable=None) -> None:
        if False:
            while True:
                i = 10
        if value and self.validate(obj, value):
            setattr(obj, self.private_name, value)
        else:
            setattr(obj, self.private_name, None)

    def __get__(self, obj: object, objtype: type=None):
        if False:
            i = 10
            return i + 15
        return getattr(obj, self.private_name)

class Order:
    discount_strategy = DiscountStrategyValidator()

    def __init__(self, price: float, discount_strategy: Callable=None) -> None:
        if False:
            return 10
        self.price: float = price
        self.discount_strategy = discount_strategy

    def apply_discount(self) -> float:
        if False:
            while True:
                i = 10
        if self.discount_strategy:
            discount = self.discount_strategy(self)
        else:
            discount = 0
        return self.price - discount

    def __repr__(self) -> str:
        if False:
            return 10
        return f"<Order price: {self.price} with discount strategy: {getattr(self.discount_strategy, '__name__', None)}>"

def ten_percent_discount(order: Order) -> float:
    if False:
        for i in range(10):
            print('nop')
    return order.price * 0.1

def on_sale_discount(order: Order) -> float:
    if False:
        i = 10
        return i + 15
    return order.price * 0.25 + 20

def main():
    if False:
        print('Hello World!')
    '\n    >>> order = Order(100, discount_strategy=ten_percent_discount)\n    >>> print(order)\n    <Order price: 100 with discount strategy: ten_percent_discount>\n    >>> print(order.apply_discount())\n    90.0\n    >>> order = Order(100, discount_strategy=on_sale_discount)\n    >>> print(order)\n    <Order price: 100 with discount strategy: on_sale_discount>\n    >>> print(order.apply_discount())\n    55.0\n    >>> order = Order(10, discount_strategy=on_sale_discount)\n    Discount cannot be applied due to negative price resulting. on_sale_discount\n    >>> print(order)\n    <Order price: 10 with discount strategy: None>\n    '
if __name__ == '__main__':
    import doctest
    doctest.testmod()