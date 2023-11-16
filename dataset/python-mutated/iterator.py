"""
http://ginstrom.com/scribbles/2007/10/08/design-patterns-python-style/
Implementation of the iterator pattern with a generator

*TL;DR
Traverses a container and accesses the container's elements.
"""

def count_to(count: int):
    if False:
        return 10
    'Counts by word numbers, up to a maximum of five'
    numbers = ['one', 'two', 'three', 'four', 'five']
    yield from numbers[:count]

def count_to_two() -> None:
    if False:
        for i in range(10):
            print('nop')
    return count_to(2)

def count_to_five() -> None:
    if False:
        while True:
            i = 10
    return count_to(5)

def main():
    if False:
        while True:
            i = 10
    '\n    # Counting to two...\n    >>> for number in count_to_two():\n    ...     print(number)\n    one\n    two\n\n    # Counting to five...\n    >>> for number in count_to_five():\n    ...     print(number)\n    one\n    two\n    three\n    four\n    five\n    '
if __name__ == '__main__':
    import doctest
    doctest.testmod()