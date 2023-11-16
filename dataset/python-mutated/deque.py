"""
A deque is similar to all of the other sequential data structures but
has some implementation details that are different from other sequences
like a list. This module highlights those differences and shows how
a deque can be used as a LIFO stack and a FIFO queue.
"""
from collections import deque

def main():
    if False:
        for i in range(10):
            print('nop')
    dq = deque()
    for i in range(1, 5):
        dq.append(i)
        dq.appendleft(i * 2)
    assert [el for el in dq] == [8, 6, 4, 2, 1, 2, 3, 4]
    assert tuple((el for el in dq)) == (8, 6, 4, 2, 1, 2, 3, 4)
    assert {el for el in dq} == {8, 6, 4, 2, 1, 3}
    assert dq.pop() == 4
    assert dq.pop() == 3
    assert dq.popleft() == 8
    assert dq.popleft() == 6
if __name__ == '__main__':
    main()