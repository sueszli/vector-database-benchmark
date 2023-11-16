def zipped():
    if False:
        for i in range(10):
            print('nop')
    return zip([1, 2, 3], 'ABC')
[print(x, y) for (x, y) in zipped()]
(print(x, y) for (x, y) in zipped())
{print(x, y) for (x, y) in zipped()}
from itertools import starmap as sm
[print(x, y) for (x, y) in zipped()]
(print(x, y) for (x, y) in zipped())
{print(x, y) for (x, y) in zipped()}
[foo(*t) for t in [(85, 60), (100, 80)]]
(foo(*t) for t in [(85, 60), (100, 80)])
{foo(*t) for t in [(85, 60), (100, 80)]}
[print(x, int) for (x, _) in zipped()]
[print(x, *y) for (x, y) in zipped()]
[print(x, y, 1) for (x, y) in zipped()]
[print(y, x) for (x, y) in zipped()]
[print(x + 1, y) for (x, y) in zipped()]
[print(x) for x in range(100)]
[print() for (x, y) in zipped()]
[print(x, end=y) for (x, y) in zipped()]
[print(*x, y) for (x, y) in zipped()]
[print(x, *y) for (x, y) in zipped()]
[print(*x, *y) for (x, y) in zipped()]