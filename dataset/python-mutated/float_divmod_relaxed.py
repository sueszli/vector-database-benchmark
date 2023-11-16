def test(x, y):
    if False:
        while True:
            i = 10
    (div, mod) = divmod(x, y)
    print(div == x // y, mod == x % y, abs(div * y + mod - x) < 1e-06)
test(1.23456, 0.7)
test(-1.23456, 0.7)
test(1.23456, -0.7)
test(-1.23456, -0.7)
a = 1.23456
b = 0.7
test(a, b)
test(a, -b)
test(-a, b)
test(-a, -b)
for i in range(25):
    x = (i - 12.5) / 6
    for j in range(25):
        y = (j - 12.5) / 6
        test(x, y)
try:
    divmod(1.0, 0)
except ZeroDivisionError:
    print('ZeroDivisionError')