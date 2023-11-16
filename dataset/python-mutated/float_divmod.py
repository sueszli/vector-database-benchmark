def test(x, y):
    if False:
        for i in range(10):
            print('nop')
    (div, mod) = divmod(x, y)
    print('%.8f %.8f %.8f %.8f' % (x // y, x % y, div, mod))
    print(div == x // y, mod == x % y, abs(div * y + mod - x) < 1e-15)
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