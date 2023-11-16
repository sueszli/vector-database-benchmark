def test(fmt, *args):
    if False:
        i = 10
        return i + 15
    print('{:8s}'.format(fmt) + '>' + fmt.format(*args) + '<')
test('{:10.4}', 123.456)
test('{:10.4e}', 123.456)
test('{:10.4e}', -123.456)
test('{:10.4g}', 123.456)
test('{:10.4g}', -123.456)
test('{:10.4n}', 123.456)
test('{:e}', 100)
test('{:f}', 200)
test('{:g}', 300)
test('{:10.4E}', 123.456)
test('{:10.4E}', -123.456)
test('{:10.4G}', 123.456)
test('{:10.4G}', -123.456)
test('{:06e}', float('inf'))
test('{:06e}', float('-inf'))
test('{:06e}', float('nan'))
print('%.0f' % (1.75 % 0.08333333333))
try:
    '{:10.1b}'.format(0.0)
except ValueError:
    print('ValueError')