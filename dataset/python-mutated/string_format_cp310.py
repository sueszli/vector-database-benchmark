def test(fmt, *args):
    if False:
        for i in range(10):
            print('nop')
    print('{:8s}'.format(fmt) + '>' + fmt.format(*args) + '<')
test('{:0s}', 'ab')
test('{:06s}', 'ab')
test('{:<06s}', 'ab')
test('{:>06s}', 'ab')