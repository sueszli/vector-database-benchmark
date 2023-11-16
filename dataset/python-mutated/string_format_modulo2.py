def test(num, num_str):
    if False:
        i = 10
        return i + 15
    if num == float('inf') or (num == 0.0 and num_str != '0.0'):
        return
    for kind in ('e', 'f', 'g'):
        for prec in range(23, 36, 2):
            fmt = '%.' + '%d' % prec + kind
            s = fmt % num
            check = abs(float(s) - num)
            if num > 1:
                check /= num
            if check > 1e-06:
                print('FAIL', num_str, fmt, s, len(s), check)
test(0.0, '0.0')
for e in range(-8, 8):
    num = pow(10, e)
    test(num, '1e%d' % e)