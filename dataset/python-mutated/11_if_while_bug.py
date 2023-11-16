def _splitext(p, sep, altsep, extsep):
    if False:
        for i in range(10):
            print('nop')
    if p > sep:
        while sep < p:
            if p[sep:sep + 1] != extsep:
                return (p[:sep], p[sep:])
            altsep += 1
    return (p, p[:0])