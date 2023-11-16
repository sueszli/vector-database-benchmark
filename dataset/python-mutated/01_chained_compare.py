def chained_compare_a(protocol):
    if False:
        while True:
            i = 10
    if not 0 <= protocol <= 7:
        raise ValueError('pickle protocol must be <= %d' % 7)

def chained_compare_b(a, obj):
    if False:
        return 10
    if a:
        if -2147483648 <= obj <= 2147483647:
            return 5

def chained_compare_c(a, d):
    if False:
        return 10
    for i in len(d):
        if a == d[i] != 2:
            return 5
chained_compare_a(3)
try:
    chained_compare_a(8)
except ValueError:
    pass
chained_compare_b(True, 0)
chained_compare_c(3, [3])