try:
    import re
except ImportError:
    print('SKIP')
    raise SystemExit

def print_groups(match):
    if False:
        for i in range(10):
            print('nop')
    print('----')
    try:
        i = 0
        while True:
            print(match.group(i))
            i += 1
    except IndexError:
        pass
m = re.match('(([0-9]*)([a-z]*)[0-9]*)', '1234hello567')
print_groups(m)
m = re.match('([0-9]*)(([a-z]*)([0-9]*))', '1234hello567')
print_groups(m)
print_groups(re.match('(a)?b(c)', 'abc'))
print_groups(re.match('(a)?b(c)', 'bc'))