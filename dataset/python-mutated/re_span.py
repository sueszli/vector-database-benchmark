try:
    import re
except ImportError:
    print('SKIP')
    raise SystemExit
try:
    m = re.match('.', 'a')
    m.span
except AttributeError:
    print('SKIP')
    raise SystemExit

def print_spans(match):
    if False:
        return 10
    print('----')
    try:
        i = 0
        while True:
            print(match.span(i), match.start(i), match.end(i))
            i += 1
    except IndexError:
        pass
m = re.match('(([0-9]*)([a-z]*)[0-9]*)', '1234hello567')
print_spans(m)
m = re.match('([0-9]*)(([a-z]*)([0-9]*))', '1234hello567')
print_spans(m)
print_spans(re.match('(a)?b(c)', 'abc'))
print_spans(re.match('(a)?b(c)', 'bc'))