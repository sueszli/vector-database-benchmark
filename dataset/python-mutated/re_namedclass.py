try:
    import re
except ImportError:
    print('SKIP')
    raise SystemExit

def print_groups(match):
    if False:
        while True:
            i = 10
    print('----')
    try:
        i = 0
        while True:
            print(match.group(i))
            i += 1
    except IndexError:
        pass
m = re.match('\\w+', '1234hello567 abc')
print_groups(m)
m = re.match('(\\w+)\\s+(\\w+)', 'ABC \t1234hello567 abc')
print_groups(m)
m = re.match('(\\S+)\\s+(\\D+)', 'ABC \thello abc567 abc')
print_groups(m)
m = re.match('(([0-9]*)([a-z]*)\\d*)', '1234hello567')
print_groups(m)
print_groups(re.match('([^\\s]+)\\s*([^\\s]+)', '1 23'))
print_groups(re.match('([\\s\\d]+)([\\W]+)', '1  2-+='))
print_groups(re.match('([\\W]+)([^\\W]+)([^\\S]+)([^\\D]+)', ' a_1 23'))