try:
    import re
except ImportError:
    print('SKIP')
    raise SystemExit
try:
    re.sub
except AttributeError:
    print('SKIP')
    raise SystemExit

def multiply(m):
    if False:
        while True:
            i = 10
    return str(int(m.group(0)) * 2)
print(re.sub('\\d+', multiply, '10 20 30 40 50'))
print(re.sub('\\d+', lambda m: str(int(m.group(0)) // 2), '10 20 30 40 50'))

def A():
    if False:
        i = 10
        return i + 15
    return 'A'
print(re.sub('a', A(), 'aBCBABCDabcda.'))
print(re.sub('def\\s+([a-zA-Z_][a-zA-Z_0-9]*)\\s*\\(\\s*\\):', 'static PyObject*\npy_\\1(void){\n  return;\n}\n', '\n\ndef myfunc():\n\ndef myfunc1():\n\ndef myfunc2():'))
print(re.compile('(calzino) (blu|bianco|verde) e (scarpa) (blu|bianco|verde)').sub('\\g<1> colore \\2 con \\g<3> colore \\4? ...', 'calzino blu e scarpa verde'))
print(re.sub('(abc)', '\\g<1>\\g<1>', 'abc'))
print(re.sub('a', 'b', 'c'))
print(re.sub('a', 'b', '1a2a3a', 2))
try:
    re.sub('(a)', 'b\\2', 'a')
except:
    print('invalid group')
try:
    re.sub('(a)', 'b\\199999999999999999999999999999999999999', 'a')
except:
    print('invalid group')
print(re.sub('a', 'a', 'a'))
print(re.sub(b'.', b'a', b'a'))
print(re.sub(re.compile('a'), 'a', 'a'))
try:
    re.sub(123, 'a', 'a')
except TypeError:
    print('TypeError')
print(re.sub('b', '\\\\b', 'abc'))
print(re.sub('^ab', '*', 'abababcabab'))
print(re.sub('^ab|cab', '*', 'abababcabab'))