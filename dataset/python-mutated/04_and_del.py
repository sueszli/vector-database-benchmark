def normpath(comps):
    if False:
        while True:
            i = 10
    i = 0
    while i < len(comps):
        if comps[i] == '.':
            del comps[i]
        elif comps[i] == '..' and i > 0 and (comps[i - 1] not in ('', '..')):
            del comps[i - 1:i + 1]
            i = i - 1
        elif comps[i] == '' and i > 0 and (comps[i - 1] != ''):
            del comps[i]
        else:
            i = i + 1
    return comps
assert normpath(['.']) == []
assert normpath(['a', 'b', '..']) == ['a']
assert normpath(['a', 'b', '', 'c']) == ['a', 'b', 'c']
assert normpath(['a', 'b', '.', '', 'c', '..']) == ['a', 'b']

def handle(format, html, text):
    if False:
        print('Hello World!')
    formatter = format and html or text
    return formatter
assert handle(False, False, True)
assert not handle(True, False, False)
assert handle(True, True, False)