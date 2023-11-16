import re

def _process_docstring(app, what, name, obj, options, lines):
    if False:
        i = 10
        return i + 15
    liter_re = re.compile('\\s*```\\s*$')
    liter_flag = False
    offset = 0
    for j in range(len(lines)):
        i = j + offset
        line = lines[i]
        if not liter_flag and liter_re.match(line):
            liter_flag = True
            lines.insert(i + 1, '')
            offset += 1
            lines[i] = '::'
        elif liter_flag and liter_re.match(line):
            liter_flag = False
            lines[i] = ''
        elif liter_flag:
            line = ' ' + line
            lines[i] = line
        else:
            lines[i] = line.lstrip()

def setup(app):
    if False:
        return 10
    app.connect('autodoc-process-docstring', _process_docstring)