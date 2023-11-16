import re
import sys
from collections import OrderedDict
reCommand = re.compile('gam ')
reProduction = re.compile('<(.*?)>\\s+::=')
reBlankLine = re.compile('\\s*')
reWhitespace = re.compile('\\s+')
reProdRef = re.compile('<(.*?)>')
reBrackets = re.compile('[<>]')
reBadLink1 = re.compile('[/a-zA-Z]+>')
reBadLink2 = re.compile('<[/a-zA-Z]+')
specialCases = [[re.compile('<(Number in range )(\\d+)-(\\d+)>'), '&lt;\\1\\2-\\3&gt;'], ['<|<=|>=|>|=|!=', '&lt; | &lt;= | &gt;= | &gt; | = | !='], ['<RRULE, EXRULE, RDATE and EXDATE line>', '&lt;RRULE, EXRULE, RDATE and EXDATE line&gt;']]
commands = []
productions = OrderedDict()
prefix = '<html>\n    <head></head>\n    <body style="font-size:large; font-family:monospace">\n    <p><a href="#_commands">Commands</a>\n    <p><a href="#_parameters">Parameters</a>\n'
suffix = '    </body>\n</html>\n'

def examine(line):
    if False:
        i = 10
        return i + 15
    if reBlankLine.fullmatch(line):
        return (0, '', '')
    m = reCommand.match(line)
    if m:
        return (1, '', line)
    m = reProduction.match(line)
    if m:
        return (2, m.group(1), line)
    return (3, '', line)

def save(state, key, data):
    if False:
        while True:
            i = 10
    global commands, productions
    if state == 1:
        commands += [data]
    elif state == 2:
        if key in productions:
            if data == productions[key]:
                pass
            else:
                sys.stderr.write(f'Conflicting duplicate production {key} ignored\n')
        else:
            productions[key] = data

def fixSpecialCases(line):
    if False:
        for i in range(10):
            print('nop')
    for c in specialCases:
        if isinstance(c[0], re.Pattern):
            if c[0].search(line):
                line = c[0].sub(c[1], line)
        else:
            pos = line.find(c[0])
            if pos >= 0:
                line = line[:pos] + c[1] + line[pos + len(c[1]):]
    return line

def resolve(data, id=None):
    if False:
        while True:
            i = 10
    if id:
        result = f'<div id="__{id}" style="padding:12px; padding-left:100px; text-indent:-90px">'
    else:
        result = f'<div style="padding:12px; padding-left:100px; text-indent:-90px">'
    for line in data:
        if reWhitespace.match(line):
            result += '<br/>'
        result += reProdRef.sub('<a href="#__\\1">&lt;\\1&gt;</a>', line)
    result += '</div>'
    return result

def validate(line):
    if False:
        print('Hello World!')
    blist = reBrackets.findall(line)
    depth = 0
    nnest = False
    for c in blist:
        if c == '<':
            depth += 1
        elif c == '>':
            depth -= 1
        if depth > 1:
            nnest = True
    if depth != 0 or nnest:
        return False
    for m in reBadLink1.finditer(line):
        if m.start() == 0 or line[m.start() - 1] != '<':
            return False
    for m in reBadLink2.finditer(line):
        if m.end() >= len(line) or line[m.end()] != '>':
            return False
    return True
if __name__ == '__main__':
    if len(sys.argv) != 2:
        sys.stderr.write('Single argument must be the input file name\n')
        sys.exit(1)
    fname = sys.argv[1]
    state = 0
    buffer = []
    lnum = 0
    key = ''
    with open(fname, 'r') as infile:
        for line in infile:
            lnum += 1
            line = fixSpecialCases(line.rstrip())
            if not validate(line):
                sys.stderr.write(f'Unbalanced angle-brackets in line {lnum}\n')
            else:
                (token, k, line) = examine(line)
                if state == 0:
                    if token == 1 or token == 2:
                        buffer = [line]
                        if token == 2:
                            key = k
                        state = token
                elif token < 3:
                    save(state, key, buffer)
                    buffer = [line]
                    if token == 2:
                        key = k
                    state = token
                else:
                    buffer.append(line)
        if state > 0:
            save(state, key, buffer)
    print(prefix)
    print('<h1 id="_commands">Commands</h1>')
    for v in sorted(commands):
        print(resolve(v))
    print('<h1 id="_parameters">Parameters</h1>')
    for (k, v) in productions.items():
        print(resolve(v, k))
    print(suffix)