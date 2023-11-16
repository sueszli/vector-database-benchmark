"""
Convert simple documentation to epydoc/pydoctor-compatible markup
"""
from sys import stdin, stdout, argv
import os
from tempfile import mkstemp
from subprocess import call
import re
spaces = re.compile('\\s+')
singleLineExp = re.compile('\\s+"([^"]+)"')
commentStartExp = re.compile('\\s+"""')
commentEndExp = re.compile('"""$')
returnExp = re.compile('\\s+(returns:.*)')
lastindent = ''
comment = False

def fixParam(line):
    if False:
        i = 10
        return i + 15
    'Change foo: bar to @foo bar'
    result = re.sub('(\\w+):', '@param \\1', line)
    result = re.sub('   @', '@', result)
    return result

def fixReturns(line):
    if False:
        return 10
    'Change returns: foo to @return foo'
    return re.sub('returns:', '@returns', line)

def fixLine(line):
    if False:
        print('Hello World!')
    global comment
    match = spaces.match(line)
    if not match:
        return line
    else:
        indent = match.group(0)
    if singleLineExp.match(line):
        return re.sub('"', '"""', line)
    if commentStartExp.match(line):
        comment = True
    if comment:
        line = fixReturns(line)
        line = fixParam(line)
    if commentEndExp.search(line):
        comment = False
    return line

def test():
    if False:
        for i in range(10):
            print('nop')
    'Test transformations'
    assert fixLine(' "foo"') == ' """foo"""'
    assert fixParam('foo: bar') == '@param foo bar'
    assert commentStartExp.match('   """foo"""')

def funTest():
    if False:
        i = 10
        return i + 15
    testFun = 'def foo():\n   "Single line comment"\n   """This is a test"""\n      bar: int\n      baz: string\n      returns: junk"""\n   if True:\n       print "OK"\n'.splitlines(True)
    fixLines(testFun)

def fixLines(lines, fid):
    if False:
        i = 10
        return i + 15
    for line in lines:
        os.write(fid, fixLine(line))
if __name__ == '__main__':
    if False:
        funTest()
    infile = open(argv[1])
    (outfid, outname) = mkstemp()
    fixLines(infile.readlines(), outfid)
    infile.close()
    os.close(outfid)
    call(['doxypy', outname])