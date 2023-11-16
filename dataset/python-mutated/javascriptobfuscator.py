"""deobfuscator for scripts messed up with JavascriptObfuscator.com"""
import re
PRIORITY = 1

def smartsplit(code):
    if False:
        print('Hello World!')
    'Split `code` at " symbol, only if it is not escaped.'
    strings = []
    pos = 0
    while pos < len(code):
        if code[pos] == '"':
            word = ''
            pos += 1
            while pos < len(code):
                if code[pos] == '"':
                    break
                if code[pos] == '\\':
                    word += '\\'
                    pos += 1
                word += code[pos]
                pos += 1
            strings.append('"%s"' % word)
        pos += 1
    return strings

def detect(code):
    if False:
        return 10
    'Detects if `code` is JavascriptObfuscator.com packed.'
    return re.search('^var _0x[a-f0-9]+ ?\\= ?\\[', code) is not None

def unpack(code):
    if False:
        i = 10
        return i + 15
    'Unpacks JavascriptObfuscator.com packed code.'
    if detect(code):
        matches = re.search('var (_0x[a-f\\d]+) ?\\= ?\\[(.*?)\\];', code)
        if matches:
            variable = matches.group(1)
            dictionary = smartsplit(matches.group(2))
            code = code[len(matches.group(0)):]
            for (key, value) in enumerate(dictionary):
                code = code.replace('%s[%s]' % (variable, key), value)
    return code