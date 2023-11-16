import re
import pyparsing
PartialQuotedString = pyparsing.Regex(re.compile('\n            "[^"]*(?:"|$)  # double-quoted string that ends with double quote or EOF\n            |\n            \'[^\']*(?:\'|$)  # single-quoted string that ends with double quote or EOF\n        ', re.VERBOSE))
expr = pyparsing.ZeroOrMore(PartialQuotedString | pyparsing.Word(' \r\n\t') | pyparsing.CharsNotIn('\'" \r\n\t')).leaveWhitespace()

def quote(val: str) -> str:
    if False:
        for i in range(10):
            print('nop')
    if val and all((char not in val for char in '\'" \r\n\t')):
        return val
    if '"' not in val:
        return f'"{val}"'
    if "'" not in val:
        return f"'{val}'"
    return '"' + val.replace('"', '\\x22') + '"'

def unquote(x: str) -> str:
    if False:
        for i in range(10):
            print('nop')
    if len(x) > 1 and x[0] in '\'"' and (x[0] == x[-1]):
        return x[1:-1]
    else:
        return x