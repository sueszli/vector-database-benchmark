from antlr4 import ParserRuleContext
from posthog.hogql.errors import SyntaxException

def parse_string(text: str) -> str:
    if False:
        return 10
    'Converts a string received from antlr via ctx.getText() into a Python string'
    if text.startswith("'") and text.endswith("'"):
        text = text[1:-1]
        text = text.replace("''", "'")
        text = text.replace("\\'", "'")
    elif text.startswith('"') and text.endswith('"'):
        text = text[1:-1]
        text = text.replace('""', '"')
        text = text.replace('\\"', '"')
    elif text.startswith('`') and text.endswith('`'):
        text = text[1:-1]
        text = text.replace('``', '`')
        text = text.replace('\\`', '`')
    elif text.startswith('{') and text.endswith('}'):
        text = text[1:-1]
        text = text.replace('{{', '{')
        text = text.replace('\\{', '{')
    else:
        raise SyntaxException(f'Invalid string literal, must start and end with the same quote type: {text}')
    text = text.replace('\\b', '\x08')
    text = text.replace('\\f', '\x0c')
    text = text.replace('\\r', '\r')
    text = text.replace('\\n', '\n')
    text = text.replace('\\t', '\t')
    text = text.replace('\\0', '')
    text = text.replace('\\a', '\x07')
    text = text.replace('\\v', '\x0b')
    text = text.replace('\\\\', '\\')
    return text

def parse_string_literal(ctx: ParserRuleContext) -> str:
    if False:
        print('Hello World!')
    'Converts a STRING_LITERAL received from antlr via ctx.getText() into a Python string'
    text = ctx.getText()
    return parse_string(text)