"""
Statement pre-processors.
"""

def clean_whitespace(statement):
    if False:
        return 10
    '\n    Remove any consecutive whitespace characters from the statement text.\n    '
    import re
    statement.text = statement.text.replace('\n', ' ').replace('\r', ' ').replace('\t', ' ')
    statement.text = statement.text.strip()
    statement.text = re.sub(' +', ' ', statement.text)
    return statement

def unescape_html(statement):
    if False:
        for i in range(10):
            print('nop')
    '\n    Convert escaped html characters into unescaped html characters.\n    For example: "&lt;b&gt;" becomes "<b>".\n    '
    import html
    statement.text = html.unescape(statement.text)
    return statement

def convert_to_ascii(statement):
    if False:
        return 10
    '\n    Converts unicode characters to ASCII character equivalents.\n    For example: "på fédéral" becomes "pa federal".\n    '
    import unicodedata
    text = unicodedata.normalize('NFKD', statement.text)
    text = text.encode('ascii', 'ignore').decode('utf-8')
    statement.text = str(text)
    return statement