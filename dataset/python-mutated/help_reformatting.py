def reformat_help_message(help_message):
    if False:
        i = 10
        return i + 15
    return _collapse_usage_paragraph(_normalize_options(help_message))

def _normalize_options(help_message):
    if False:
        for i in range(10):
            print('nop')
    return help_message.replace('optional arguments', 'options')

def _collapse_usage_paragraph(help_message):
    if False:
        print('Hello World!')
    paragraphs = split_paragraphs(help_message)
    return '\n'.join([normalize_spaces(paragraphs[0]) + '\n'] + paragraphs[1:])

def normalize_spaces(text):
    if False:
        while True:
            i = 10
    return ' '.join(text.split())

def split_paragraphs(text):
    if False:
        for i in range(10):
            print('nop')
    paragraphs = []
    par = ''
    for line in text.splitlines(True):
        if _is_empty_line(line):
            paragraphs.append(par)
            par = ''
        else:
            par += line
    paragraphs.append(par)
    return paragraphs

def _is_empty_line(line):
    if False:
        i = 10
        return i + 15
    return '' == line.strip()