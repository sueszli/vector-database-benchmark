"""
An example of the Template pattern in Python

*TL;DR
Defines the skeleton of a base algorithm, deferring definition of exact
steps to subclasses.

*Examples in Python ecosystem:
Django class based views: https://docs.djangoproject.com/en/2.1/topics/class-based-views/
"""

def get_text() -> str:
    if False:
        print('Hello World!')
    return 'plain-text'

def get_pdf() -> str:
    if False:
        return 10
    return 'pdf'

def get_csv() -> str:
    if False:
        for i in range(10):
            print('nop')
    return 'csv'

def convert_to_text(data: str) -> str:
    if False:
        print('Hello World!')
    print('[CONVERT]')
    return f'{data} as text'

def saver() -> None:
    if False:
        while True:
            i = 10
    print('[SAVE]')

def template_function(getter, converter=False, to_save=False) -> None:
    if False:
        for i in range(10):
            print('nop')
    data = getter()
    print(f'Got `{data}`')
    if len(data) <= 3 and converter:
        data = converter(data)
    else:
        print('Skip conversion')
    if to_save:
        saver()
    print(f'`{data}` was processed')

def main():
    if False:
        while True:
            i = 10
    '\n    >>> template_function(get_text, to_save=True)\n    Got `plain-text`\n    Skip conversion\n    [SAVE]\n    `plain-text` was processed\n\n    >>> template_function(get_pdf, converter=convert_to_text)\n    Got `pdf`\n    [CONVERT]\n    `pdf as text` was processed\n\n    >>> template_function(get_csv, to_save=True)\n    Got `csv`\n    Skip conversion\n    [SAVE]\n    `csv` was processed\n    '
if __name__ == '__main__':
    import doctest
    doctest.testmod()