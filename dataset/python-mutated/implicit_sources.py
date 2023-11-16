def foo():
    if False:
        i = 10
        return i + 15
    return '123.456.789.123'

def bar_format_strings():
    if False:
        return 10
    user_controlled = 1
    return f'{user_controlled}:123.456.789.123'

def bar_percent_format():
    if False:
        i = 10
        return i + 15
    user_controlled = 1
    return '%s:123.456.789.123' % (user_controlled,)

def bar_dot_format():
    if False:
        return 10
    user_controlled = 1
    return '{}:123.456.789.123'.format(user_controlled)

def does_not_match():
    if False:
        for i in range(10):
            print('nop')
    return '123.456'

def multiple_patterns():
    if False:
        while True:
            i = 10
    return '<123.456.789.123>'
GOOGLE_API_KEY = 'AIzaSyB2qiehH9CMRIuRVJghvnluwA1GvQ3FCe4'

def string_source_top_level():
    if False:
        i = 10
        return i + 15
    params = {'key': GOOGLE_API_KEY}
    return params

def string_source_not_top_level():
    if False:
        for i in range(10):
            print('nop')
    params = {'key': 'AIzaSyB2qiehH9CMRIuRVJghvnluwA1GvQ3FCe4'}
    return params

def string_source_top_level_local_overwrite():
    if False:
        while True:
            i = 10
    GOOGLE_API_KEY = 'safe'
    params = {'key': GOOGLE_API_KEY}
    return params

def string_literal_arguments_source(template: str, x):
    if False:
        return 10
    if 1 == 1:
        return template.format('SELECT1', 1)
    elif 1 == 1:
        return template % 'SELECT2'
    else:
        return x + 'SELECT3'
(START, BODY, END) = ('<html>', 'lorem ipsum', '</html>')

def toplevel_simultaneous_assignment():
    if False:
        print('Hello World!')
    return START + BODY + END