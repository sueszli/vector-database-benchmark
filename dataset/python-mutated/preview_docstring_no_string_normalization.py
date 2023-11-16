def do_not_touch_this_prefix():
    if False:
        print('Hello World!')
    'There was a bug where docstring prefixes would be normalized even with -S.'

def do_not_touch_this_prefix2():
    if False:
        while True:
            i = 10
    f'There was a bug where docstring prefixes would be normalized even with -S.'

def do_not_touch_this_prefix3():
    if False:
        print('Hello World!')
    u'There was a bug where docstring prefixes would be normalized even with -S.'