from os.path import join, abspath
__all__ = ['join_with_execdir', 'abspath', 'attr_is_not_kw', '_not_kw_even_if_listed_in_all', 'extra stuff', None]

def join_with_execdir(arg):
    if False:
        while True:
            i = 10
    return join(abspath('.'), arg)

def not_in_all():
    if False:
        print('Hello World!')
    pass
attr_is_not_kw = 'Listed in __all__ but not a fuction'

def _not_kw_even_if_listed_in_all():
    if False:
        print('Hello World!')
    print('Listed in __all__ but starts with an underscore')