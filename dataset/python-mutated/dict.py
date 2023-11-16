from __future__ import print_function
"\n>>> from dict_ext import *\n>>> def printer(*args):\n...     for x in args: print(x, end='')\n...     print('')\n...\n>>> print(new_dict())\n{}\n>>> print(data_dict())\n{1: {'key2': 'value2'}, 'key1': 'value1'}\n>>> tmp = data_dict()\n>>> print(dict_keys(tmp))\n[1, 'key1']\n>>> print(dict_values(tmp))\n[{'key2': 'value2'}, 'value1']\n>>> print(dict_items(tmp))\n[(1, {'key2': 'value2'}), ('key1', 'value1')]\n>>> print(dict_from_sequence([(1,1),(2,2),(3,3)]))\n{1: 1, 2: 2, 3: 3}\n>>> test_templates(printer) #doctest: +NORMALIZE_WHITESPACE\na test string\n13\nNone\n{1.5: 13, 1: 'a test string'}\ndefault\ndefault\n"

def run(args=None):
    if False:
        for i in range(10):
            print('nop')
    import sys
    import doctest
    if args is not None:
        sys.argv = args
    return doctest.testmod(sys.modules.get(__name__))
if __name__ == '__main__':
    print('running...')
    import sys
    status = run()[0]
    if status == 0:
        print('Done.')
    sys.exit(status)