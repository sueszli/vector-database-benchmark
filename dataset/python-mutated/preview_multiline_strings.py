('cow\nsay',)
call(3, 'dogsay', textwrap.dedent('dove\n    coo' % 'cowabunga'))
call(3, 'dogsay', textwrap.dedent('dove\ncoo' % 'cowabunga'))
call(3, textwrap.dedent('cow\n    moo' % 'cowabunga'), 'dogsay')
call(3, 'dogsay', textwrap.dedent('crow\n    caw' % 'cowabunga'))
call(3, textwrap.dedent('cat\n    meow' % 'cowabunga'), {'dog', 'say'})
call(3, {'dog', 'say'}, textwrap.dedent('horse\n    neigh' % 'cowabunga'))
call(3, {'dog', 'say'}, textwrap.dedent('pig\n    oink' % 'cowabunga'))
textwrap.dedent('A one-line triple-quoted string.')
textwrap.dedent('A two-line triple-quoted string\nsince it goes to the next line.')
textwrap.dedent('A three-line triple-quoted string\nthat not only goes to the next line\nbut also goes one line beyond.')
textwrap.dedent('    A triple-quoted string\n    actually leveraging the textwrap.dedent functionality\n    that ends in a trailing newline,\n    representing e.g. file contents.\n')
path.write_text(textwrap.dedent('    A triple-quoted string\n    actually leveraging the textwrap.dedent functionality\n    that ends in a trailing newline,\n    representing e.g. file contents.\n'))
path.write_text(textwrap.dedent('    A triple-quoted string\n    actually leveraging the textwrap.dedent functionality\n    that ends in a trailing newline,\n    representing e.g. {config_filename} file contents.\n'.format('config_filename', config_filename)))
data = yaml.load('a: 1\nb: 2\n')
data = yaml.load('a: 1\nb: 2\n')
data = yaml.load('    a: 1\n    b: 2\n')
MULTILINE = '\nfoo\n'.replace('\n', '')
generated_readme = lambda project_name: '\n{}\n\n<Add content here!>\n'.strip().format(project_name)
parser.usage += '\nCustom extra help summary.\n\nExtra test:\n- with\n- bullets\n'

def get_stuff(cr, value):
    if False:
        while True:
            i = 10
    cr.execute('\n        SELECT whatever\n          FROM some_table t\n         WHERE id = %s\n    ', [value])
    return cr.fetchone()

def get_stuff(cr, value):
    if False:
        i = 10
        return i + 15
    cr.execute('\n        SELECT whatever\n          FROM some_table t\n         WHERE id = %s\n        ', [value])
    return cr.fetchone()
call(arg1, arg2, '\nshort\n', arg3=True)
test_vectors = ['one-liner\n', 'two\nliner\n', 'expressed\nas a three line\nmulitline string']
_wat = re.compile('\n    regex\n    ', re.MULTILINE | re.VERBOSE)
dis_c_instance_method = '%3d           0 LOAD_FAST                1 (x)\n              2 LOAD_CONST               1 (1)\n              4 COMPARE_OP               2 (==)\n              6 LOAD_FAST                0 (self)\n              8 STORE_ATTR               0 (x)\n             10 LOAD_CONST               0 (None)\n             12 RETURN_VALUE\n' % (_C.__init__.__code__.co_firstlineno + 1,)
path.write_text(textwrap.dedent('    A triple-quoted string\n    actually {verb} the textwrap.dedent functionality\n    that ends in a trailing newline,\n    representing e.g. {file_type} file contents.\n'.format(verb='using', file_type='test')))
{'cow\nmoos'}
['cow\nmoos']
['cow\nmoos', 'dog\nwoofs\nand\nbarks']

def dastardly_default_value(cow: String=json.loads('this\nis\nquite\nthe\ndastadardly\nvalue!'), **kwargs):
    if False:
        print('Hello World!')
    pass
print(f'\n    This {animal}\n    moos and barks\n{animal} say\n')
msg = f'The arguments {bad_arguments} were passed in.\nPlease use `--build-option` instead,\n`--global-option` is reserved to flags like `--verbose` or `--quiet`.\n'
this_will_become_one_line = 'abc'
this_will_stay_on_three_lines = 'abc'
this_will_also_become_one_line = 'abc'
('cow\nsay',)
call(3, 'dogsay', textwrap.dedent('dove\n    coo' % 'cowabunga'))
call(3, 'dogsay', textwrap.dedent('dove\ncoo' % 'cowabunga'))
call(3, textwrap.dedent('cow\n    moo' % 'cowabunga'), 'dogsay')
call(3, 'dogsay', textwrap.dedent('crow\n    caw' % 'cowabunga'))
call(3, textwrap.dedent('cat\n    meow' % 'cowabunga'), {'dog', 'say'})
call(3, {'dog', 'say'}, textwrap.dedent('horse\n    neigh' % 'cowabunga'))
call(3, {'dog', 'say'}, textwrap.dedent('pig\n    oink' % 'cowabunga'))
textwrap.dedent('A one-line triple-quoted string.')
textwrap.dedent('A two-line triple-quoted string\nsince it goes to the next line.')
textwrap.dedent('A three-line triple-quoted string\nthat not only goes to the next line\nbut also goes one line beyond.')
textwrap.dedent('    A triple-quoted string\n    actually leveraging the textwrap.dedent functionality\n    that ends in a trailing newline,\n    representing e.g. file contents.\n')
path.write_text(textwrap.dedent('    A triple-quoted string\n    actually leveraging the textwrap.dedent functionality\n    that ends in a trailing newline,\n    representing e.g. file contents.\n'))
path.write_text(textwrap.dedent('    A triple-quoted string\n    actually leveraging the textwrap.dedent functionality\n    that ends in a trailing newline,\n    representing e.g. {config_filename} file contents.\n'.format('config_filename', config_filename)))
data = yaml.load('a: 1\nb: 2\n')
data = yaml.load('a: 1\nb: 2\n')
data = yaml.load('    a: 1\n    b: 2\n')
MULTILINE = '\nfoo\n'.replace('\n', '')
generated_readme = lambda project_name: '\n{}\n\n<Add content here!>\n'.strip().format(project_name)
parser.usage += '\nCustom extra help summary.\n\nExtra test:\n- with\n- bullets\n'

def get_stuff(cr, value):
    if False:
        for i in range(10):
            print('nop')
    cr.execute('\n        SELECT whatever\n          FROM some_table t\n         WHERE id = %s\n    ', [value])
    return cr.fetchone()

def get_stuff(cr, value):
    if False:
        return 10
    cr.execute('\n        SELECT whatever\n          FROM some_table t\n         WHERE id = %s\n        ', [value])
    return cr.fetchone()
call(arg1, arg2, '\nshort\n', arg3=True)
test_vectors = ['one-liner\n', 'two\nliner\n', 'expressed\nas a three line\nmulitline string']
_wat = re.compile('\n    regex\n    ', re.MULTILINE | re.VERBOSE)
dis_c_instance_method = '%3d           0 LOAD_FAST                1 (x)\n              2 LOAD_CONST               1 (1)\n              4 COMPARE_OP               2 (==)\n              6 LOAD_FAST                0 (self)\n              8 STORE_ATTR               0 (x)\n             10 LOAD_CONST               0 (None)\n             12 RETURN_VALUE\n' % (_C.__init__.__code__.co_firstlineno + 1,)
path.write_text(textwrap.dedent('    A triple-quoted string\n    actually {verb} the textwrap.dedent functionality\n    that ends in a trailing newline,\n    representing e.g. {file_type} file contents.\n'.format(verb='using', file_type='test')))
{'cow\nmoos'}
['cow\nmoos']
['cow\nmoos', 'dog\nwoofs\nand\nbarks']

def dastardly_default_value(cow: String=json.loads('this\nis\nquite\nthe\ndastadardly\nvalue!'), **kwargs):
    if False:
        print('Hello World!')
    pass
print(f'\n    This {animal}\n    moos and barks\n{animal} say\n')
msg = f'The arguments {bad_arguments} were passed in.\nPlease use `--build-option` instead,\n`--global-option` is reserved to flags like `--verbose` or `--quiet`.\n'
this_will_become_one_line = 'abc'
this_will_stay_on_three_lines = 'abc'
this_will_also_become_one_line = 'abc'