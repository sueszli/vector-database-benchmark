""" Indentation of code.

Language independent, the amount of the spaces is not configurable, as it needs
to be the same as in templates.
"""

def _indentedCode(codes, count):
    if False:
        return 10
    return '\n'.join((' ' * count + line if line and (not line.startswith('#')) else line for line in codes))

def indented(codes, level=1, vert_block=False):
    if False:
        for i in range(10):
            print('nop')
    if type(codes) is str:
        codes = codes.split('\n')
    if vert_block and codes != ['']:
        codes.insert(0, '')
        codes.append('')
    return _indentedCode(codes, level * 4)

def getCommentCode(comment, emit):
    if False:
        for i in range(10):
            print('nop')
    emit('// ' + comment)