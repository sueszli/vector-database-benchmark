from b2.build import type

def register():
    if False:
        print('Hello World!')
    type.register_type('EXE', ['exe'], None, ['NT', 'CYGWIN'])
    type.register_type('EXE', [], None, [])
register()