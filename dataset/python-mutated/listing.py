"""
Contains the listing of all code generator invocations.
"""

def generate_all(projectdir):
    if False:
        for i in range(10):
            print('nop')
    '\n    Generates all source files in targetdir.\n    '
    from .cpp_testlist import generate_testlist
    generate_testlist(projectdir)
    from .coord import generate_coord_basetypes
    generate_coord_basetypes(projectdir)