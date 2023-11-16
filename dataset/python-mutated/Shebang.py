""" Utils to work with shebang lines.

"""
import os
import re

def getShebangFromSource(source_code):
    if False:
        i = 10
        return i + 15
    'Given source code, extract the shebang (#!) part.\n\n    Notes:\n        This function is less relevant on Windows, because it will not use\n        this method of determining the execution. Still scripts aimed at\n        multiple platforms will contain it and it can be used to e.g. guess\n        the Python version expected, if it is a Python script at all.\n\n        There are variants of the function that will work on filenames instead.\n    Args:\n        source_code: The source code as a unicode string\n    Returns:\n        The binary and arguments that the kernel will use (Linux and compatible).\n    '
    if source_code.startswith('#!'):
        shebang = re.match('^#!\\s*(.*?)\\n', source_code)
        if shebang is not None:
            shebang = shebang.group(0).rstrip('\n')
    else:
        shebang = None
    return shebang

def getShebangFromFile(filename):
    if False:
        return 10
    'Given a filename, extract the shebang (#!) part from it.\n\n    Notes:\n        This function is less relevant on Windows, because it will not use\n        this method of determining the execution. Still scripts aimed at\n        multiple platforms will contain it and it can be used to e.g. guess\n        the Python version expected, if it is a Python script at all.\n\n        There are variants of the function that will work on file content\n        instead.\n    Args:\n        filename: The filename to get the shebang of\n    Returns:\n        The binary that the kernel will use (Linux and compatible).\n    '
    with open(filename, 'rb') as f:
        source_code = f.readline()
        if str is not bytes:
            try:
                source_code = source_code.decode('utf8')
            except UnicodeDecodeError:
                source_code = ''
        return getShebangFromSource(source_code)

def parseShebang(shebang):
    if False:
        for i in range(10):
            print('nop')
    'Given a concrete shebang value, it will extract the binary used.\n\n    Notes:\n        This function is less relevant on Windows, because it will not use\n        this method of determining the execution.\n\n        This handles that many times people use `env` binary to search the\n        PATH for an actual binary, e.g. `/usr/bin/env python3.7` where we\n        would care most about the `python3.7` part and want to see through\n        the `env` usage.\n    Args:\n        shebang: The shebang extracted with one of the methods to do so.\n    Returns:\n        The binary the kernel will use (Linux and compatible).\n    '
    parts = shebang.split()
    if os.path.basename(parts[0]) == 'env':
        del parts[0]
        while parts[0].startswith('-'):
            del parts[0]
        while '=' in parts[0]:
            del parts[0]
    return (parts[0][2:].lstrip(), parts[1:])