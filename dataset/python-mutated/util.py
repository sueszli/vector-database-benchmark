"""
Some utilities.
"""
import logging
import os
SHEBANG = '#!/.*\n(#?\n)?'
FILECACHE = {}
BADUTF8FILES = set()

def log_setup(setting, default=1):
    if False:
        for i in range(10):
            print('nop')
    '\n    Perform setup for the logger.\n    Run before any logging.log thingy is called.\n\n    if setting is 0: the default is used, which is WARNING.\n    else: setting + default is used.\n    '
    levels = (logging.ERROR, logging.WARNING, logging.INFO, logging.DEBUG, logging.NOTSET)
    factor = clamp(default + setting, 0, len(levels) - 1)
    level = levels[factor]
    logging.basicConfig(level=level, format='[%(asctime)s] %(message)s')
    logging.captureWarnings(True)

def clamp(number, smallest, largest):
    if False:
        for i in range(10):
            print('nop')
    ' return number but limit it to the inclusive given value range '
    return max(smallest, min(number, largest))

class Strlazy:
    """
    to be used like this: logging.debug("rolf %s", strlazy(lambda: do_something()))
    so do_something is only called when the debug message is actually printed
    do_something could also be an f-string.
    """

    def __init__(self, fun):
        if False:
            return 10
        self.fun = fun

    def __str__(self):
        if False:
            return 10
        return self.fun()

def has_ext(fname, exts):
    if False:
        print('Hello World!')
    '\n    Returns true if fname ends in any of the extensions in ext.\n    '
    for ext in exts:
        if ext == '':
            if os.path.splitext(fname)[1] == '':
                return True
        elif fname.endswith(ext):
            return True
    return False

def readfile(filename):
    if False:
        while True:
            i = 10
    '\n    reads the file, and returns it as a str object.\n\n    if the file has already been read in the past,\n    returns it from the cache.\n    '
    if filename not in FILECACHE:
        with open(filename, 'rb') as fileobj:
            data = fileobj.read()
        try:
            data = data.decode('utf-8')
        except UnicodeDecodeError:
            data = data.decode('utf-8', errors='replace')
            BADUTF8FILES.add(filename)
        FILECACHE[filename] = data
    return FILECACHE[filename]

def writefile(filename, new_content):
    if False:
        return 10
    '\n    writes the file and update it in the cache.\n    '
    if filename in BADUTF8FILES:
        raise ValueError(f'{filename}: cannot write due to utf8-errors.')
    with open(filename, 'w', encoding='utf8') as fileobj:
        fileobj.write(new_content)
    FILECACHE[filename] = new_content

def findfiles(paths, exts=None):
    if False:
        print('Hello World!')
    '\n    yields all files in paths with names ending in an ext from exts.\n\n    If exts is None, all extensions are accepted.\n\n    hidden dirs and files are ignored.\n    '
    for path in paths:
        for filename in os.listdir(path):
            if filename.startswith('.'):
                continue
            filename = os.path.join(path, filename)
            if os.path.isdir(filename):
                yield from findfiles((filename,), exts)
                continue
            if exts is None or has_ext(filename, exts):
                yield filename

def issue_str(title, filename, fix=None):
    if False:
        while True:
            i = 10
    '\n    Creates a formated (title, text) desciption of an issue.\n\n    TODO use this function and issue_str_line for all issues, so the format\n    can be easily changed (exta text, colors, etc)\n    '
    return (title, filename, fix)

def issue_str_line(title, filename, line, line_number, highlight, fix=None):
    if False:
        print('Hello World!')
    '\n    Creates a formated (title, text) desciption of an issue with information\n    about the location in the file.\n    line:        line content\n    line_number: line id in the file\n    highlight:   a tuple of (start, end), where\n        start:   match start in the line\n        end:     match end in the line\n    '
    (start, end) = highlight
    start += 1
    line = line.replace('\n', '').replace('\t', ' ')
    return (title, filename + '\n\tline: ' + str(line_number) + "\n\tat:   '" + line + "'\n\t      " + ' ' * start + '\x1b[32;1m^' + '~' * (end - start) + '\x1b[m', fix)