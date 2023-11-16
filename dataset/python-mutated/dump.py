from __future__ import division, absolute_import, with_statement, print_function, unicode_literals
from renpy.compat import PY2, basestring, bchr, bord, chr, open, pystr, range, round, str, tobytes, unicode
import inspect
import json
import sys
import os
import renpy
definitions = []
transforms = []
screens = []
file_exists_cache = {}

def file_exists(fn):
    if False:
        i = 10
        return i + 15
    rv = file_exists_cache.get(fn, None)
    if rv is None:
        fullfn = renpy.lexer.unelide_filename(fn)
        rv = os.path.exists(fullfn)
        file_exists_cache[fn] = rv
    return rv
completed_dump = False

def dump(error):
    if False:
        i = 10
        return i + 15
    '\n    Causes a JSON dump file to be written, if the user has requested it.\n\n    `error`\n        An error flag that is added to the written file.\n    '
    global completed_dump
    args = renpy.game.args
    if completed_dump:
        return
    completed_dump = True
    if not args.json_dump:
        return

    def name_filter(name, filename):
        if False:
            return 10
        '\n        Returns true if the name is included by the name_filter, or false if it is excluded.\n        '
        filename = filename.replace('\\', '/')
        if name.startswith('_') and (not args.json_dump_private):
            if name.startswith('__') and name.endswith('__'):
                pass
            else:
                return False
        if not file_exists(filename):
            return False
        if filename.startswith('common/') or filename.startswith('renpy/common/'):
            return args.json_dump_common
        if not filename.startswith('game/'):
            return False
        return True
    result = {}
    result['error'] = error
    result['size'] = [renpy.config.screen_width, renpy.config.screen_height]
    result['name'] = renpy.config.name
    result['version'] = renpy.config.version
    location = {}
    result['location'] = location
    label = location['label'] = {}
    for (name, n) in renpy.game.script.namemap.items():
        filename = n.filename
        line = n.linenumber
        if not isinstance(name, basestring):
            continue
        if not name_filter(name, filename):
            continue
        label[name] = [filename, line]
    define = location['define'] = {}
    for (name, filename, line) in definitions:
        if not name_filter(name, filename):
            continue
        define[name] = [filename, line]
    screen = location['screen'] = {}
    for (name, filename, line) in screens:
        if not name_filter(name, filename):
            continue
        screen[name] = [filename, line]
    transform = location['transform'] = {}
    for (name, filename, line) in transforms:
        if not name_filter(name, filename):
            continue
        transform[name] = [filename, line]

    def get_line(o):
        if False:
            print('Hello World!')
        "\n        Returns the filename and the first line number of the class or function o. Returns\n        None, None if unknown.\n\n        For a class, this doesn't return the first line number of the class, but rather\n        the line number of the first method in the class - hopefully.\n        "
        if inspect.isfunction(o):
            return (inspect.getfile(o), o.__code__.co_firstlineno)
        if inspect.ismethod(o):
            return get_line(o.__func__)
        return (None, None)
    code = location['callable'] = {}
    for (modname, mod) in sys.modules.copy().items():
        if mod is None:
            continue
        if modname == 'store':
            prefix = ''
        elif modname.startswith('store.'):
            prefix = modname[6:] + '.'
        else:
            continue
        for (name, o) in mod.__dict__.items():
            if inspect.isfunction(o):
                try:
                    if inspect.getmodule(o) != mod:
                        continue
                    (filename, line) = get_line(o)
                    if filename is None:
                        continue
                    if not name_filter(name, filename):
                        continue
                    code[prefix + name] = [filename, line]
                except Exception:
                    continue
            if inspect.isclass(o):
                for (methname, method) in o.__dict__.items():
                    try:
                        if inspect.getmodule(method) != mod:
                            continue
                        (filename, line) = get_line(method)
                        if filename is None:
                            continue
                        if not name_filter(name, filename):
                            continue
                        if not name_filter(methname, filename):
                            continue
                        code[prefix + name + '.' + methname] = [filename, line]
                    except Exception:
                        continue
    try:
        result['build'] = renpy.store.build.dump()
    except Exception:
        pass
    filename = renpy.exports.fsdecode(args.json_dump)
    if filename != '-':
        new = filename + '.new'
        if PY2:
            with open(new, 'wb') as f:
                json.dump(result, f)
        else:
            with open(new, 'w') as f:
                json.dump(result, f)
        if os.path.exists(filename):
            os.unlink(filename)
        os.rename(new, filename)
    else:
        json.dump(result, sys.stdout, indent=2)