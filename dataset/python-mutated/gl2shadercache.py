from __future__ import print_function
import re
import io
import os
import renpy
shader_part = {}

def register_shader(name, **kwargs):
    if False:
        return 10
    '\n    :doc: register_shader\n\n    This registers a shader part. This takes `name`, and then\n    keyword arguments.\n\n    `name`\n        A string giving the name of the shader part. Names starting with an\n        underscore or "renpy." are reserved for Ren\'Py.\n\n    `variables`\n        The variables used by the shader part. These should be listed one per\n        line, a storage (uniform, attribute, or varying) followed by a type,\n        name, and semicolon. For example::\n\n            variables=\'\'\'\n            uniform sampler2D tex0;\n            attribute vec2 a_tex_coord;\n            varying vec2 v_tex_coord;\n            \'\'\'\n\n    `vertex_functions`\n        If given, a string containing functions that will be included in the\n        vertex shader.\n\n    `fragment_functions`\n        If given, a string containing functions that will be included in the\n        fragment shader.\n\n    Other keyword arguments should start with ``vertex_`` or ``fragment_``,\n    and end with an integer priority. So "fragment_200" or "vertex_300". These\n    give text that\'s placed in the appropriate shader at the given priority,\n    with lower priority numbers inserted before higher priority numbers.\n    '
    ShaderPart(name, **kwargs)

class ShaderPart(object):
    """
    Arguments are as for register_shader.

    """

    def __init__(self, name, variables='', vertex_functions='', fragment_functions='', **kwargs):
        if False:
            print('Hello World!')
        if not re.match('^[\\w\\.]+$', name):
            raise Exception('The shader name {!r} contains an invalid character. Shader names are limited to ASCII alphanumeric characters, _, and .'.format(name))
        self.name = name
        shader_part[name] = self
        self.vertex_functions = vertex_functions
        self.fragment_functions = fragment_functions
        self.vertex_parts = []
        self.fragment_parts = []
        self.vertex_variables = set()
        self.fragment_variables = set()
        vertex_used = set()
        fragment_used = set()
        for (k, v) in kwargs.items():
            (shader, _, priority) = k.partition('_')
            if not priority:
                shader = None
            try:
                priority = int(priority)
            except Exception:
                shader = None
            if shader == 'vertex':
                parts = self.vertex_parts
                used = vertex_used
            elif shader == 'fragment':
                parts = self.fragment_parts
                used = fragment_used
            else:
                raise Exception('Keyword arguments to ShaderPart must be of the form {vertex,fragment}_{priority}.')
            parts.append((priority, v))
            for m in re.finditer('\\b\\w+\\b', v):
                used.add(m.group(0))
        for l in variables.split('\n'):
            l = l.partition('//')[0].strip(' ;')
            a = l.split()
            if not a:
                continue
            a = tuple(a)
            if len(a) != 3:
                raise Exception("{}: Unknown shader variable line {!r}. Only the form '{{uniform,attribute,vertex}} {{type}} {{name}} is allowed.".format(self.name, l))
            kind = a[0]
            name = a[2]
            if name in vertex_used:
                self.vertex_variables.add(a)
            if name in fragment_used:
                self.fragment_variables.add(a)
            if kind == 'uniform':
                renpy.display.transform.add_uniform(name)
        self.raw_variables = variables
cache = {}

def source(variables, parts, functions, fragment, gles):
    if False:
        print('Hello World!')
    '\n    Given lists of variables and parts, converts them into textual source\n    code for a shader.\n\n    `fragment`\n        Should be set to true to generate the code for a fragment shader.\n    '
    rv = []
    if gles:
        rv.append('#version 100\n')
        if fragment:
            rv.append('precision mediump float;\n')
    else:
        rv.append('#version 120\n')
    rv.extend(functions)
    for (storage, type_, name) in sorted(variables):
        rv.append('{} {} {};\n'.format(storage, type_, name))
    rv.append('\nvoid main() {\n')
    parts.sort()
    for (_, part) in parts:
        rv.append(part)
    rv.append('}\n')
    return ''.join(rv)

class ShaderCache(object):
    """
    This class caches shaders that were compiled. It's also responsible for
    recording shaders that have been used, persisting them to disk, and then
    loading the shaders back into the cache.
    """

    def __init__(self, filename, gles):
        if False:
            for i in range(10):
                print('nop')
        self.filename = filename
        self.gles = gles
        self.cache = {}
        self.missing = set()
        self.dirty = False

    def get(self, partnames):
        if False:
            print('Hello World!')
        '\n        Gets a shader, creating it if necessary.\n\n        `partnames`\n            A tuple of strings, giving the names of the shader parts to include in\n            the cache.\n        '
        rv = self.cache.get(partnames, None)
        if rv is not None:
            return rv
        partnameset = set()
        partnamenotset = set()
        for i in partnames:
            if i.startswith('-'):
                partnamenotset.add(i[1:])
            else:
                partnameset.add(i)
        partnameset -= partnamenotset
        if 'renpy.ftl' not in partnameset:
            partnameset.add(renpy.config.default_shader)
        sortedpartnames = tuple(sorted(partnameset))
        rv = self.cache.get(sortedpartnames, None)
        if rv is not None:
            self.cache[partnames] = rv
            return rv
        vertex_variables = set()
        vertex_parts = []
        vertex_functions = []
        fragment_variables = set()
        fragment_parts = []
        fragment_functions = []
        for i in sortedpartnames:
            p = shader_part.get(i, None)
            if p is None:
                raise Exception('{!r} is not a known shader part.'.format(i))
            vertex_variables |= p.vertex_variables
            vertex_parts.extend(p.vertex_parts)
            vertex_functions.append(p.vertex_functions)
            fragment_variables |= p.fragment_variables
            fragment_parts.extend(p.fragment_parts)
            fragment_functions.append(p.fragment_functions)
        vertex = source(vertex_variables, vertex_parts, vertex_functions, False, self.gles)
        fragment = source(fragment_variables, fragment_parts, fragment_functions, True, self.gles)
        self.log_shader('vertex', sortedpartnames, vertex)
        self.log_shader('fragment', sortedpartnames, fragment)
        from renpy.gl2.gl2shader import Program
        rv = Program(sortedpartnames, vertex, fragment)
        rv.load()
        self.cache[partnames] = rv
        self.cache[sortedpartnames] = rv
        self.dirty = True
        return rv

    def check(self, partnames):
        if False:
            while True:
                i = 10
        '\n        Returns true if every part in partnames is a known part, or False\n        otherwise.\n        '
        for i in partnames:
            if i not in shader_part:
                return False
        return True

    def save(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Saves the list of shaders to the file.\n        '
        if not self.dirty:
            return
        if not renpy.config.developer:
            return
        fn = '<unknown>'
        try:
            fn = os.path.join(renpy.config.gamedir, renpy.loader.get_path(self.filename))
            tmp = fn + '.tmp'
            with io.open(tmp, 'w', encoding='utf-8') as f:
                shaders = set(self.cache.keys()) | self.missing
                for i in shaders:
                    f.write(u' '.join(i) + '\r\n')
            try:
                os.unlink(fn)
            except Exception:
                pass
            os.rename(tmp, fn)
            self.dirty = False
        except Exception:
            renpy.display.log.write('Saving shaders to {!r}:'.format(fn))
            renpy.display.log.exception()

    def load(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Loads the list of shaders from the file, and compiles all shaders\n        for which the parts exist, and for which compilation can succeed.\n        '
        try:
            with renpy.loader.load(self.filename) as f:
                for l in f:
                    l = l.strip().decode('utf-8')
                    partnames = tuple(l.strip().split())
                    if not partnames:
                        continue
                    if not self.check(partnames):
                        self.missing.add(partnames)
                        continue
                    try:
                        self.get(partnames)
                    except Exception:
                        renpy.display.log.write('Precompiling shader {!r}:'.format(partnames))
                        renpy.display.log.exception()
                        self.missing.add(partnames)
        except Exception:
            renpy.display.log.write('Could not open {!r}:'.format(self.filename))
            return

    def clear(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Clears the shader cache and the shaders inside it.\n        '
        self.cache.clear()
        self.missing.clear()

    def log_shader(self, kind, partnames, text):
        if False:
            while True:
                i = 10
        '\n        Logs the shader text to the log.\n        '
        if not renpy.config.log_gl_shaders:
            return
        name = kind + ' ' + ', '.join(partnames) + ' '
        name = name + '-' * max(0, 80 - len(name))
        renpy.display.log.write('%s', name)
        renpy.display.log.write('%s', text)
        renpy.display.log.write('-' * 80)