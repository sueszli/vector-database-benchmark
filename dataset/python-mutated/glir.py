"""GL Intermediate Representation Desktop Implementation
=====================================================

The glir module holds the desktop implementation of the GL Intermediate
Representation (GLIR). Parsing and handling of the GLIR for other platforms
can be found in external libraries.

We propose the specification of a simple intermediate representation for
OpenGL. It provides a means to serialize a visualization, so that the
high-level API and the part that does the GL commands can be separated,
and even be in separate processes.

GLIR is a high level representation that consists of commands without
return values. In effect, the commands can be streamed from one node to
another without having to wait for a reply. Only in the event of an
error information needs to go in the other direction, but this can be
done asynchronously.

The purpose for GLIR has been to allow the usage gloo (our high level object
oriented interface to OpenGL), while executing the visualization in the
browser (via JS/WebGL). The fact that the stream of commands is
one-directional is essential to realize reactive visualizations.

The separation between API and implementation also provides a nice
abstraction leading to cleaner code.

GLIR commands are represented as tuples. As such the overhead for
"parsing" the commands is minimal. The commands can, however, be
serialized so they can be send to another process. Further, a series of
GLIR commands can be stored in a file. This way we can store
visualizations to disk that can be displayed with any application that
can interpret GLIR.

The GLIR specification is tied to the version of the vispy python library
that supports it. The current specification described below was first
created for::

    VisPy 0.6

The shape of a command
~~~~~~~~~~~~~~~~~~~~~~

GLIR consists of a sequence of commands that are defined as tuples. Each
command has the following shape:

::

   (<command>, <ID>, [arg1, [arg2, [arg3]]])

-  ``<command>`` is one of 15 commands: CURRENT, CREATE, DELETE,
   UNIFORM, ATTRIBUTE, DRAW, SIZE, DATA, WRAPPING,
   INTERPOLATION, ATTACH, FRAMEBUFFER, FUNC, SWAP, LINK.
-  In all commands except SET, ``<ID>`` is an integer unique within the
   current GL context that is used as a reference to a GL object. It is
   the responsibility of the code that generates the command to keep
   track of id's and to ensure that they are unique.
-  The number of arguments and their type differs per command and are
   explained further below.
-  Some commands accept GL enums in the form of a string. In these cases
   the enum can also be given as an int, but a string is recommended for
   better debugging. The string is case insensitive.

CURRENT
~~~~~~~

::

   ('CURRENT', 0)

Will be called when the context is made current. The GLIR implementation
can use this to reset some caches.

CREATE
~~~~~~

::

   ('CREATE', <id>, <class:str>)
   # Example:
   ('CREATE', 4, 'VertexBuffer')

Applies to: All objects

The create command is used to create a new GL object. It has one string
argument that can be any of 10 classes: 'Program', 'VertexBuffer',
'IndexBuffer', 'Texture2D', 'Texture3D', 'RenderBuffer', 'FrameBuffer',
'VertexShader', 'FragmentShader', 'GeometryShader'

DELETE
~~~~~~

::

   ('DELETE', <id>)
   # Example:
   ('DELETE', 4)

Applies to: All objects

The delete command is used to delete the GL object corresponding to the
given id. If the id does not exist, this command is ignored. This
command does not have arguments. When used with Shader objects, the
shader is freed from GPU memory.

UNIFORM
~~~~~~~

::

   ('UNIFORM', <program_id>, <name:str>, <type:str>, <value>)
   # Examples:
   ('UNIFORM', 4, 'u_scale', 'vec3', <array 3>)

Applies to: Program

This command is used to set the uniform of a program object. A uniform
has a string name, a type, and a value.

The type can be 'float', 'vec2', 'vec3', 'vec4', 'int', 'ivec2',
'ivec3', 'ivec4', 'bool', 'bvec2', 'bvec3', 'bvec4', 'mat2', 'mat3',
'mat4'. The value must be tuple or array with number of elements that
matches with the type.

It is an error to provide this command before the shaders are set. After
resetting shaders, all uniforms and attributes have to be re-submitted.

Discussion: for the uniform and attribute commands, the type argument
should not strictly be necessary, but it makes the GLIR implementation
simpler. Plus in gloo we *have* this information.

TEXTURE
~~~~~~~

::

   ('TEXTURE', <program_id>, <name:str>, <texture_id>)
   # Examples:
   ('TEXTURE', 4, 'u_texture1', 6)

Applies to: Program

This command is used to link a texture to a GLSL uniform sampler.

ATTRIBUTE
~~~~~~~~~

::

   ('ATTRIBUTE', <program_id>, <name:str>, <type:str>, <vbo_id>, <stride:int>, <offset:int>)
   # Example: Buffer id 5, stride 4, offset 0
   ('ATTRIBUTE', 4, 'a_position', 'vec3', 5, 4, 0)

Applies to: Program

This command is used to set the attribute of a program object. An
attribute has a string name, a type, and a value.

The type can be 'float', 'vec2', 'vec3', 'vec4'. If the first value
element is zero, the remaining elements represent the data to pass to
``glVertexAttribNf``.

It is an error to provide this command before the shaders are set. After
resetting shaders, all uniforms and attributes have to be re-submitted.

DRAW
~~~~

::

   ('DRAW', <program_id>, <mode:str>, <selection:tuple>, <instances:int>)
   # Example: Draw 100 lines with non-instanced rendering
   ('DRAW', 4, 'lines', (0, 100), 1)
   # Example: Draw 100 lines using index buffer with id 5
   ('DRAW', 4, 'points', (5, 'unsigned_int', 100), 1)
   # Example: Draw a mesh with 10 vertices 20 times using instanced rendering
   ('DRAW', 2, 'mesh', (0, 10), 20)

Applies to: Program

This command is used to draw the program. It has a ``mode`` argument
which can be 'points', 'lines', 'line_strip', 'line_loop', 'lines_adjacency',
'line_strip_adjacency', 'triangles', 'triangle_strip', or 'triangle_fan'
(case insensitive).

If the ``selection`` argument has two elements, it contains two integers
``(start, count)``. If it has three elements, it contains
``(<index-buffer-id>, gtype, count)``, where ``gtype`` is
'unsigned_byte','unsigned_short', or 'unsigned_int'.

SIZE
~~~~

::

   ('SIZE', <id>, <size>, [<format>], [<internalformat>])
   # Example: resize a buffer
   ('SIZE', 4, 500)
   # Example: resize a 2D texture
   ('SIZE', 4, (500, 300, 3), 'rgb', None)
   ('SIZE', 4, (500, 300, 3), 'rgb', 'rgb16f')

Applies to: VertexBuffer, IndexBuffer, Texture2D, Texture3D,
RenderBuffer

This command is used to set the size of the buffer with the given id.
The GLIR implementation should be such that if the size/format
corresponds to the current size, it is ignored. The high level
implementation can use the SIZE command to discard previous DATA
commands.

For buffers: the size argument is an integer and the format argument is
not specified.

For textures and render buffer: the size argument is a shape tuple
(z,y,x). This tuple may contain the dimension for the color channels,
but this information is ignored. The format *should* be set to
'luminance', 'alpha', 'luminance_alpha', 'rgb' or 'rgba'. The
internalformat is a hint for backends that can control the internal GL
storage format; a value of None is a hint to use the default storage
format. The internalformat, if specified, *should* be a base channel
configuration of 'r', 'rg', 'rgb', or 'rgba' with a precision qualifying
suffix of '8', '16', '16f', or '32f'.

For render buffers: the size argument is a shape tuple (z,y,x). This
tuple may contain the dimension for the color channels, but this
information is ignored. The format *should* be set to 'color', 'depth'
or 'stencil'.

DATA
~~~~

::

   ('DATA', <id>, <offset>, <data:array>)
   # Example:
   ('DATA', 4, 100, <array 200x2>)

Applies to: VertexBuffer, IndexBuffer, Texture2D, Texture3D, VertexShader,
FragmentShader, GeometryShader

The data command is used to set the data of the object with the given
id. For VertexBuffer and IndexBuffer the offset is an integer. For
textures it is a tuple that matches with the dimension of the texture.
For shader objects it is always 0 and the data must be a ``str`` object.

WRAPPING
~~~~~~~~

::

   ('WRAPPING', <texture_id>, <wrapping:tuple>)
   # Example:
   ('WRAPPING', 4, ('CLAMP_TO_EDGE', 'CLAMP_TO_EDGE'))

Applies to: Texture2D, Texture3D

Set the wrapping mode for each dimension of the texture. Each element
must be a string: 'repeat', 'clamp_to_edge' or 'mirrored_repeat'.

INTERPOLATION
~~~~~~~~~~~~~

::

   ('INTERPOLATION', <texture_id>, <min:str>, <mag:str>)
   # Example:
   ('INTERPOLATION', 4, True, True)

Applies to: Texture2D, Texture3D

Set the interpolation mode of the texture for minification and
magnification. The min and mag argument can both be either 'nearest' or
'linear'.

ATTACH
~~~~~~

::

   ('ATTACH', <framebuffer_id>, <attachment:str>, <object>)
   ('ATTACH', <program_id>, <shader_id>)
   # Example:
   ('ATTACH', 4, 'color', 5)
   ('ATTACH', 1, 3)

Applies to: FrameBuffer, Program

Attach color, depth, or stencil buffer to the framebuffer. The
attachment argument can be 'color', 'depth' or 'stencil'. The object
argument must be the id for a RenderBuffer or Texture2D.
For Program this attaches an existing Shader object to the program.

FRAMEBUFFER
~~~~~~~~~~~

::

   ('FRAMEBUFFER', <framebuffer_id>, <use:bool>)
   # Example:
   ('FRAMEBUFFER', 4, True)

Applies to: FrameBuffer

Turn the framebuffer on or off. When deactivating a frame buffer, the
GLIR implementation should activate any previously activated
framebuffer.

FUNC
~~~~

::

   ('FUNC', <gl_function_name>, [arg1, [arg2, [arg3]]])

The ``FUNC`` command is a special command that can be applied to call a
variety of OpenGL calls. Use the documentation OpenGL for the required
arguments. Any args that are strings are converted to GL enums.

Supported functions are in principle all gl functions that do not have a
return value or covered by the above commands: glEnable, glDisable,
glClear, glClearColor, glClearDepth, glClearStencil, glViewport,
glDepthRange, glFrontFace, glCullFace, glPolygonOffset,
glBlendFuncSeparate, glBlendEquationSeparate, glBlendColor, glScissor,
glStencilFuncSeparate, glStencilMaskSeparate, glStencilOpSeparate,
glDepthFunc, glDepthMask, glColorMask, glSampleCoverage, glFlush,
glFinish, glHint.

SWAP
~~~~

::

    ('SWAP',)

The ``SWAP`` command is a special synchronization command for remote
rendering. This command tells the renderer that it should swap drawing
buffers. This is especially important when rendering with WebGL where
drawing buffers are implicitly swapped.

LINK
~~~~

::

    ('LINK', <program_id>)

Applies to: Program

Link the current program together (shaders, etc). Additionally this should
cause shaders to be detached and deleted. See the
`OpenGL documentation <https://www.khronos.org/registry/OpenGL-Refpages/gl4/html/glLinkProgram.xhtml>`_
for details on program linking.

"""
import os
import sys
import re
import json
import weakref
from packaging.version import Version
import numpy as np
from . import gl
from ..util import logger
_internalformats = [gl.Enum('GL_DEPTH_COMPONENT', 6402), gl.Enum('GL_DEPTH_COMPONENT16', 33189), gl.Enum('GL_DEPTH_COMPONENT32_OES', 33191), gl.Enum('GL_RED', 6403), gl.Enum('GL_R', 8194), gl.Enum('GL_R8', 33321), gl.Enum('GL_R16', 33322), gl.Enum('GL_R16F', 33325), gl.Enum('GL_R32F', 33326), gl.Enum('GL_RG', 33319), gl.Enum('GL_RG8', 333323), gl.Enum('GL_RG16', 333324), gl.Enum('GL_RG16F', 333327), gl.Enum('GL_RG32F', 33328), gl.Enum('GL_RGB', 6407), gl.Enum('GL_RGB8', 32849), gl.Enum('GL_RGB16', 32852), gl.Enum('GL_RGB16F', 34843), gl.Enum('GL_RGB32F', 34837), gl.Enum('GL_RGBA', 6408), gl.Enum('GL_RGBA8', 32856), gl.Enum('GL_RGBA16', 32859), gl.Enum('GL_RGBA16F', 34842), gl.Enum('GL_RGBA32F', 34836)]
_internalformats = dict([(enum.name, enum) for enum in _internalformats])
JUST_DELETED = 'JUST_DELETED'

def as_enum(enum):
    if False:
        while True:
            i = 10
    'Turn a possibly string enum into an integer enum.'
    if isinstance(enum, str):
        try:
            enum = getattr(gl, 'GL_' + enum.upper())
        except AttributeError:
            try:
                enum = _internalformats['GL_' + enum.upper()]
            except KeyError:
                raise ValueError('Could not find int value for enum %r' % enum)
    return enum

class _GlirQueueShare(object):
    """This class contains the actual queues of GLIR commands that are
    collected until a context becomes available to execute the commands.

    Instances of this class are further wrapped by GlirQueue to allow the
    underlying queues to be transparently merged when GL objects become
    associated.

    The motivation for this design is that it allows most glir commands to be
    added directly to their final queue (the same one used by the context),
    which reduces the effort required at draw time to determine the complete
    set of GL commands to be issued.

    At the same time, all GLObjects begin with their own local queue to allow
    commands to be queued at any time, even if the GLObject has
    not been associated yet. This works as expected even for complex topologies
    of GL objects, when some queues may only be joined at the last possible
    moment.
    """

    def __init__(self, queue):
        if False:
            while True:
                i = 10
        self._commands = []
        self._verbose = False
        self._associations = weakref.WeakKeyDictionary({queue: None})

    def command(self, *args):
        if False:
            return 10
        'Send a command. See the command spec at:\n        https://github.com/vispy/vispy/wiki/Spec.-Gloo-IR\n        '
        self._commands.append(args)

    def set_verbose(self, verbose):
        if False:
            i = 10
            return i + 15
        'Set verbose or not. If True, the GLIR commands are printed right before they get parsed.\n        If a string is given, use it as a filter.\n        '
        self._verbose = verbose

    def show(self, filter=None):
        if False:
            print('Hello World!')
        'Print the list of commands currently in the queue. If filter is\n        given, print only commands that match the filter.\n        '
        for command in self._commands:
            if command[0] is None:
                continue
            if filter and command[0] != filter:
                continue
            t = []
            for e in command:
                if isinstance(e, np.ndarray):
                    t.append('array %s' % str(e.shape))
                elif isinstance(e, str):
                    s = e.strip()
                    if len(s) > 20:
                        s = s[:18] + '... %i lines' % (e.count('\n') + 1)
                    t.append(s)
                else:
                    t.append(e)
            print(tuple(t))

    def clear(self):
        if False:
            while True:
                i = 10
        'Pop the whole queue (and associated queues) and return a\n        list of commands.\n        '
        commands = self._commands
        self._commands = []
        return commands

    def flush(self, parser):
        if False:
            print('Hello World!')
        'Flush all current commands to the GLIR interpreter.'
        if self._verbose:
            show = self._verbose if isinstance(self._verbose, str) else None
            self.show(show)
        parser.parse(self._filter(self.clear(), parser))

    def _filter(self, commands, parser):
        if False:
            print('Hello World!')
        'Filter DATA/SIZE commands that are overridden by a\n        SIZE command.\n        '
        resized = set()
        commands2 = []
        for command in reversed(commands):
            if command[1] in resized:
                if command[0] in ('SIZE', 'DATA'):
                    continue
            elif command[0] == 'SIZE':
                resized.add(command[1])
            commands2.append(command)
        return list(reversed(commands2))

class GlirQueue(object):
    """Representation of a queue of GLIR commands

    One instance of this class is attached to each context object, and
    to each gloo object. Internally, commands are stored in a shared queue
    object that may be swapped out and merged with other queues when
    ``associate()`` is called.

    Upon drawing (i.e. `Program.draw()`) and framebuffer switching, the
    commands in the queue are pushed to a parser, which is stored at
    context.shared. The parser can interpret the commands in Python,
    send them to a browser, etc.
    """

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        self._shared = _GlirQueueShare(self)

    def command(self, *args):
        if False:
            return 10
        'Send a command. See the command spec at:\n        https://github.com/vispy/vispy/wiki/Spec.-GLIR\n        '
        self._shared.command(*args)

    def set_verbose(self, verbose):
        if False:
            while True:
                i = 10
        'Set verbose or not. If True, the GLIR commands are printed\n        right before they get parsed. If a string is given, use it as\n        a filter.\n        '
        self._shared.set_verbose(verbose)

    def clear(self):
        if False:
            i = 10
            return i + 15
        'Pop the whole queue (and associated queues) and return a\n        list of commands.\n        '
        return self._shared.clear()

    def associate(self, queue):
        if False:
            return 10
        'Merge this queue with another.\n\n        Both queues will use a shared command list and either one can be used\n        to fill or flush the shared queue.\n        '
        assert isinstance(queue, GlirQueue)
        if queue._shared is self._shared:
            return
        self._shared._commands.extend(queue.clear())
        self._shared._verbose |= queue._shared._verbose
        self._shared._associations[queue] = None
        for ch in queue._shared._associations:
            ch._shared = self._shared
            self._shared._associations[ch] = None
        queue._shared = self._shared

    def flush(self, parser):
        if False:
            while True:
                i = 10
        'Flush all current commands to the GLIR interpreter.'
        self._shared.flush(parser)

def _convert_es2_shader(shader):
    if False:
        print('Hello World!')
    has_version = False
    has_prec_float = False
    has_prec_int = False
    lines = []
    extensions = []
    for line in shader.lstrip().splitlines():
        line_strip = line.lstrip()
        if line_strip.startswith('#version'):
            continue
        if line_strip.startswith('#extension'):
            extensions.append(line_strip)
            line = ''
        if line_strip.startswith('precision '):
            has_prec_float = has_prec_float or 'float' in line
            has_prec_int = has_prec_int or 'int' in line
        lines.append(line.rstrip())
    if not has_prec_float:
        lines.insert(has_version, 'precision highp float;')
    if not has_prec_int:
        lines.insert(has_version, 'precision highp int;')
    if extensions:
        for ext_line in extensions:
            lines.insert(has_version, ext_line)
    return '\n'.join(lines)

def _convert_desktop_shader(shader):
    if False:
        while True:
            i = 10
    has_version = False
    lines = []
    extensions = []
    for line in shader.lstrip().splitlines():
        line_strip = line.lstrip()
        has_version = has_version or line.startswith('#version')
        if line_strip.startswith('precision '):
            line = ''
        if line_strip.startswith('#extension'):
            extensions.append(line_strip)
            line = ''
        for prec in (' highp ', ' mediump ', ' lowp '):
            line = line.replace(prec, ' ')
        lines.append(line.rstrip())
    if extensions:
        for ext_line in extensions:
            lines.insert(has_version, ext_line)
    if not has_version:
        lines.insert(0, '#version 120\n')
    return '\n'.join(lines)

def convert_shader(backend_type, shader):
    if False:
        i = 10
        return i + 15
    'Modify shader code to be compatible with `backend_type` backend.'
    if backend_type == 'es2':
        return _convert_es2_shader(shader)
    elif backend_type == 'desktop':
        return _convert_desktop_shader(shader)
    else:
        raise ValueError('Cannot backend_type shaders to %r.' % backend_type)

def as_es2_command(command):
    if False:
        for i in range(10):
            print('nop')
    'Modify a desktop command so it works on es2.'
    if command[0] == 'FUNC':
        return (command[0], re.sub('^gl([A-Z])', lambda m: m.group(1).lower(), command[1])) + command[2:]
    elif command[0] == 'UNIFORM':
        return command[:-1] + (command[-1].tolist(),)
    return command

class BaseGlirParser(object):
    """Base class for GLIR parsers that can be attached to a GLIR queue."""

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        self.capabilities = dict(gl_version='Unknown', max_texture_size=None)

    def is_remote(self):
        if False:
            print('Hello World!')
        'Whether the code is executed remotely. i.e. gloo.gl cannot\n        be used.\n        '
        raise NotImplementedError()

    @property
    def shader_compatibility(self):
        if False:
            while True:
                i = 10
        "Whether to convert shading code. Valid values are 'es2' and\n        'desktop'. If None, the shaders are not modified.\n        "
        raise NotImplementedError()

    def parse(self, commands):
        if False:
            i = 10
            return i + 15
        'Parse the GLIR commands. Or sent them away.'
        raise NotImplementedError()

class GlirParser(BaseGlirParser):
    """A class for interpreting GLIR commands using gloo.gl

    We make use of relatively light GLIR objects that are instantiated
    on CREATE commands. These objects are stored by their id in a
    dictionary so that commands like ACTIVATE and DATA can easily
    be executed on the corresponding objects.
    """

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        super(GlirParser, self).__init__()
        self._objects = {}
        self._invalid_objects = set()
        self._classmap = {'VertexShader': GlirVertexShader, 'FragmentShader': GlirFragmentShader, 'GeometryShader': GlirGeometryShader, 'Program': GlirProgram, 'VertexBuffer': GlirVertexBuffer, 'IndexBuffer': GlirIndexBuffer, 'Texture1D': GlirTexture1D, 'Texture2D': GlirTexture2D, 'Texture3D': GlirTexture3D, 'TextureCube': GlirTextureCube, 'RenderBuffer': GlirRenderBuffer, 'FrameBuffer': GlirFrameBuffer}
        self.env = {}

    @property
    def shader_compatibility(self):
        if False:
            for i in range(10):
                print('nop')
        'Type of shader compatibility'
        if '.es' in gl.current_backend.__name__:
            return 'es2'
        else:
            return 'desktop'

    def is_remote(self):
        if False:
            print('Hello World!')
        return False

    def _parse(self, command):
        if False:
            print('Hello World!')
        'Parse a single command.'
        (cmd, id_, args) = (command[0], command[1], command[2:])
        if cmd == 'CURRENT':
            self.env.clear()
            self._gl_initialize()
            self.env['fbo'] = args[0]
            gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, args[0])
        elif cmd == 'FUNC':
            args = [as_enum(a) for a in args]
            try:
                getattr(gl, id_)(*args)
            except AttributeError:
                logger.warning('Invalid gl command: %r' % id_)
        elif cmd == 'CREATE':
            if args[0] is not None:
                klass = self._classmap[args[0]]
                self._objects[id_] = klass(self, id_)
            else:
                self._invalid_objects.add(id_)
        elif cmd == 'DELETE':
            ob = self._objects.get(id_, None)
            if ob is not None:
                self._objects[id_] = JUST_DELETED
                ob.delete()
        else:
            ob = self._objects.get(id_, None)
            if ob == JUST_DELETED:
                return
            if ob is None:
                if id_ not in self._invalid_objects:
                    raise RuntimeError('Cannot %s object %i because it does not exist' % (cmd, id_))
                return
            if cmd == 'DRAW':
                ob.draw(*args)
            elif cmd == 'TEXTURE':
                ob.set_texture(*args)
            elif cmd == 'UNIFORM':
                ob.set_uniform(*args)
            elif cmd == 'ATTRIBUTE':
                ob.set_attribute(*args)
            elif cmd == 'DATA':
                ob.set_data(*args)
            elif cmd == 'SIZE':
                ob.set_size(*args)
            elif cmd == 'ATTACH':
                ob.attach(*args)
            elif cmd == 'FRAMEBUFFER':
                ob.set_framebuffer(*args)
            elif cmd == 'LINK':
                ob.link_program(*args)
            elif cmd == 'WRAPPING':
                ob.set_wrapping(*args)
            elif cmd == 'INTERPOLATION':
                ob.set_interpolation(*args)
            else:
                logger.warning('Invalid GLIR command %r' % cmd)

    def parse(self, commands):
        if False:
            print('Hello World!')
        'Parse a list of commands.'
        to_delete = []
        for (id_, val) in self._objects.items():
            if val == JUST_DELETED:
                to_delete.append(id_)
        for id_ in to_delete:
            self._objects.pop(id_)
        for command in commands:
            self._parse(command)

    def get_object(self, id_):
        if False:
            print('Hello World!')
        'Get the object with the given id or None if it does not exist.'
        return self._objects.get(id_, None)

    def _gl_initialize(self):
        if False:
            for i in range(10):
                print('nop')
        'Deal with compatibility; desktop does not have sprites enabled by default. ES has.'
        if '.es' in gl.current_backend.__name__:
            pass
        else:
            GL_VERTEX_PROGRAM_POINT_SIZE = 34370
            GL_POINT_SPRITE = 34913
            gl.glEnable(GL_VERTEX_PROGRAM_POINT_SIZE)
            gl.glEnable(GL_POINT_SPRITE)
        if self.capabilities['max_texture_size'] is None:
            self.capabilities['gl_version'] = gl.glGetParameter(gl.GL_VERSION)
            self.capabilities['max_texture_size'] = gl.glGetParameter(gl.GL_MAX_TEXTURE_SIZE)
            this_version = self.capabilities['gl_version'].split(' ')
            if this_version[0] == 'OpenGL':
                this_version = this_version[2]
            else:
                this_version = this_version[0]
            if not this_version:
                logger.warning('OpenGL version could not be determined, which might be a sign that OpenGL is not loaded correctly.')
            elif Version(this_version) < Version('2.1'):
                if os.getenv('VISPY_IGNORE_OLD_VERSION', '').lower() != 'true':
                    logger.warning('OpenGL version 2.1 or higher recommended, got %s. Some functionality may fail.' % self.capabilities['gl_version'])

def glir_logger(parser_cls, file_or_filename):
    if False:
        return 10
    from ..util.logs import NumPyJSONEncoder

    class cls(parser_cls):

        def __init__(self, *args, **kwargs):
            if False:
                i = 10
                return i + 15
            parser_cls.__init__(self, *args, **kwargs)
            if isinstance(file_or_filename, str):
                self._file = open(file_or_filename, 'w')
            else:
                self._file = file_or_filename
            self._file.write('[]')
            self._empty = True

        def _parse(self, command):
            if False:
                i = 10
                return i + 15
            parser_cls._parse(self, command)
            self._file.seek(self._file.tell() - 1)
            if self._empty:
                self._empty = False
            else:
                self._file.write(',\n')
            json.dump(as_es2_command(command), self._file, cls=NumPyJSONEncoder)
            self._file.write(']')
    return cls

class GlirObject(object):

    def __init__(self, parser, id_):
        if False:
            print('Hello World!')
        self._parser = parser
        self._id = id_
        self._handle = -1
        self.create()

    @property
    def handle(self):
        if False:
            i = 10
            return i + 15
        return self._handle

    @property
    def id(self):
        if False:
            for i in range(10):
                print('nop')
        return self._id

    def __repr__(self):
        if False:
            return 10
        return '<%s %i at 0x%x>' % (self.__class__.__name__, self.id, id(self))

class GlirShader(GlirObject):
    _target = None

    def create(self):
        if False:
            i = 10
            return i + 15
        self._handle = gl.glCreateShader(self._target)

    def set_data(self, offset, code):
        if False:
            while True:
                i = 10
        convert = self._parser.shader_compatibility
        if convert:
            code = convert_shader(convert, code)
        gl.glShaderSource(self._handle, code)
        gl.glCompileShader(self._handle)
        status = gl.glGetShaderParameter(self._handle, gl.GL_COMPILE_STATUS)
        if not status:
            errors = gl.glGetShaderInfoLog(self._handle)
            errormsg = self._get_error(code, errors, 4)
            raise RuntimeError('Shader compilation error in %s:\n%s' % (self._target, errormsg))

    def delete(self):
        if False:
            while True:
                i = 10
        gl.glDeleteShader(self._handle)

    def _get_error(self, code, errors, indentation=0):
        if False:
            for i in range(10):
                print('nop')
        'Get error and show the faulty line + some context\n        Other GLIR implementations may omit this.\n        '
        results = []
        lines = None
        if code is not None:
            lines = [line.strip() for line in code.split('\n')]
        for error in errors.split('\n'):
            error = error.strip()
            if not error:
                continue
            (linenr, error) = self._parse_error(error)
            if None in (linenr, lines):
                results.append('%s' % error)
            else:
                results.append('on line %i: %s' % (linenr, error))
                if linenr > 0 and linenr < len(lines):
                    results.append('  %s' % lines[linenr - 1])
        results = [' ' * indentation + r for r in results]
        return '\n'.join(results)

    def _parse_error(self, error):
        if False:
            for i in range(10):
                print('nop')
        'Parses a single GLSL error and extracts the linenr and description\n        Other GLIR implementations may omit this.\n        '
        error = str(error)
        m = re.match('(\\d+)\\((\\d+)\\)\\s*:\\s(.*)', error)
        if m:
            return (int(m.group(2)), m.group(3))
        m = re.match('ERROR:\\s(\\d+):(\\d+):\\s(.*)', error)
        if m:
            return (int(m.group(2)), m.group(3))
        m = re.match('(\\d+):(\\d+)\\((\\d+)\\):\\s(.*)', error)
        if m:
            return (int(m.group(2)), m.group(4))
        return (None, error)

class GlirVertexShader(GlirShader):
    _target = gl.GL_VERTEX_SHADER

class GlirFragmentShader(GlirShader):
    _target = gl.GL_FRAGMENT_SHADER

class GlirGeometryShader(GlirShader):
    _target = None

    def __init__(self, *args, **kwargs):
        if False:
            print('Hello World!')
        if not hasattr(gl, 'GL_GEOMETRY_SHADER'):
            raise RuntimeError(gl.current_backend.__name__ + " backend does not support geometry shaders. Try gloo.gl.use_gl('gl+').")
        GlirGeometryShader._target = gl.GL_GEOMETRY_SHADER
        GlirShader.__init__(self, *args, **kwargs)

class GlirProgram(GlirObject):
    UTYPEMAP = {'float': 'glUniform1fv', 'vec2': 'glUniform2fv', 'vec3': 'glUniform3fv', 'vec4': 'glUniform4fv', 'int': 'glUniform1iv', 'ivec2': 'glUniform2iv', 'ivec3': 'glUniform3iv', 'ivec4': 'glUniform4iv', 'bool': 'glUniform1iv', 'bvec2': 'glUniform2iv', 'bvec3': 'glUniform3iv', 'bvec4': 'glUniform4iv', 'mat2': 'glUniformMatrix2fv', 'mat3': 'glUniformMatrix3fv', 'mat4': 'glUniformMatrix4fv', 'sampler1D': 'glUniform1i', 'sampler2D': 'glUniform1i', 'sampler3D': 'glUniform1i'}
    ATYPEMAP = {'float': 'glVertexAttrib1f', 'vec2': 'glVertexAttrib2f', 'vec3': 'glVertexAttrib3f', 'vec4': 'glVertexAttrib4f'}
    ATYPEINFO = {'float': (1, gl.GL_FLOAT, np.float32), 'vec2': (2, gl.GL_FLOAT, np.float32), 'vec3': (3, gl.GL_FLOAT, np.float32), 'vec4': (4, gl.GL_FLOAT, np.float32), 'ivec2': (2, gl.GL_INT, np.int32), 'ivec3': (3, gl.GL_INT, np.int32), 'ivec4': (4, gl.GL_INT, np.int32), 'int': (1, gl.GL_INT, np.int32), 'bool': (1, gl.GL_BOOL, np.int32)}

    def create(self):
        if False:
            return 10
        self._handle = gl.glCreateProgram()
        self._attached_shaders = []
        self._validated = False
        self._linked = False
        self._handles = {}
        self._unset_variables = set()
        self._samplers = {}
        self._attributes = {}
        self._known_invalid = set()

    def delete(self):
        if False:
            i = 10
            return i + 15
        gl.glDeleteProgram(self._handle)

    def activate(self):
        if False:
            i = 10
            return i + 15
        'Avoid overhead in calling glUseProgram with same arg.\n        Warning: this will break if glUseProgram is used somewhere else.\n        Per context we keep track of one current program.\n        '
        if self._handle != self._parser.env.get('current_program', False):
            self._parser.env['current_program'] = self._handle
            gl.glUseProgram(self._handle)

    def deactivate(self):
        if False:
            for i in range(10):
                print('nop')
        'Avoid overhead in calling glUseProgram with same arg.\n        Warning: this will break if glUseProgram is used somewhere else.\n        Per context we keep track of one current program.\n        '
        if self._parser.env.get('current_program', 0) != 0:
            self._parser.env['current_program'] = 0
            gl.glUseProgram(0)

    def set_shaders(self, vert, frag):
        if False:
            i = 10
            return i + 15
        'This function takes care of setting the shading code and\n        compiling+linking it into a working program object that is ready\n        to use.\n        '
        self._linked = False
        for (code, type_) in [(vert, 'vertex'), (frag, 'fragment')]:
            self.attach_shader(code, type_)
        self.link_program()

    def attach(self, id_):
        if False:
            print('Hello World!')
        'Attach a shader to this program.'
        shader = self._parser.get_object(id_)
        gl.glAttachShader(self._handle, shader.handle)
        self._attached_shaders.append(shader)

    def link_program(self):
        if False:
            i = 10
            return i + 15
        'Link the complete program and check.\n\n        All shaders are detached and deleted if the program was successfully\n        linked.\n        '
        gl.glLinkProgram(self._handle)
        if not gl.glGetProgramParameter(self._handle, gl.GL_LINK_STATUS):
            raise RuntimeError('Program linking error:\n%s' % gl.glGetProgramInfoLog(self._handle))
        for shader in self._attached_shaders:
            gl.glDetachShader(self._handle, shader.handle)
        self._attached_shaders = []
        self._unset_variables = self._get_active_attributes_and_uniforms()
        self._handles = {}
        self._known_invalid = set()
        self._linked = True

    def _get_active_attributes_and_uniforms(self):
        if False:
            print('Hello World!')
        'Retrieve active attributes and uniforms to be able to check that\n        all uniforms/attributes are set by the user.\n        Other GLIR implementations may omit this.\n        '
        regex = re.compile('(?P<name>\\w+)\\s*(\\[(?P<size>\\d+)\\])\\s*')
        cu = gl.glGetProgramParameter(self._handle, gl.GL_ACTIVE_UNIFORMS)
        ca = gl.glGetProgramParameter(self.handle, gl.GL_ACTIVE_ATTRIBUTES)
        attributes = []
        uniforms = []
        for (container, count, func) in [(attributes, ca, gl.glGetActiveAttrib), (uniforms, cu, gl.glGetActiveUniform)]:
            for i in range(count):
                (name, size, gtype) = func(self._handle, i)
                m = regex.match(name)
                if m:
                    name = m.group('name')
                    for i in range(size):
                        container.append(('%s[%d]' % (name, i), gtype))
                else:
                    container.append((name, gtype))
        return set([v[0] for v in attributes] + [v[0] for v in uniforms])

    def set_texture(self, name, value):
        if False:
            print('Hello World!')
        'Set a texture sampler. Value is the id of the texture to link.'
        if not self._linked:
            raise RuntimeError('Cannot set uniform when program has no code')
        handle = self._handles.get(name, -1)
        if handle < 0:
            if name in self._known_invalid:
                return
            handle = gl.glGetUniformLocation(self._handle, name)
            self._unset_variables.discard(name)
            self._handles[name] = handle
            if handle < 0:
                self._known_invalid.add(name)
                logger.info('Not setting texture data for variable %s; uniform is not active.' % name)
                return
        self.activate()
        if True:
            tex = self._parser.get_object(value)
            if tex == JUST_DELETED:
                return
            if tex is None:
                raise RuntimeError('Could not find texture with id %i' % value)
            unit = len(self._samplers)
            if name in self._samplers:
                unit = self._samplers[name][-1]
            self._samplers[name] = (tex._target, tex.handle, unit)
            gl.glUniform1i(handle, unit)

    def set_uniform(self, name, type_, value):
        if False:
            print('Hello World!')
        'Set a uniform value. Value is assumed to have been checked.'
        if not self._linked:
            raise RuntimeError('Cannot set uniform when program has no code')
        handle = self._handles.get(name, -1)
        count = 1
        if handle < 0:
            if name in self._known_invalid:
                return
            handle = gl.glGetUniformLocation(self._handle, name)
            self._unset_variables.discard(name)
            if not type_.startswith('mat'):
                count = value.nbytes // (4 * self.ATYPEINFO[type_][0])
            if count > 1:
                for ii in range(count):
                    if '%s[%s]' % (name, ii) in self._unset_variables:
                        self._unset_variables.discard('%s[%s]' % (name, ii))
            self._handles[name] = handle
            if handle < 0:
                self._known_invalid.add(name)
                logger.info('Not setting value for variable %s %s; uniform is not active.' % (type_, name))
                return
        funcname = self.UTYPEMAP[type_]
        func = getattr(gl, funcname)
        self.activate()
        if type_.startswith('mat'):
            transpose = False
            func(handle, 1, transpose, value)
        else:
            func(handle, count, value)

    def set_attribute(self, name, type_, value, divisor=None):
        if False:
            print('Hello World!')
        'Set an attribute value. Value is assumed to have been checked.'
        if not self._linked:
            raise RuntimeError('Cannot set attribute when program has no code')
        handle = self._handles.get(name, -1)
        if handle < 0:
            if name in self._known_invalid:
                return
            handle = gl.glGetAttribLocation(self._handle, name)
            self._unset_variables.discard(name)
            self._handles[name] = handle
            if handle < 0:
                self._known_invalid.add(name)
                if value[0] != 0 and value[2] > 0:
                    return
                logger.info('Not setting data for variable %s %s; attribute is not active.' % (type_, name))
                return
        self.activate()
        if value[0] == 0:
            funcname = self.ATYPEMAP[type_]
            func = getattr(gl, funcname)
            self._attributes[name] = (0, handle, func, value[1:], divisor)
        else:
            (vbo_id, stride, offset) = value
            (size, gtype, dtype) = self.ATYPEINFO[type_]
            vbo = self._parser.get_object(vbo_id)
            if vbo == JUST_DELETED:
                return
            if vbo is None:
                raise RuntimeError('Could not find VBO with id %i' % vbo_id)
            func = gl.glVertexAttribPointer
            args = (size, gtype, gl.GL_FALSE, stride, offset)
            self._attributes[name] = (vbo.handle, handle, func, args, divisor)

    def _pre_draw(self):
        if False:
            i = 10
            return i + 15
        self.activate()
        for (tex_target, tex_handle, unit) in self._samplers.values():
            gl.glActiveTexture(gl.GL_TEXTURE0 + unit)
            gl.glBindTexture(tex_target, tex_handle)
        for (vbo_handle, attr_handle, func, args, divisor) in self._attributes.values():
            if vbo_handle:
                gl.glBindBuffer(gl.GL_ARRAY_BUFFER, vbo_handle)
                gl.glEnableVertexAttribArray(attr_handle)
                func(attr_handle, *args)
                if divisor is not None:
                    gl.glVertexAttribDivisor(attr_handle, divisor)
            else:
                gl.glBindBuffer(gl.GL_ARRAY_BUFFER, 0)
                gl.glDisableVertexAttribArray(attr_handle)
                func(attr_handle, *args)
        if not self._validated:
            self._validated = True
            self._validate()

    def _validate(self):
        if False:
            for i in range(10):
                print('nop')
        if self._unset_variables:
            logger.warning('Program has unset variables: %r' % self._unset_variables)
        gl.glValidateProgram(self._handle)
        if not gl.glGetProgramParameter(self._handle, gl.GL_VALIDATE_STATUS):
            raise RuntimeError('Program validation error:\n%s' % gl.glGetProgramInfoLog(self._handle))

    def _post_draw(self):
        if False:
            while True:
                i = 10
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, 0)
        gl.glBindTexture(gl.GL_TEXTURE_2D, 0)
        if USE_TEX_3D:
            gl.glBindTexture(GL_TEXTURE_3D, 0)
            gl.glBindTexture(GL_TEXTURE_1D, 0)

    def draw(self, mode, selection, instances=1):
        if False:
            while True:
                i = 10
        'Draw program in given mode, with given selection (IndexBuffer or\n        first, count).\n        '
        if not self._linked:
            raise RuntimeError('Cannot draw program if code has not been set')
        gl.check_error('Check before draw')
        try:
            mode = as_enum(mode)
        except ValueError:
            if mode == 'lines_adjacency' or mode == 'line_strip_adjacency':
                raise RuntimeError(gl.current_backend.__name__ + " backend does not support lines_adjacency and line_strip_adjacency primitives. Try gloo.gl.use_gl('gl+').")
            raise
        if len(selection) == 3:
            (id_, gtype, count) = selection
            if count:
                self._pre_draw()
                ibuf = self._parser.get_object(id_)
                ibuf.activate()
                if instances > 1:
                    gl.glDrawElementsInstanced(mode, count, as_enum(gtype), None, instances)
                else:
                    gl.glDrawElements(mode, count, as_enum(gtype), None)
                ibuf.deactivate()
        else:
            (first, count) = selection
            if count:
                self._pre_draw()
                if instances > 1:
                    gl.glDrawArraysInstanced(mode, first, count, instances)
                else:
                    gl.glDrawArrays(mode, first, count)
        gl.check_error('Check after draw')
        self._post_draw()

class GlirBuffer(GlirObject):
    _target = None
    _usage = gl.GL_DYNAMIC_DRAW

    def create(self):
        if False:
            print('Hello World!')
        self._handle = gl.glCreateBuffer()
        self._buffer_size = 0
        self._bufferSubDataOk = False

    def delete(self):
        if False:
            for i in range(10):
                print('nop')
        gl.glDeleteBuffer(self._handle)

    def activate(self):
        if False:
            for i in range(10):
                print('nop')
        gl.glBindBuffer(self._target, self._handle)

    def deactivate(self):
        if False:
            for i in range(10):
                print('nop')
        gl.glBindBuffer(self._target, 0)

    def set_size(self, nbytes):
        if False:
            print('Hello World!')
        if nbytes != self._buffer_size:
            self.activate()
            gl.glBufferData(self._target, nbytes, self._usage)
            self._buffer_size = nbytes

    def set_data(self, offset, data):
        if False:
            for i in range(10):
                print('nop')
        self.activate()
        nbytes = data.nbytes
        check_ati_bug = not self._bufferSubDataOk and gl.current_backend.__name__.split('.')[-1] == 'gl2' and sys.platform.startswith('win')
        if check_ati_bug:
            gl.check_error('periodic check')
        try:
            gl.glBufferSubData(self._target, offset, data)
            if check_ati_bug:
                gl.check_error('glBufferSubData')
            self._bufferSubDataOk = True
        except Exception:
            if offset == 0 and nbytes == self._buffer_size:
                gl.glBufferData(self._target, data, self._usage)
                logger.debug('Using glBufferData instead of ' + 'glBufferSubData (known ATI bug).')
            else:
                raise

class GlirVertexBuffer(GlirBuffer):
    _target = gl.GL_ARRAY_BUFFER

class GlirIndexBuffer(GlirBuffer):
    _target = gl.GL_ELEMENT_ARRAY_BUFFER

class GlirTexture(GlirObject):
    _target = None
    _types = {np.dtype(np.int8): gl.GL_BYTE, np.dtype(np.uint8): gl.GL_UNSIGNED_BYTE, np.dtype(np.int16): gl.GL_SHORT, np.dtype(np.uint16): gl.GL_UNSIGNED_SHORT, np.dtype(np.int32): gl.GL_INT, np.dtype(np.uint32): gl.GL_UNSIGNED_INT, np.dtype(np.float32): gl.GL_FLOAT}

    def create(self):
        if False:
            for i in range(10):
                print('nop')
        self._handle = gl.glCreateTexture()
        self._shape_formats = 0

    def delete(self):
        if False:
            for i in range(10):
                print('nop')
        gl.glDeleteTexture(self._handle)

    def activate(self):
        if False:
            while True:
                i = 10
        gl.glBindTexture(self._target, self._handle)

    def deactivate(self):
        if False:
            i = 10
            return i + 15
        gl.glBindTexture(self._target, 0)

    def _get_alignment(self, width):
        if False:
            i = 10
            return i + 15
        "Determines a textures byte alignment.\n\n        If the width isn't a power of 2\n        we need to adjust the byte alignment of the image.\n        The image height is unimportant\n\n        www.opengl.org/wiki/Common_Mistakes#Texture_upload_and_pixel_reads\n        "
        alignments = [8, 4, 2, 1]
        for alignment in alignments:
            if width % alignment == 0:
                return alignment

    def set_wrapping(self, wrapping):
        if False:
            print('Hello World!')
        self.activate()
        wrapping = [as_enum(w) for w in wrapping]
        if len(wrapping) == 3:
            GL_TEXTURE_WRAP_R = 32882
            gl.glTexParameterf(self._target, GL_TEXTURE_WRAP_R, wrapping[0])
        if len(wrapping) >= 2:
            gl.glTexParameterf(self._target, gl.GL_TEXTURE_WRAP_S, wrapping[-2])
        gl.glTexParameterf(self._target, gl.GL_TEXTURE_WRAP_T, wrapping[-1])

    def set_interpolation(self, min, mag):
        if False:
            while True:
                i = 10
        self.activate()
        (min, mag) = (as_enum(min), as_enum(mag))
        gl.glTexParameterf(self._target, gl.GL_TEXTURE_MIN_FILTER, min)
        gl.glTexParameterf(self._target, gl.GL_TEXTURE_MAG_FILTER, mag)
GL_SAMPLER_1D = gl.Enum('GL_SAMPLER_1D', 35677)
GL_TEXTURE_1D = gl.Enum('GL_TEXTURE_1D', 3552)

class GlirTexture1D(GlirTexture):
    _target = GL_TEXTURE_1D

    def set_size(self, shape, format, internalformat):
        if False:
            for i in range(10):
                print('nop')
        format = as_enum(format)
        if internalformat is not None:
            internalformat = as_enum(internalformat)
        else:
            internalformat = format
        if (shape, format, internalformat) != self._shape_formats:
            self.activate()
            self._shape_formats = (shape, format, internalformat)
            glTexImage1D(self._target, 0, internalformat, format, gl.GL_BYTE, shape[:1])

    def set_data(self, offset, data):
        if False:
            print('Hello World!')
        self.activate()
        (shape, format, internalformat) = self._shape_formats
        x = offset[0]
        gtype = self._types.get(np.dtype(data.dtype), None)
        if gtype is None:
            raise ValueError('Type %r not allowed for texture' % data.dtype)
        alignment = self._get_alignment(data.shape[-1] * data.itemsize)
        if alignment != 4:
            gl.glPixelStorei(gl.GL_UNPACK_ALIGNMENT, alignment)
        glTexSubImage1D(self._target, 0, x, format, gtype, data)
        if alignment != 4:
            gl.glPixelStorei(gl.GL_UNPACK_ALIGNMENT, 4)

class GlirTexture2D(GlirTexture):
    _target = gl.GL_TEXTURE_2D

    def set_size(self, shape, format, internalformat):
        if False:
            i = 10
            return i + 15
        format = as_enum(format)
        internalformat = format if internalformat is None else as_enum(internalformat)
        if (shape, format, internalformat) != self._shape_formats:
            self._shape_formats = (shape, format, internalformat)
            self.activate()
            gl.glTexImage2D(self._target, 0, internalformat, format, gl.GL_UNSIGNED_BYTE, shape[:2])

    def set_data(self, offset, data):
        if False:
            print('Hello World!')
        self.activate()
        (shape, format, internalformat) = self._shape_formats
        (y, x) = offset
        gtype = self._types.get(np.dtype(data.dtype), None)
        if gtype is None:
            raise ValueError('Type %r not allowed for texture' % data.dtype)
        alignment = self._get_alignment(data.shape[-2] * data.shape[-1] * data.itemsize)
        if alignment != 4:
            gl.glPixelStorei(gl.GL_UNPACK_ALIGNMENT, alignment)
        gl.glTexSubImage2D(self._target, 0, x, y, format, gtype, data)
        if alignment != 4:
            gl.glPixelStorei(gl.GL_UNPACK_ALIGNMENT, 4)
GL_SAMPLER_3D = gl.Enum('GL_SAMPLER_3D', 35679)
GL_TEXTURE_3D = gl.Enum('GL_TEXTURE_3D', 32879)
USE_TEX_3D = False

def _check_pyopengl_3D():
    if False:
        while True:
            i = 10
    'Helper to ensure users have OpenGL for 3D texture support (for now)'
    global USE_TEX_3D
    USE_TEX_3D = True
    try:
        import OpenGL.GL as _gl
    except ImportError:
        raise ImportError('PyOpenGL is required for 3D texture support')
    return _gl

def glTexImage3D(target, level, internalformat, format, type, pixels):
    if False:
        while True:
            i = 10
    _gl = _check_pyopengl_3D()
    border = 0
    assert isinstance(pixels, (tuple, list))
    (depth, height, width) = pixels
    _gl.glTexImage3D(target, level, internalformat, width, height, depth, border, format, type, None)

def glTexImage1D(target, level, internalformat, format, type, pixels):
    if False:
        print('Hello World!')
    _gl = _check_pyopengl_3D()
    border = 0
    assert isinstance(pixels, (tuple, list))
    width = pixels[0]
    _gl.glTexImage1D(target, level, internalformat, width, border, format, type, None)

def glTexSubImage1D(target, level, xoffset, format, type, pixels):
    if False:
        for i in range(10):
            print('nop')
    _gl = _check_pyopengl_3D()
    width = pixels.shape[:1]
    _gl.glTexSubImage1D(target, level, xoffset, width[0], format, type, pixels)

def glTexSubImage3D(target, level, xoffset, yoffset, zoffset, format, type, pixels):
    if False:
        for i in range(10):
            print('nop')
    _gl = _check_pyopengl_3D()
    (depth, height, width) = pixels.shape[:3]
    _gl.glTexSubImage3D(target, level, xoffset, yoffset, zoffset, width, height, depth, format, type, pixels)

class GlirTexture3D(GlirTexture):
    _target = GL_TEXTURE_3D

    def set_size(self, shape, format, internalformat):
        if False:
            print('Hello World!')
        format = as_enum(format)
        if internalformat is not None:
            internalformat = as_enum(internalformat)
        else:
            internalformat = format
        if (shape, format, internalformat) != self._shape_formats:
            self.activate()
            self._shape_formats = (shape, format, internalformat)
            glTexImage3D(self._target, 0, internalformat, format, gl.GL_BYTE, shape[:3])

    def set_data(self, offset, data):
        if False:
            print('Hello World!')
        self.activate()
        (shape, format, internalformat) = self._shape_formats
        (z, y, x) = offset
        gtype = self._types.get(np.dtype(data.dtype), None)
        if gtype is None:
            raise ValueError('Type not allowed for texture')
        alignment = self._get_alignment(data.shape[-2] * data.shape[-1] * data.itemsize)
        if alignment != 4:
            gl.glPixelStorei(gl.GL_UNPACK_ALIGNMENT, alignment)
        glTexSubImage3D(self._target, 0, x, y, z, format, gtype, data)
        if alignment != 4:
            gl.glPixelStorei(gl.GL_UNPACK_ALIGNMENT, 4)

class GlirTextureCube(GlirTexture):
    _target = gl.GL_TEXTURE_CUBE_MAP
    _cube_targets = [gl.GL_TEXTURE_CUBE_MAP_POSITIVE_X, gl.GL_TEXTURE_CUBE_MAP_NEGATIVE_X, gl.GL_TEXTURE_CUBE_MAP_POSITIVE_Y, gl.GL_TEXTURE_CUBE_MAP_NEGATIVE_Y, gl.GL_TEXTURE_CUBE_MAP_POSITIVE_Z, gl.GL_TEXTURE_CUBE_MAP_NEGATIVE_Z]

    def set_size(self, shape, format, internalformat):
        if False:
            print('Hello World!')
        format = as_enum(format)
        internalformat = format if internalformat is None else as_enum(internalformat)
        if (shape, format, internalformat) != self._shape_formats:
            self._shape_formats = (shape, format, internalformat)
            self.activate()
            for target in self._cube_targets:
                gl.glTexImage2D(target, 0, internalformat, format, gl.GL_UNSIGNED_BYTE, shape[1:3])

    def set_data(self, offset, data):
        if False:
            for i in range(10):
                print('nop')
        (shape, format, internalformat) = self._shape_formats
        (y, x) = offset[:2]
        gtype = self._types.get(np.dtype(data.dtype), None)
        if gtype is None:
            raise ValueError('Type %r not allowed for texture' % data.dtype)
        self.activate()
        alignment = self._get_alignment(data.shape[-2] * data.shape[-1] * data.itemsize)
        if alignment != 4:
            gl.glPixelStorei(gl.GL_UNPACK_ALIGNMENT, alignment)
        for (i, target) in enumerate(self._cube_targets):
            gl.glTexSubImage2D(target, 0, x, y, format, gtype, data[i])
        if alignment != 4:
            gl.glPixelStorei(gl.GL_UNPACK_ALIGNMENT, 4)

class GlirRenderBuffer(GlirObject):

    def create(self):
        if False:
            for i in range(10):
                print('nop')
        self._handle = gl.glCreateRenderbuffer()
        self._shape_format = 0

    def delete(self):
        if False:
            i = 10
            return i + 15
        gl.glDeleteRenderbuffer(self._handle)

    def activate(self):
        if False:
            return 10
        gl.glBindRenderbuffer(gl.GL_RENDERBUFFER, self._handle)

    def deactivate(self):
        if False:
            return 10
        gl.glBindRenderbuffer(gl.GL_RENDERBUFFER, 0)

    def set_size(self, shape, format):
        if False:
            while True:
                i = 10
        if isinstance(format, str):
            format = GlirFrameBuffer._formats[format][1]
        if (shape, format) != self._shape_format:
            self._shape_format = (shape, format)
            self.activate()
            gl.glRenderbufferStorage(gl.GL_RENDERBUFFER, format, shape[1], shape[0])

class GlirFrameBuffer(GlirObject):
    _formats = {'color': (gl.GL_COLOR_ATTACHMENT0, gl.GL_RGBA), 'depth': (gl.GL_DEPTH_ATTACHMENT, gl.GL_DEPTH_COMPONENT16), 'stencil': (gl.GL_STENCIL_ATTACHMENT, gl.GL_STENCIL_INDEX8)}

    def create(self):
        if False:
            while True:
                i = 10
        self._handle = gl.glCreateFramebuffer()
        self._validated = False

    def delete(self):
        if False:
            return 10
        gl.glDeleteFramebuffer(self._handle)

    def set_framebuffer(self, yes):
        if False:
            while True:
                i = 10
        if yes:
            self.activate()
            if not self._validated:
                self._validated = True
                self._validate()
        else:
            self.deactivate()

    def activate(self):
        if False:
            for i in range(10):
                print('nop')
        stack = self._parser.env.setdefault('fb_stack', [self._parser.env['fbo']])
        if stack[-1] != self._handle:
            stack.append(self._handle)
            gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, self._handle)

    def deactivate(self):
        if False:
            for i in range(10):
                print('nop')
        stack = self._parser.env.setdefault('fb_stack', [self._parser.env['fbo']])
        while self._handle in stack:
            stack.remove(self._handle)
        gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, stack[-1])

    def attach(self, attachment, buffer_id):
        if False:
            i = 10
            return i + 15
        attachment = GlirFrameBuffer._formats[attachment][0]
        self.activate()
        if buffer_id == 0:
            gl.glFramebufferRenderbuffer(gl.GL_FRAMEBUFFER, attachment, gl.GL_RENDERBUFFER, 0)
        else:
            buffer = self._parser.get_object(buffer_id)
            if buffer == JUST_DELETED:
                return
            if buffer is None:
                raise ValueError('Unknown buffer with id %i for attachement' % buffer_id)
            elif isinstance(buffer, GlirRenderBuffer):
                buffer.activate()
                gl.glFramebufferRenderbuffer(gl.GL_FRAMEBUFFER, attachment, gl.GL_RENDERBUFFER, buffer.handle)
                buffer.deactivate()
            elif isinstance(buffer, GlirTexture2D):
                buffer.activate()
                gl.glFramebufferTexture2D(gl.GL_FRAMEBUFFER, attachment, gl.GL_TEXTURE_2D, buffer.handle, 0)
                buffer.deactivate()
            else:
                raise ValueError('Invalid attachment: %s' % type(buffer))
        self._validated = False
        self.deactivate()

    def _validate(self):
        if False:
            return 10
        res = gl.glCheckFramebufferStatus(gl.GL_FRAMEBUFFER)
        if res == gl.GL_FRAMEBUFFER_COMPLETE:
            return
        _bad_map = {0: 'Target not equal to GL_FRAMEBUFFER', gl.GL_FRAMEBUFFER_INCOMPLETE_ATTACHMENT: 'FrameBuffer attachments are incomplete.', gl.GL_FRAMEBUFFER_INCOMPLETE_MISSING_ATTACHMENT: 'No valid attachments in the FrameBuffer.', gl.GL_FRAMEBUFFER_INCOMPLETE_DIMENSIONS: 'attachments do not have the same width and height.', gl.GL_FRAMEBUFFER_UNSUPPORTED: 'Combination of internal formats used by attachments is not supported.'}
        raise RuntimeError(_bad_map.get(res, 'Unknown framebuffer error: %r.' % res))