"""GL definitions converted to Python by codegen/createglapi.py.

THIS CODE IS AUTO-GENERATED. DO NOT EDIT.

GL ES 2.0 API (via Angle/DirectX on Windows)

"""
import ctypes
from .es2 import _lib
_lib.glActiveTexture.argtypes = (ctypes.c_uint,)

def glActiveTexture(texture):
    if False:
        return 10
    _lib.glActiveTexture(texture)
_lib.glAttachShader.argtypes = (ctypes.c_uint, ctypes.c_uint)

def glAttachShader(program, shader):
    if False:
        return 10
    _lib.glAttachShader(program, shader)
_lib.glBindAttribLocation.argtypes = (ctypes.c_uint, ctypes.c_uint, ctypes.c_char_p)

def glBindAttribLocation(program, index, name):
    if False:
        return 10
    name = ctypes.c_char_p(name.encode('utf-8'))
    res = _lib.glBindAttribLocation(program, index, name)
_lib.glBindBuffer.argtypes = (ctypes.c_uint, ctypes.c_uint)

def glBindBuffer(target, buffer):
    if False:
        return 10
    _lib.glBindBuffer(target, buffer)
_lib.glBindFramebuffer.argtypes = (ctypes.c_uint, ctypes.c_uint)

def glBindFramebuffer(target, framebuffer):
    if False:
        for i in range(10):
            print('nop')
    _lib.glBindFramebuffer(target, framebuffer)
_lib.glBindRenderbuffer.argtypes = (ctypes.c_uint, ctypes.c_uint)

def glBindRenderbuffer(target, renderbuffer):
    if False:
        i = 10
        return i + 15
    _lib.glBindRenderbuffer(target, renderbuffer)
_lib.glBindTexture.argtypes = (ctypes.c_uint, ctypes.c_uint)

def glBindTexture(target, texture):
    if False:
        while True:
            i = 10
    _lib.glBindTexture(target, texture)
_lib.glBlendColor.argtypes = (ctypes.c_float, ctypes.c_float, ctypes.c_float, ctypes.c_float)

def glBlendColor(red, green, blue, alpha):
    if False:
        print('Hello World!')
    _lib.glBlendColor(red, green, blue, alpha)
_lib.glBlendEquation.argtypes = (ctypes.c_uint,)

def glBlendEquation(mode):
    if False:
        while True:
            i = 10
    _lib.glBlendEquation(mode)
_lib.glBlendEquationSeparate.argtypes = (ctypes.c_uint, ctypes.c_uint)

def glBlendEquationSeparate(modeRGB, modeAlpha):
    if False:
        for i in range(10):
            print('nop')
    _lib.glBlendEquationSeparate(modeRGB, modeAlpha)
_lib.glBlendFunc.argtypes = (ctypes.c_uint, ctypes.c_uint)

def glBlendFunc(sfactor, dfactor):
    if False:
        print('Hello World!')
    _lib.glBlendFunc(sfactor, dfactor)
_lib.glBlendFuncSeparate.argtypes = (ctypes.c_uint, ctypes.c_uint, ctypes.c_uint, ctypes.c_uint)

def glBlendFuncSeparate(srcRGB, dstRGB, srcAlpha, dstAlpha):
    if False:
        print('Hello World!')
    _lib.glBlendFuncSeparate(srcRGB, dstRGB, srcAlpha, dstAlpha)
_lib.glBufferData.argtypes = (ctypes.c_uint, ctypes.c_ssize_t, ctypes.c_void_p, ctypes.c_uint)

def glBufferData(target, data, usage):
    if False:
        while True:
            i = 10
    'Data can be numpy array or the size of data to allocate.'
    if isinstance(data, int):
        size = data
        data = ctypes.c_voidp(0)
    else:
        if not data.flags['C_CONTIGUOUS'] or not data.flags['ALIGNED']:
            data = data.copy('C')
        data_ = data
        size = data_.nbytes
        data = data_.ctypes.data
    res = _lib.glBufferData(target, size, data, usage)
_lib.glBufferSubData.argtypes = (ctypes.c_uint, ctypes.c_ssize_t, ctypes.c_ssize_t, ctypes.c_void_p)

def glBufferSubData(target, offset, data):
    if False:
        for i in range(10):
            print('nop')
    if not data.flags['C_CONTIGUOUS']:
        data = data.copy('C')
    data_ = data
    size = data_.nbytes
    data = data_.ctypes.data
    res = _lib.glBufferSubData(target, offset, size, data)
_lib.glCheckFramebufferStatus.argtypes = (ctypes.c_uint,)
_lib.glCheckFramebufferStatus.restype = ctypes.c_uint

def glCheckFramebufferStatus(target):
    if False:
        print('Hello World!')
    return _lib.glCheckFramebufferStatus(target)
_lib.glClear.argtypes = (ctypes.c_uint,)

def glClear(mask):
    if False:
        return 10
    _lib.glClear(mask)
_lib.glClearColor.argtypes = (ctypes.c_float, ctypes.c_float, ctypes.c_float, ctypes.c_float)

def glClearColor(red, green, blue, alpha):
    if False:
        return 10
    _lib.glClearColor(red, green, blue, alpha)
_lib.glClearDepthf.argtypes = (ctypes.c_float,)

def glClearDepth(depth):
    if False:
        while True:
            i = 10
    _lib.glClearDepthf(depth)
_lib.glClearStencil.argtypes = (ctypes.c_int,)

def glClearStencil(s):
    if False:
        for i in range(10):
            print('nop')
    _lib.glClearStencil(s)
_lib.glColorMask.argtypes = (ctypes.c_bool, ctypes.c_bool, ctypes.c_bool, ctypes.c_bool)

def glColorMask(red, green, blue, alpha):
    if False:
        return 10
    _lib.glColorMask(red, green, blue, alpha)
_lib.glCompileShader.argtypes = (ctypes.c_uint,)

def glCompileShader(shader):
    if False:
        print('Hello World!')
    _lib.glCompileShader(shader)
_lib.glCompressedTexImage2D.argtypes = (ctypes.c_uint, ctypes.c_int, ctypes.c_uint, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_void_p)

def glCompressedTexImage2D(target, level, internalformat, width, height, border, data):
    if False:
        while True:
            i = 10
    if not data.flags['C_CONTIGUOUS']:
        data = data.copy('C')
    data_ = data
    size = data_.size
    data = data_.ctypes.data
    res = _lib.glCompressedTexImage2D(target, level, internalformat, width, height, border, imageSize, data)
_lib.glCompressedTexSubImage2D.argtypes = (ctypes.c_uint, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_uint, ctypes.c_int, ctypes.c_void_p)

def glCompressedTexSubImage2D(target, level, xoffset, yoffset, width, height, format, data):
    if False:
        for i in range(10):
            print('nop')
    if not data.flags['C_CONTIGUOUS']:
        data = data.copy('C')
    data_ = data
    size = data_.size
    data = data_.ctypes.data
    res = _lib.glCompressedTexSubImage2D(target, level, xoffset, yoffset, width, height, format, imageSize, data)
_lib.glCopyTexImage2D.argtypes = (ctypes.c_uint, ctypes.c_int, ctypes.c_uint, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int)

def glCopyTexImage2D(target, level, internalformat, x, y, width, height, border):
    if False:
        for i in range(10):
            print('nop')
    _lib.glCopyTexImage2D(target, level, internalformat, x, y, width, height, border)
_lib.glCopyTexSubImage2D.argtypes = (ctypes.c_uint, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int)

def glCopyTexSubImage2D(target, level, xoffset, yoffset, x, y, width, height):
    if False:
        i = 10
        return i + 15
    _lib.glCopyTexSubImage2D(target, level, xoffset, yoffset, x, y, width, height)
_lib.glCreateProgram.argtypes = ()
_lib.glCreateProgram.restype = ctypes.c_uint

def glCreateProgram():
    if False:
        print('Hello World!')
    return _lib.glCreateProgram()
_lib.glCreateShader.argtypes = (ctypes.c_uint,)
_lib.glCreateShader.restype = ctypes.c_uint

def glCreateShader(type):
    if False:
        for i in range(10):
            print('nop')
    return _lib.glCreateShader(type)
_lib.glCullFace.argtypes = (ctypes.c_uint,)

def glCullFace(mode):
    if False:
        i = 10
        return i + 15
    _lib.glCullFace(mode)
_lib.glDeleteBuffers.argtypes = (ctypes.c_int, ctypes.POINTER(ctypes.c_uint))

def glDeleteBuffer(buffer):
    if False:
        i = 10
        return i + 15
    n = 1
    buffers = (ctypes.c_uint * n)(buffer)
    res = _lib.glDeleteBuffers(n, buffers)
_lib.glDeleteFramebuffers.argtypes = (ctypes.c_int, ctypes.POINTER(ctypes.c_uint))

def glDeleteFramebuffer(framebuffer):
    if False:
        while True:
            i = 10
    n = 1
    framebuffers = (ctypes.c_uint * n)(framebuffer)
    res = _lib.glDeleteFramebuffers(n, framebuffers)
_lib.glDeleteProgram.argtypes = (ctypes.c_uint,)

def glDeleteProgram(program):
    if False:
        i = 10
        return i + 15
    _lib.glDeleteProgram(program)
_lib.glDeleteRenderbuffers.argtypes = (ctypes.c_int, ctypes.POINTER(ctypes.c_uint))

def glDeleteRenderbuffer(renderbuffer):
    if False:
        i = 10
        return i + 15
    n = 1
    renderbuffers = (ctypes.c_uint * n)(renderbuffer)
    res = _lib.glDeleteRenderbuffers(n, renderbuffers)
_lib.glDeleteShader.argtypes = (ctypes.c_uint,)

def glDeleteShader(shader):
    if False:
        i = 10
        return i + 15
    _lib.glDeleteShader(shader)
_lib.glDeleteTextures.argtypes = (ctypes.c_int, ctypes.POINTER(ctypes.c_uint))

def glDeleteTexture(texture):
    if False:
        print('Hello World!')
    n = 1
    textures = (ctypes.c_uint * n)(texture)
    res = _lib.glDeleteTextures(n, textures)
_lib.glDepthFunc.argtypes = (ctypes.c_uint,)

def glDepthFunc(func):
    if False:
        while True:
            i = 10
    _lib.glDepthFunc(func)
_lib.glDepthMask.argtypes = (ctypes.c_bool,)

def glDepthMask(flag):
    if False:
        for i in range(10):
            print('nop')
    _lib.glDepthMask(flag)
_lib.glDepthRangef.argtypes = (ctypes.c_float, ctypes.c_float)

def glDepthRange(zNear, zFar):
    if False:
        return 10
    _lib.glDepthRangef(zNear, zFar)
_lib.glDetachShader.argtypes = (ctypes.c_uint, ctypes.c_uint)

def glDetachShader(program, shader):
    if False:
        print('Hello World!')
    _lib.glDetachShader(program, shader)
_lib.glDisable.argtypes = (ctypes.c_uint,)

def glDisable(cap):
    if False:
        i = 10
        return i + 15
    _lib.glDisable(cap)
_lib.glDisableVertexAttribArray.argtypes = (ctypes.c_uint,)

def glDisableVertexAttribArray(index):
    if False:
        for i in range(10):
            print('nop')
    _lib.glDisableVertexAttribArray(index)
_lib.glDrawArrays.argtypes = (ctypes.c_uint, ctypes.c_int, ctypes.c_int)

def glDrawArrays(mode, first, count):
    if False:
        print('Hello World!')
    _lib.glDrawArrays(mode, first, count)
_lib.glDrawElements.argtypes = (ctypes.c_uint, ctypes.c_int, ctypes.c_uint, ctypes.c_void_p)

def glDrawElements(mode, count, type, offset):
    if False:
        print('Hello World!')
    if offset is None:
        offset = ctypes.c_void_p(0)
    elif isinstance(offset, ctypes.c_void_p):
        pass
    elif isinstance(offset, (int, ctypes.c_int)):
        offset = ctypes.c_void_p(int(offset))
    else:
        if not offset.flags['C_CONTIGUOUS']:
            offset = offset.copy('C')
        offset_ = offset
        offset = offset.ctypes.data
    indices = offset
    res = _lib.glDrawElements(mode, count, type, indices)
_lib.glEnable.argtypes = (ctypes.c_uint,)

def glEnable(cap):
    if False:
        return 10
    _lib.glEnable(cap)
_lib.glEnableVertexAttribArray.argtypes = (ctypes.c_uint,)

def glEnableVertexAttribArray(index):
    if False:
        i = 10
        return i + 15
    _lib.glEnableVertexAttribArray(index)
_lib.glFinish.argtypes = ()

def glFinish():
    if False:
        print('Hello World!')
    _lib.glFinish()
_lib.glFlush.argtypes = ()

def glFlush():
    if False:
        print('Hello World!')
    _lib.glFlush()
_lib.glFramebufferRenderbuffer.argtypes = (ctypes.c_uint, ctypes.c_uint, ctypes.c_uint, ctypes.c_uint)

def glFramebufferRenderbuffer(target, attachment, renderbuffertarget, renderbuffer):
    if False:
        return 10
    _lib.glFramebufferRenderbuffer(target, attachment, renderbuffertarget, renderbuffer)
_lib.glFramebufferTexture2D.argtypes = (ctypes.c_uint, ctypes.c_uint, ctypes.c_uint, ctypes.c_uint, ctypes.c_int)

def glFramebufferTexture2D(target, attachment, textarget, texture, level):
    if False:
        for i in range(10):
            print('nop')
    _lib.glFramebufferTexture2D(target, attachment, textarget, texture, level)
_lib.glFrontFace.argtypes = (ctypes.c_uint,)

def glFrontFace(mode):
    if False:
        print('Hello World!')
    _lib.glFrontFace(mode)
_lib.glGenBuffers.argtypes = (ctypes.c_int, ctypes.POINTER(ctypes.c_uint))

def glCreateBuffer():
    if False:
        return 10
    n = 1
    buffers = (ctypes.c_uint * n)()
    res = _lib.glGenBuffers(n, buffers)
    return buffers[0]
_lib.glGenFramebuffers.argtypes = (ctypes.c_int, ctypes.POINTER(ctypes.c_uint))

def glCreateFramebuffer():
    if False:
        i = 10
        return i + 15
    n = 1
    framebuffers = (ctypes.c_uint * n)()
    res = _lib.glGenFramebuffers(n, framebuffers)
    return framebuffers[0]
_lib.glGenRenderbuffers.argtypes = (ctypes.c_int, ctypes.POINTER(ctypes.c_uint))

def glCreateRenderbuffer():
    if False:
        i = 10
        return i + 15
    n = 1
    renderbuffers = (ctypes.c_uint * n)()
    res = _lib.glGenRenderbuffers(n, renderbuffers)
    return renderbuffers[0]
_lib.glGenTextures.argtypes = (ctypes.c_int, ctypes.POINTER(ctypes.c_uint))

def glCreateTexture():
    if False:
        return 10
    n = 1
    textures = (ctypes.c_uint * n)()
    res = _lib.glGenTextures(n, textures)
    return textures[0]
_lib.glGenerateMipmap.argtypes = (ctypes.c_uint,)

def glGenerateMipmap(target):
    if False:
        i = 10
        return i + 15
    _lib.glGenerateMipmap(target)
_lib.glGetActiveAttrib.argtypes = (ctypes.c_uint, ctypes.c_uint, ctypes.c_int, ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_uint), ctypes.c_char_p)

def glGetActiveAttrib(program, index):
    if False:
        for i in range(10):
            print('nop')
    bufsize = 256
    length = (ctypes.c_int * 1)()
    size = (ctypes.c_int * 1)()
    type = (ctypes.c_uint * 1)()
    name = ctypes.create_string_buffer(bufsize)
    res = _lib.glGetActiveAttrib(program, index, bufsize, length, size, type, name)
    name = name[:length[0]].decode('utf-8')
    return (name, size[0], type[0])
_lib.glGetActiveUniform.argtypes = (ctypes.c_uint, ctypes.c_uint, ctypes.c_int, ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_uint), ctypes.c_char_p)

def glGetActiveUniform(program, index):
    if False:
        i = 10
        return i + 15
    bufsize = 256
    length = (ctypes.c_int * 1)()
    size = (ctypes.c_int * 1)()
    type = (ctypes.c_uint * 1)()
    name = ctypes.create_string_buffer(bufsize)
    res = _lib.glGetActiveUniform(program, index, bufsize, length, size, type, name)
    name = name[:length[0]].decode('utf-8')
    return (name, size[0], type[0])
_lib.glGetAttachedShaders.argtypes = (ctypes.c_uint, ctypes.c_int, ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_uint))

def glGetAttachedShaders(program):
    if False:
        print('Hello World!')
    maxcount = 256
    count = (ctypes.c_int * 1)()
    shaders = (ctypes.c_uint * maxcount)()
    res = _lib.glGetAttachedShaders(program, maxcount, count, shaders)
    return tuple(shaders[:count[0]])
_lib.glGetAttribLocation.argtypes = (ctypes.c_uint, ctypes.c_char_p)
_lib.glGetAttribLocation.restype = ctypes.c_int

def glGetAttribLocation(program, name):
    if False:
        return 10
    name = ctypes.c_char_p(name.encode('utf-8'))
    res = _lib.glGetAttribLocation(program, name)
    return res
_lib.glGetBooleanv.argtypes = (ctypes.c_uint, ctypes.POINTER(ctypes.c_bool))

def _glGetBooleanv(pname):
    if False:
        print('Hello World!')
    params = (ctypes.c_bool * 1)()
    res = _lib.glGetBooleanv(pname, params)
    return params[0]
_lib.glGetBufferParameteriv.argtypes = (ctypes.c_uint, ctypes.c_uint, ctypes.POINTER(ctypes.c_int))

def glGetBufferParameter(target, pname):
    if False:
        print('Hello World!')
    d = -2 ** 31
    params = (ctypes.c_int * 1)(d)
    res = _lib.glGetBufferParameteriv(target, pname, params)
    return params[0]
_lib.glGetError.argtypes = ()
_lib.glGetError.restype = ctypes.c_uint

def glGetError():
    if False:
        i = 10
        return i + 15
    return _lib.glGetError()
_lib.glGetFloatv.argtypes = (ctypes.c_uint, ctypes.POINTER(ctypes.c_float))

def _glGetFloatv(pname):
    if False:
        i = 10
        return i + 15
    n = 16
    d = float('Inf')
    params = (ctypes.c_float * n)(*[d for i in range(n)])
    res = _lib.glGetFloatv(pname, params)
    params = [p for p in params if p != d]
    if len(params) == 1:
        return params[0]
    else:
        return tuple(params)
_lib.glGetFramebufferAttachmentParameteriv.argtypes = (ctypes.c_uint, ctypes.c_uint, ctypes.c_uint, ctypes.POINTER(ctypes.c_int))

def glGetFramebufferAttachmentParameter(target, attachment, pname):
    if False:
        while True:
            i = 10
    d = -2 ** 31
    params = (ctypes.c_int * 1)(d)
    res = _lib.glGetFramebufferAttachmentParameteriv(target, attachment, pname, params)
    return params[0]
_lib.glGetIntegerv.argtypes = (ctypes.c_uint, ctypes.POINTER(ctypes.c_int))

def _glGetIntegerv(pname):
    if False:
        for i in range(10):
            print('nop')
    n = 16
    d = -2 ** 31
    params = (ctypes.c_int * n)(*[d for i in range(n)])
    res = _lib.glGetIntegerv(pname, params)
    params = [p for p in params if p != d]
    if len(params) == 1:
        return params[0]
    else:
        return tuple(params)
_lib.glGetProgramInfoLog.argtypes = (ctypes.c_uint, ctypes.c_int, ctypes.POINTER(ctypes.c_int), ctypes.c_char_p)

def glGetProgramInfoLog(program):
    if False:
        for i in range(10):
            print('nop')
    bufsize = 1024
    length = (ctypes.c_int * 1)()
    infolog = ctypes.create_string_buffer(bufsize)
    res = _lib.glGetProgramInfoLog(program, bufsize, length, infolog)
    return infolog[:length[0]].decode('utf-8')
_lib.glGetProgramiv.argtypes = (ctypes.c_uint, ctypes.c_uint, ctypes.POINTER(ctypes.c_int))

def glGetProgramParameter(program, pname):
    if False:
        while True:
            i = 10
    params = (ctypes.c_int * 1)()
    res = _lib.glGetProgramiv(program, pname, params)
    return params[0]
_lib.glGetRenderbufferParameteriv.argtypes = (ctypes.c_uint, ctypes.c_uint, ctypes.POINTER(ctypes.c_int))

def glGetRenderbufferParameter(target, pname):
    if False:
        while True:
            i = 10
    d = -2 ** 31
    params = (ctypes.c_int * 1)(d)
    res = _lib.glGetRenderbufferParameteriv(target, pname, params)
    return params[0]
_lib.glGetShaderInfoLog.argtypes = (ctypes.c_uint, ctypes.c_int, ctypes.POINTER(ctypes.c_int), ctypes.c_char_p)

def glGetShaderInfoLog(shader):
    if False:
        return 10
    bufsize = 1024
    length = (ctypes.c_int * 1)()
    infolog = ctypes.create_string_buffer(bufsize)
    res = _lib.glGetShaderInfoLog(shader, bufsize, length, infolog)
    return infolog[:length[0]].decode('utf-8')
_lib.glGetShaderPrecisionFormat.argtypes = (ctypes.c_uint, ctypes.c_uint, ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int))

def glGetShaderPrecisionFormat(shadertype, precisiontype):
    if False:
        while True:
            i = 10
    range = (ctypes.c_int * 1)()
    precision = (ctypes.c_int * 1)()
    res = _lib.glGetShaderPrecisionFormat(shadertype, precisiontype, range, precision)
    return (range[0], precision[0])
_lib.glGetShaderSource.argtypes = (ctypes.c_uint, ctypes.c_int, ctypes.POINTER(ctypes.c_int), ctypes.c_char_p)

def glGetShaderSource(shader):
    if False:
        print('Hello World!')
    bufsize = 1024 * 1024
    length = (ctypes.c_int * 1)()
    source = (ctypes.c_char * bufsize)()
    res = _lib.glGetShaderSource(shader, bufsize, length, source)
    return source.value[:length[0]].decode('utf-8')
_lib.glGetShaderiv.argtypes = (ctypes.c_uint, ctypes.c_uint, ctypes.POINTER(ctypes.c_int))

def glGetShaderParameter(shader, pname):
    if False:
        i = 10
        return i + 15
    params = (ctypes.c_int * 1)()
    res = _lib.glGetShaderiv(shader, pname, params)
    return params[0]
_lib.glGetString.argtypes = (ctypes.c_uint,)
_lib.glGetString.restype = ctypes.c_char_p

def glGetParameter(pname):
    if False:
        return 10
    if pname in [33902, 33901, 32773, 3106, 2931, 2928, 2849, 32824, 10752, 32938]:
        return _glGetFloatv(pname)
    elif pname in [7936, 7937, 7938, 35724, 7939]:
        pass
    else:
        return _glGetIntegerv(pname)
    name = pname
    res = _lib.glGetString(name)
    return ctypes.string_at(res).decode('utf-8') if res else ''
_lib.glGetTexParameterfv.argtypes = (ctypes.c_uint, ctypes.c_uint, ctypes.POINTER(ctypes.c_float))

def glGetTexParameter(target, pname):
    if False:
        return 10
    d = float('Inf')
    params = (ctypes.c_float * 1)(d)
    res = _lib.glGetTexParameterfv(target, pname, params)
    return params[0]
_lib.glGetUniformfv.argtypes = (ctypes.c_uint, ctypes.c_int, ctypes.POINTER(ctypes.c_float))

def glGetUniform(program, location):
    if False:
        for i in range(10):
            print('nop')
    n = 16
    d = float('Inf')
    params = (ctypes.c_float * n)(*[d for i in range(n)])
    res = _lib.glGetUniformfv(program, location, params)
    params = [p for p in params if p != d]
    if len(params) == 1:
        return params[0]
    else:
        return tuple(params)
_lib.glGetUniformLocation.argtypes = (ctypes.c_uint, ctypes.c_char_p)
_lib.glGetUniformLocation.restype = ctypes.c_int

def glGetUniformLocation(program, name):
    if False:
        while True:
            i = 10
    name = ctypes.c_char_p(name.encode('utf-8'))
    res = _lib.glGetUniformLocation(program, name)
    return res
_lib.glGetVertexAttribfv.argtypes = (ctypes.c_uint, ctypes.c_uint, ctypes.POINTER(ctypes.c_float))

def glGetVertexAttrib(index, pname):
    if False:
        for i in range(10):
            print('nop')
    n = 4
    d = float('Inf')
    params = (ctypes.c_float * n)(*[d for i in range(n)])
    res = _lib.glGetVertexAttribfv(index, pname, params)
    params = [p for p in params if p != d]
    if len(params) == 1:
        return params[0]
    else:
        return tuple(params)
_lib.glGetVertexAttribPointerv.argtypes = (ctypes.c_uint, ctypes.c_uint, ctypes.POINTER(ctypes.c_void_p))

def glGetVertexAttribOffset(index, pname):
    if False:
        return 10
    pointer = (ctypes.c_void_p * 1)()
    res = _lib.glGetVertexAttribPointerv(index, pname, pointer)
    return pointer[0] or 0
_lib.glHint.argtypes = (ctypes.c_uint, ctypes.c_uint)

def glHint(target, mode):
    if False:
        for i in range(10):
            print('nop')
    _lib.glHint(target, mode)
_lib.glIsBuffer.argtypes = (ctypes.c_uint,)
_lib.glIsBuffer.restype = ctypes.c_bool

def glIsBuffer(buffer):
    if False:
        print('Hello World!')
    return _lib.glIsBuffer(buffer)
_lib.glIsEnabled.argtypes = (ctypes.c_uint,)
_lib.glIsEnabled.restype = ctypes.c_bool

def glIsEnabled(cap):
    if False:
        i = 10
        return i + 15
    return _lib.glIsEnabled(cap)
_lib.glIsFramebuffer.argtypes = (ctypes.c_uint,)
_lib.glIsFramebuffer.restype = ctypes.c_bool

def glIsFramebuffer(framebuffer):
    if False:
        while True:
            i = 10
    return _lib.glIsFramebuffer(framebuffer)
_lib.glIsProgram.argtypes = (ctypes.c_uint,)
_lib.glIsProgram.restype = ctypes.c_bool

def glIsProgram(program):
    if False:
        return 10
    return _lib.glIsProgram(program)
_lib.glIsRenderbuffer.argtypes = (ctypes.c_uint,)
_lib.glIsRenderbuffer.restype = ctypes.c_bool

def glIsRenderbuffer(renderbuffer):
    if False:
        for i in range(10):
            print('nop')
    return _lib.glIsRenderbuffer(renderbuffer)
_lib.glIsShader.argtypes = (ctypes.c_uint,)
_lib.glIsShader.restype = ctypes.c_bool

def glIsShader(shader):
    if False:
        while True:
            i = 10
    return _lib.glIsShader(shader)
_lib.glIsTexture.argtypes = (ctypes.c_uint,)
_lib.glIsTexture.restype = ctypes.c_bool

def glIsTexture(texture):
    if False:
        i = 10
        return i + 15
    return _lib.glIsTexture(texture)
_lib.glLineWidth.argtypes = (ctypes.c_float,)

def glLineWidth(width):
    if False:
        for i in range(10):
            print('nop')
    _lib.glLineWidth(width)
_lib.glLinkProgram.argtypes = (ctypes.c_uint,)

def glLinkProgram(program):
    if False:
        for i in range(10):
            print('nop')
    _lib.glLinkProgram(program)
_lib.glPixelStorei.argtypes = (ctypes.c_uint, ctypes.c_int)

def glPixelStorei(pname, param):
    if False:
        for i in range(10):
            print('nop')
    _lib.glPixelStorei(pname, param)
_lib.glPolygonOffset.argtypes = (ctypes.c_float, ctypes.c_float)

def glPolygonOffset(factor, units):
    if False:
        print('Hello World!')
    _lib.glPolygonOffset(factor, units)
_lib.glReadPixels.argtypes = (ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_uint, ctypes.c_uint, ctypes.c_void_p)

def glReadPixels(x, y, width, height, format, type):
    if False:
        for i in range(10):
            print('nop')
    t = {6406: 1, 6407: 3, 6408: 4}[format]
    nb = {5121: 1, 5126: 4}[type]
    size = int(width * height * t * nb)
    pixels = ctypes.create_string_buffer(size)
    res = _lib.glReadPixels(x, y, width, height, format, type, pixels)
    return pixels[:]
_lib.glRenderbufferStorage.argtypes = (ctypes.c_uint, ctypes.c_uint, ctypes.c_int, ctypes.c_int)

def glRenderbufferStorage(target, internalformat, width, height):
    if False:
        for i in range(10):
            print('nop')
    _lib.glRenderbufferStorage(target, internalformat, width, height)
_lib.glSampleCoverage.argtypes = (ctypes.c_float, ctypes.c_bool)

def glSampleCoverage(value, invert):
    if False:
        for i in range(10):
            print('nop')
    _lib.glSampleCoverage(value, invert)
_lib.glScissor.argtypes = (ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int)

def glScissor(x, y, width, height):
    if False:
        return 10
    _lib.glScissor(x, y, width, height)
_lib.glShaderSource.argtypes = (ctypes.c_uint, ctypes.c_int, ctypes.POINTER(ctypes.c_char_p), ctypes.POINTER(ctypes.c_int))

def glShaderSource(shader, source):
    if False:
        return 10
    if isinstance(source, (tuple, list)):
        strings = [s for s in source]
    else:
        strings = [source]
    count = len(strings)
    string = (ctypes.c_char_p * count)(*[s.encode('utf-8') for s in strings])
    length = (ctypes.c_int * count)(*[len(s) for s in strings])
    res = _lib.glShaderSource(shader, count, string, length)
_lib.glStencilFunc.argtypes = (ctypes.c_uint, ctypes.c_int, ctypes.c_uint)

def glStencilFunc(func, ref, mask):
    if False:
        while True:
            i = 10
    _lib.glStencilFunc(func, ref, mask)
_lib.glStencilFuncSeparate.argtypes = (ctypes.c_uint, ctypes.c_uint, ctypes.c_int, ctypes.c_uint)

def glStencilFuncSeparate(face, func, ref, mask):
    if False:
        print('Hello World!')
    _lib.glStencilFuncSeparate(face, func, ref, mask)
_lib.glStencilMask.argtypes = (ctypes.c_uint,)

def glStencilMask(mask):
    if False:
        print('Hello World!')
    _lib.glStencilMask(mask)
_lib.glStencilMaskSeparate.argtypes = (ctypes.c_uint, ctypes.c_uint)

def glStencilMaskSeparate(face, mask):
    if False:
        i = 10
        return i + 15
    _lib.glStencilMaskSeparate(face, mask)
_lib.glStencilOp.argtypes = (ctypes.c_uint, ctypes.c_uint, ctypes.c_uint)

def glStencilOp(fail, zfail, zpass):
    if False:
        for i in range(10):
            print('nop')
    _lib.glStencilOp(fail, zfail, zpass)
_lib.glStencilOpSeparate.argtypes = (ctypes.c_uint, ctypes.c_uint, ctypes.c_uint, ctypes.c_uint)

def glStencilOpSeparate(face, fail, zfail, zpass):
    if False:
        i = 10
        return i + 15
    _lib.glStencilOpSeparate(face, fail, zfail, zpass)
_lib.glTexImage2D.argtypes = (ctypes.c_uint, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_uint, ctypes.c_uint, ctypes.c_void_p)

def glTexImage2D(target, level, internalformat, format, type, pixels):
    if False:
        i = 10
        return i + 15
    border = 0
    if isinstance(pixels, (tuple, list)):
        (height, width) = pixels
        pixels = ctypes.c_void_p(0)
        pixels = None
    else:
        if not pixels.flags['C_CONTIGUOUS']:
            pixels = pixels.copy('C')
        pixels_ = pixels
        pixels = pixels_.ctypes.data
        (height, width) = pixels_.shape[:2]
    res = _lib.glTexImage2D(target, level, internalformat, width, height, border, format, type, pixels)
_lib.glTexParameterf.argtypes = (ctypes.c_uint, ctypes.c_uint, ctypes.c_float)

def glTexParameterf(target, pname, param):
    if False:
        print('Hello World!')
    _lib.glTexParameterf(target, pname, param)
_lib.glTexParameteri.argtypes = (ctypes.c_uint, ctypes.c_uint, ctypes.c_int)

def glTexParameteri(target, pname, param):
    if False:
        i = 10
        return i + 15
    _lib.glTexParameteri(target, pname, param)
_lib.glTexSubImage2D.argtypes = (ctypes.c_uint, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_uint, ctypes.c_uint, ctypes.c_void_p)

def glTexSubImage2D(target, level, xoffset, yoffset, format, type, pixels):
    if False:
        while True:
            i = 10
    if not pixels.flags['C_CONTIGUOUS']:
        pixels = pixels.copy('C')
    pixels_ = pixels
    pixels = pixels_.ctypes.data
    (height, width) = pixels_.shape[:2]
    res = _lib.glTexSubImage2D(target, level, xoffset, yoffset, width, height, format, type, pixels)
_lib.glUniform1f.argtypes = (ctypes.c_int, ctypes.c_float)

def glUniform1f(location, v1):
    if False:
        while True:
            i = 10
    _lib.glUniform1f(location, v1)
_lib.glUniform2f.argtypes = (ctypes.c_int, ctypes.c_float, ctypes.c_float)

def glUniform2f(location, v1, v2):
    if False:
        i = 10
        return i + 15
    _lib.glUniform2f(location, v1, v2)
_lib.glUniform3f.argtypes = (ctypes.c_int, ctypes.c_float, ctypes.c_float, ctypes.c_float)

def glUniform3f(location, v1, v2, v3):
    if False:
        print('Hello World!')
    _lib.glUniform3f(location, v1, v2, v3)
_lib.glUniform4f.argtypes = (ctypes.c_int, ctypes.c_float, ctypes.c_float, ctypes.c_float, ctypes.c_float)

def glUniform4f(location, v1, v2, v3, v4):
    if False:
        return 10
    _lib.glUniform4f(location, v1, v2, v3, v4)
_lib.glUniform1i.argtypes = (ctypes.c_int, ctypes.c_int)

def glUniform1i(location, v1):
    if False:
        return 10
    _lib.glUniform1i(location, v1)
_lib.glUniform2i.argtypes = (ctypes.c_int, ctypes.c_int, ctypes.c_int)

def glUniform2i(location, v1, v2):
    if False:
        i = 10
        return i + 15
    _lib.glUniform2i(location, v1, v2)
_lib.glUniform3i.argtypes = (ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int)

def glUniform3i(location, v1, v2, v3):
    if False:
        print('Hello World!')
    _lib.glUniform3i(location, v1, v2, v3)
_lib.glUniform4i.argtypes = (ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int)

def glUniform4i(location, v1, v2, v3, v4):
    if False:
        return 10
    _lib.glUniform4i(location, v1, v2, v3, v4)
_lib.glUniform1fv.argtypes = (ctypes.c_int, ctypes.c_int, ctypes.POINTER(ctypes.c_float))

def glUniform1fv(location, count, values):
    if False:
        return 10
    values = [float(val) for val in values]
    values = (ctypes.c_float * len(values))(*values)
    _lib.glUniform1fv(location, count, values)
_lib.glUniform2fv.argtypes = (ctypes.c_int, ctypes.c_int, ctypes.POINTER(ctypes.c_float))

def glUniform2fv(location, count, values):
    if False:
        while True:
            i = 10
    values = [float(val) for val in values]
    values = (ctypes.c_float * len(values))(*values)
    _lib.glUniform2fv(location, count, values)
_lib.glUniform3fv.argtypes = (ctypes.c_int, ctypes.c_int, ctypes.POINTER(ctypes.c_float))

def glUniform3fv(location, count, values):
    if False:
        for i in range(10):
            print('nop')
    values = [float(val) for val in values]
    values = (ctypes.c_float * len(values))(*values)
    _lib.glUniform3fv(location, count, values)
_lib.glUniform4fv.argtypes = (ctypes.c_int, ctypes.c_int, ctypes.POINTER(ctypes.c_float))

def glUniform4fv(location, count, values):
    if False:
        while True:
            i = 10
    values = [float(val) for val in values]
    values = (ctypes.c_float * len(values))(*values)
    _lib.glUniform4fv(location, count, values)
_lib.glUniform1iv.argtypes = (ctypes.c_int, ctypes.c_int, ctypes.POINTER(ctypes.c_int))

def glUniform1iv(location, count, values):
    if False:
        while True:
            i = 10
    values = [int(val) for val in values]
    values = (ctypes.c_int * len(values))(*values)
    _lib.glUniform1iv(location, count, values)
_lib.glUniform2iv.argtypes = (ctypes.c_int, ctypes.c_int, ctypes.POINTER(ctypes.c_int))

def glUniform2iv(location, count, values):
    if False:
        for i in range(10):
            print('nop')
    values = [int(val) for val in values]
    values = (ctypes.c_int * len(values))(*values)
    _lib.glUniform2iv(location, count, values)
_lib.glUniform3iv.argtypes = (ctypes.c_int, ctypes.c_int, ctypes.POINTER(ctypes.c_int))

def glUniform3iv(location, count, values):
    if False:
        while True:
            i = 10
    values = [int(val) for val in values]
    values = (ctypes.c_int * len(values))(*values)
    _lib.glUniform3iv(location, count, values)
_lib.glUniform4iv.argtypes = (ctypes.c_int, ctypes.c_int, ctypes.POINTER(ctypes.c_int))

def glUniform4iv(location, count, values):
    if False:
        while True:
            i = 10
    values = [int(val) for val in values]
    values = (ctypes.c_int * len(values))(*values)
    _lib.glUniform4iv(location, count, values)
_lib.glUniformMatrix2fv.argtypes = (ctypes.c_int, ctypes.c_int, ctypes.c_bool, ctypes.POINTER(ctypes.c_float))

def glUniformMatrix2fv(location, count, transpose, values):
    if False:
        print('Hello World!')
    if not values.flags['C_CONTIGUOUS']:
        values = values.copy()
    assert values.dtype.name == 'float32'
    values_ = values
    values = values_.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    _lib.glUniformMatrix2fv(location, count, transpose, values)
_lib.glUniformMatrix3fv.argtypes = (ctypes.c_int, ctypes.c_int, ctypes.c_bool, ctypes.POINTER(ctypes.c_float))

def glUniformMatrix3fv(location, count, transpose, values):
    if False:
        while True:
            i = 10
    if not values.flags['C_CONTIGUOUS']:
        values = values.copy()
    assert values.dtype.name == 'float32'
    values_ = values
    values = values_.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    _lib.glUniformMatrix3fv(location, count, transpose, values)
_lib.glUniformMatrix4fv.argtypes = (ctypes.c_int, ctypes.c_int, ctypes.c_bool, ctypes.POINTER(ctypes.c_float))

def glUniformMatrix4fv(location, count, transpose, values):
    if False:
        i = 10
        return i + 15
    if not values.flags['C_CONTIGUOUS']:
        values = values.copy()
    assert values.dtype.name == 'float32'
    values_ = values
    values = values_.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    _lib.glUniformMatrix4fv(location, count, transpose, values)
_lib.glUseProgram.argtypes = (ctypes.c_uint,)

def glUseProgram(program):
    if False:
        return 10
    _lib.glUseProgram(program)
_lib.glValidateProgram.argtypes = (ctypes.c_uint,)

def glValidateProgram(program):
    if False:
        print('Hello World!')
    _lib.glValidateProgram(program)
_lib.glVertexAttrib1f.argtypes = (ctypes.c_uint, ctypes.c_float)

def glVertexAttrib1f(index, v1):
    if False:
        return 10
    _lib.glVertexAttrib1f(index, v1)
_lib.glVertexAttrib2f.argtypes = (ctypes.c_uint, ctypes.c_float, ctypes.c_float)

def glVertexAttrib2f(index, v1, v2):
    if False:
        i = 10
        return i + 15
    _lib.glVertexAttrib2f(index, v1, v2)
_lib.glVertexAttrib3f.argtypes = (ctypes.c_uint, ctypes.c_float, ctypes.c_float, ctypes.c_float)

def glVertexAttrib3f(index, v1, v2, v3):
    if False:
        return 10
    _lib.glVertexAttrib3f(index, v1, v2, v3)
_lib.glVertexAttrib4f.argtypes = (ctypes.c_uint, ctypes.c_float, ctypes.c_float, ctypes.c_float, ctypes.c_float)

def glVertexAttrib4f(index, v1, v2, v3, v4):
    if False:
        return 10
    _lib.glVertexAttrib4f(index, v1, v2, v3, v4)
_lib.glVertexAttribPointer.argtypes = (ctypes.c_uint, ctypes.c_int, ctypes.c_uint, ctypes.c_bool, ctypes.c_int, ctypes.c_void_p)

def glVertexAttribPointer(indx, size, type, normalized, stride, offset):
    if False:
        print('Hello World!')
    if offset is None:
        offset = ctypes.c_void_p(0)
    elif isinstance(offset, ctypes.c_void_p):
        pass
    elif isinstance(offset, (int, ctypes.c_int)):
        offset = ctypes.c_void_p(int(offset))
    else:
        if not offset.flags['C_CONTIGUOUS']:
            offset = offset.copy('C')
        offset_ = offset
        offset = offset.ctypes.data
        key = '_vert_attr_' + str(indx)
        setattr(glVertexAttribPointer, key, offset_)
    ptr = offset
    res = _lib.glVertexAttribPointer(indx, size, type, normalized, stride, ptr)
_lib.glViewport.argtypes = (ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int)

def glViewport(x, y, width, height):
    if False:
        print('Hello World!')
    _lib.glViewport(x, y, width, height)