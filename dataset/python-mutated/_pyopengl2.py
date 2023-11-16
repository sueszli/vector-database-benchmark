"""GL definitions converted to Python by codegen/createglapi.py.

THIS CODE IS AUTO-GENERATED. DO NOT EDIT.

Proxy API for GL ES 2.0 subset, via the PyOpenGL library.

"""
import ctypes
from OpenGL import GL
import OpenGL.GL.framebufferobjects as FBO

def glBindAttribLocation(program, index, name):
    if False:
        while True:
            i = 10
    name = name.encode('utf-8')
    return GL.glBindAttribLocation(program, index, name)

def glBufferData(target, data, usage):
    if False:
        return 10
    'Data can be numpy array or the size of data to allocate.'
    if isinstance(data, int):
        size = data
        data = None
    else:
        size = data.nbytes
    GL.glBufferData(target, size, data, usage)

def glBufferSubData(target, offset, data):
    if False:
        while True:
            i = 10
    size = data.nbytes
    GL.glBufferSubData(target, offset, size, data)

def glCompressedTexImage2D(target, level, internalformat, width, height, border, data):
    if False:
        for i in range(10):
            print('nop')
    size = data.size
    GL.glCompressedTexImage2D(target, level, internalformat, width, height, border, size, data)

def glCompressedTexSubImage2D(target, level, xoffset, yoffset, width, height, format, data):
    if False:
        print('Hello World!')
    size = data.size
    GL.glCompressedTexSubImage2D(target, level, xoffset, yoffset, width, height, format, size, data)

def glDeleteBuffer(buffer):
    if False:
        print('Hello World!')
    GL.glDeleteBuffers(1, [buffer])

def glDeleteFramebuffer(framebuffer):
    if False:
        return 10
    FBO.glDeleteFramebuffers(1, [framebuffer])

def glDeleteRenderbuffer(renderbuffer):
    if False:
        print('Hello World!')
    FBO.glDeleteRenderbuffers(1, [renderbuffer])

def glDeleteTexture(texture):
    if False:
        print('Hello World!')
    GL.glDeleteTextures([texture])

def glDrawElements(mode, count, type, offset):
    if False:
        print('Hello World!')
    if offset is None:
        offset = ctypes.c_void_p(0)
    elif isinstance(offset, (int, ctypes.c_int)):
        offset = ctypes.c_void_p(int(offset))
    return GL.glDrawElements(mode, count, type, offset)

def glCreateBuffer():
    if False:
        for i in range(10):
            print('nop')
    return GL.glGenBuffers(1)

def glCreateFramebuffer():
    if False:
        while True:
            i = 10
    return FBO.glGenFramebuffers(1)

def glCreateRenderbuffer():
    if False:
        i = 10
        return i + 15
    return FBO.glGenRenderbuffers(1)

def glCreateTexture():
    if False:
        i = 10
        return i + 15
    return GL.glGenTextures(1)

def glGetActiveAttrib(program, index):
    if False:
        print('Hello World!')
    bufsize = 256
    (name, size, type) = GL.glGetActiveAttrib(program, index, bufSize=bufsize)
    return (name.decode('utf-8'), size, type)

def glGetActiveUniform(program, index):
    if False:
        i = 10
        return i + 15
    (name, size, type) = GL.glGetActiveUniform(program, index)
    return (name.decode('utf-8'), size, type)

def glGetAttribLocation(program, name):
    if False:
        while True:
            i = 10
    name = name.encode('utf-8')
    return GL.glGetAttribLocation(program, name)

def glGetFramebufferAttachmentParameter(target, attachment, pname):
    if False:
        return 10
    d = -2 ** 31
    params = (ctypes.c_int * 1)(d)
    FBO.glGetFramebufferAttachmentParameteriv(target, attachment, pname, params)
    return params[0]

def glGetProgramInfoLog(program):
    if False:
        while True:
            i = 10
    res = GL.glGetProgramInfoLog(program)
    return res.decode('utf-8') if isinstance(res, bytes) else res

def glGetRenderbufferParameter(target, pname):
    if False:
        for i in range(10):
            print('nop')
    d = -2 ** 31
    params = (ctypes.c_int * 1)(d)
    FBO.glGetRenderbufferParameteriv(target, pname, params)
    return params[0]

def glGetShaderInfoLog(shader):
    if False:
        i = 10
        return i + 15
    res = GL.glGetShaderInfoLog(shader)
    return res.decode('utf-8') if isinstance(res, bytes) else res

def glGetShaderSource(shader):
    if False:
        return 10
    res = GL.glGetShaderSource(shader)
    return res.decode('utf-8')

def glGetParameter(pname):
    if False:
        while True:
            i = 10
    if pname in [33902, 33901, 32773, 3106, 2931, 2928, 2849, 32824, 10752, 32938]:
        return GL.glGetFloatv(pname)
    elif pname in [7936, 7937, 7938, 35724, 7939]:
        pass
    else:
        return GL.glGetIntegerv(pname)
    res = GL.glGetString(pname)
    return res.decode('utf-8')

def glGetUniform(program, location):
    if False:
        for i in range(10):
            print('nop')
    n = 16
    d = float('Inf')
    params = (ctypes.c_float * n)(*[d for i in range(n)])
    GL.glGetUniformfv(program, location, params)
    params = [p for p in params if p != d]
    if len(params) == 1:
        return params[0]
    else:
        return tuple(params)

def glGetUniformLocation(program, name):
    if False:
        return 10
    name = name.encode('utf-8')
    return GL.glGetUniformLocation(program, name)

def glGetVertexAttrib(index, pname):
    if False:
        i = 10
        return i + 15
    n = 4
    d = float('Inf')
    params = (ctypes.c_float * n)(*[d for i in range(n)])
    GL.glGetVertexAttribfv(index, pname, params)
    params = [p for p in params if p != d]
    if len(params) == 1:
        return params[0]
    else:
        return tuple(params)

def glGetVertexAttribOffset(index, pname):
    if False:
        while True:
            i = 10
    try:
        return GL.glGetVertexAttribPointerv(index, pname)
    except TypeError:
        pointer = (ctypes.c_void_p * 1)()
        GL.glGetVertexAttribPointerv(index, pname, pointer)
        return pointer[0] or 0

def glShaderSource(shader, source):
    if False:
        print('Hello World!')
    if isinstance(source, (tuple, list)):
        strings = [s for s in source]
    else:
        strings = [source]
    GL.glShaderSource(shader, strings)

def glTexImage2D(target, level, internalformat, format, type, pixels):
    if False:
        for i in range(10):
            print('nop')
    border = 0
    if isinstance(pixels, (tuple, list)):
        (height, width) = pixels
        pixels = None
    else:
        (height, width) = pixels.shape[:2]
    GL.glTexImage2D(target, level, internalformat, width, height, border, format, type, pixels)

def glTexSubImage2D(target, level, xoffset, yoffset, format, type, pixels):
    if False:
        i = 10
        return i + 15
    (height, width) = pixels.shape[:2]
    GL.glTexSubImage2D(target, level, xoffset, yoffset, width, height, format, type, pixels)

def glVertexAttribPointer(indx, size, type, normalized, stride, offset):
    if False:
        return 10
    if offset is None:
        offset = ctypes.c_void_p(0)
    elif isinstance(offset, (int, ctypes.c_int)):
        offset = ctypes.c_void_p(int(offset))
    return GL.glVertexAttribPointer(indx, size, type, normalized, stride, offset)
_functions_to_import = [('glActiveTexture', 'glActiveTexture'), ('glAttachShader', 'glAttachShader'), ('glBindBuffer', 'glBindBuffer'), ('glBindFramebuffer', 'glBindFramebuffer'), ('glBindRenderbuffer', 'glBindRenderbuffer'), ('glBindTexture', 'glBindTexture'), ('glBlendColor', 'glBlendColor'), ('glBlendEquation', 'glBlendEquation'), ('glBlendEquationSeparate', 'glBlendEquationSeparate'), ('glBlendFunc', 'glBlendFunc'), ('glBlendFuncSeparate', 'glBlendFuncSeparate'), ('glCheckFramebufferStatus', 'glCheckFramebufferStatus'), ('glClear', 'glClear'), ('glClearColor', 'glClearColor'), ('glClearDepthf', 'glClearDepth'), ('glClearStencil', 'glClearStencil'), ('glColorMask', 'glColorMask'), ('glCompileShader', 'glCompileShader'), ('glCopyTexImage2D', 'glCopyTexImage2D'), ('glCopyTexSubImage2D', 'glCopyTexSubImage2D'), ('glCreateProgram', 'glCreateProgram'), ('glCreateShader', 'glCreateShader'), ('glCullFace', 'glCullFace'), ('glDeleteProgram', 'glDeleteProgram'), ('glDeleteShader', 'glDeleteShader'), ('glDepthFunc', 'glDepthFunc'), ('glDepthMask', 'glDepthMask'), ('glDepthRangef', 'glDepthRange'), ('glDetachShader', 'glDetachShader'), ('glDisable', 'glDisable'), ('glDisableVertexAttribArray', 'glDisableVertexAttribArray'), ('glDrawArrays', 'glDrawArrays'), ('glEnable', 'glEnable'), ('glEnableVertexAttribArray', 'glEnableVertexAttribArray'), ('glFinish', 'glFinish'), ('glFlush', 'glFlush'), ('glFramebufferRenderbuffer', 'glFramebufferRenderbuffer'), ('glFramebufferTexture2D', 'glFramebufferTexture2D'), ('glFrontFace', 'glFrontFace'), ('glGenerateMipmap', 'glGenerateMipmap'), ('glGetAttachedShaders', 'glGetAttachedShaders'), ('glGetBooleanv', '_glGetBooleanv'), ('glGetBufferParameteriv', 'glGetBufferParameter'), ('glGetError', 'glGetError'), ('glGetFloatv', '_glGetFloatv'), ('glGetIntegerv', '_glGetIntegerv'), ('glGetProgramiv', 'glGetProgramParameter'), ('glGetShaderPrecisionFormat', 'glGetShaderPrecisionFormat'), ('glGetShaderiv', 'glGetShaderParameter'), ('glGetTexParameterfv', 'glGetTexParameter'), ('glHint', 'glHint'), ('glIsBuffer', 'glIsBuffer'), ('glIsEnabled', 'glIsEnabled'), ('glIsFramebuffer', 'glIsFramebuffer'), ('glIsProgram', 'glIsProgram'), ('glIsRenderbuffer', 'glIsRenderbuffer'), ('glIsShader', 'glIsShader'), ('glIsTexture', 'glIsTexture'), ('glLineWidth', 'glLineWidth'), ('glLinkProgram', 'glLinkProgram'), ('glPixelStorei', 'glPixelStorei'), ('glPolygonOffset', 'glPolygonOffset'), ('glReadPixels', 'glReadPixels'), ('glRenderbufferStorage', 'glRenderbufferStorage'), ('glSampleCoverage', 'glSampleCoverage'), ('glScissor', 'glScissor'), ('glStencilFunc', 'glStencilFunc'), ('glStencilFuncSeparate', 'glStencilFuncSeparate'), ('glStencilMask', 'glStencilMask'), ('glStencilMaskSeparate', 'glStencilMaskSeparate'), ('glStencilOp', 'glStencilOp'), ('glStencilOpSeparate', 'glStencilOpSeparate'), ('glTexParameterf', 'glTexParameterf'), ('glTexParameteri', 'glTexParameteri'), ('glUniform1f', 'glUniform1f'), ('glUniform2f', 'glUniform2f'), ('glUniform3f', 'glUniform3f'), ('glUniform4f', 'glUniform4f'), ('glUniform1i', 'glUniform1i'), ('glUniform2i', 'glUniform2i'), ('glUniform3i', 'glUniform3i'), ('glUniform4i', 'glUniform4i'), ('glUniform1fv', 'glUniform1fv'), ('glUniform2fv', 'glUniform2fv'), ('glUniform3fv', 'glUniform3fv'), ('glUniform4fv', 'glUniform4fv'), ('glUniform1iv', 'glUniform1iv'), ('glUniform2iv', 'glUniform2iv'), ('glUniform3iv', 'glUniform3iv'), ('glUniform4iv', 'glUniform4iv'), ('glUniformMatrix2fv', 'glUniformMatrix2fv'), ('glUniformMatrix3fv', 'glUniformMatrix3fv'), ('glUniformMatrix4fv', 'glUniformMatrix4fv'), ('glUseProgram', 'glUseProgram'), ('glValidateProgram', 'glValidateProgram'), ('glVertexAttrib1f', 'glVertexAttrib1f'), ('glVertexAttrib2f', 'glVertexAttrib2f'), ('glVertexAttrib3f', 'glVertexAttrib3f'), ('glVertexAttrib4f', 'glVertexAttrib4f'), ('glViewport', 'glViewport')]
_used_functions = ['glBindAttribLocation', 'glBufferData', 'glBufferSubData', 'glCompressedTexImage2D', 'glCompressedTexSubImage2D', 'glDeleteBuffers', 'glDeleteFramebuffers', 'glDeleteRenderbuffers', 'glDeleteTextures', 'glDrawElements', 'glGenBuffers', 'glGenFramebuffers', 'glGenRenderbuffers', 'glGenTextures', 'glGetActiveAttrib', 'glGetActiveUniform', 'glGetAttribLocation', 'glGetFramebufferAttachmentParameteriv', 'glGetProgramInfoLog', 'glGetRenderbufferParameteriv', 'glGetShaderInfoLog', 'glGetShaderSource', 'glGetString', 'glGetUniformfv', 'glGetUniformLocation', 'glGetVertexAttribfv', 'glGetVertexAttribPointerv', 'glShaderSource', 'glTexImage2D', 'glTexSubImage2D', 'glVertexAttribPointer']