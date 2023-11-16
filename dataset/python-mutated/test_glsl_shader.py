from panda3d import core
import os
import struct
import pytest
from _pytest.outcomes import Failed
SHADERS_DIR = core.Filename.from_os_specific(os.path.dirname(__file__))
GLSL_COMPUTE_TEMPLATE = '#version {version}\n{extensions}\n\nlayout(local_size_x = 1, local_size_y = 1) in;\n\n{preamble}\n\nlayout(r8ui) uniform writeonly uimageBuffer _triggered;\n\nvoid _reset() {{\n    imageStore(_triggered, 0, uvec4(0, 0, 0, 0));\n    memoryBarrier();\n}}\n\nvoid _assert(bool cond, int line) {{\n    if (!cond) {{\n        imageStore(_triggered, line, uvec4(1));\n    }}\n}}\n\n#define assert(cond) _assert(cond, __LINE__)\n\nvoid main() {{\n    _reset();\n{body}\n}}\n'

def run_glsl_test(gsg, body, preamble='', inputs={}, version=150, exts=set()):
    if False:
        while True:
            i = 10
    ' Runs a GLSL test on the given GSG.  The given body is executed in the\n    main function and should call assert().  The preamble should contain all\n    of the shader inputs. '
    if not gsg.supports_compute_shaders or not gsg.supports_glsl:
        pytest.skip('compute shaders not supported')
    if not gsg.supports_buffer_texture:
        pytest.skip('buffer textures not supported')
    exts = exts | {'GL_ARB_compute_shader', 'GL_ARB_shader_image_load_store'}
    missing_exts = sorted((ext for ext in exts if not gsg.has_extension(ext)))
    if missing_exts:
        pytest.skip('missing extensions: ' + ' '.join(missing_exts))
    extensions = ''
    for ext in exts:
        extensions += '#extension {ext} : require\n'.format(ext=ext)
    __tracebackhide__ = True
    preamble = preamble.strip()
    body = body.rstrip().lstrip('\n')
    code = GLSL_COMPUTE_TEMPLATE.format(version=version, extensions=extensions, preamble=preamble, body=body)
    line_offset = code[:code.find(body)].count('\n') + 1
    shader = core.Shader.make_compute(core.Shader.SL_GLSL, code)
    assert shader, code
    result = core.Texture('')
    result.set_clear_color((0, 0, 0, 0))
    result.setup_buffer_texture(code.count('\n'), core.Texture.T_unsigned_byte, core.Texture.F_r8i, core.GeomEnums.UH_static)
    attrib = core.ShaderAttrib.make(shader)
    for (name, value) in inputs.items():
        attrib = attrib.set_shader_input(name, value)
    attrib = attrib.set_shader_input('_triggered', result)
    engine = core.GraphicsEngine.get_global_ptr()
    try:
        engine.dispatch_compute((1, 1, 1), attrib, gsg)
    except AssertionError as exc:
        assert False, 'Error executing compute shader:\n' + code
    assert engine.extract_texture_data(result, gsg)
    triggered = result.get_ram_image()
    if any(triggered):
        count = len(triggered) - triggered.count(0)
        lines = body.split('\n')
        formatted = ''
        for (i, line) in enumerate(lines):
            if triggered[i + line_offset]:
                formatted += '=>  ' + line + '\n'
            else:
                formatted += '    ' + line + '\n'
        pytest.fail('{0} GLSL assertions triggered:\n{1}'.format(count, formatted))

def run_glsl_compile_check(gsg, vert_path, frag_path, expect_fail=False):
    if False:
        i = 10
        return i + 15
    'Compile supplied GLSL shader paths and check for errors'
    shader = core.Shader.load(core.Shader.SL_GLSL, vert_path, frag_path)
    assert shader is not None
    if not gsg.supports_glsl:
        expect_fail = True
    shader.prepare_now(gsg.prepared_objects, gsg)
    assert shader.is_prepared(gsg.prepared_objects)
    if expect_fail:
        assert shader.get_error_flag()
    else:
        assert not shader.get_error_flag()

def test_glsl_test(gsg):
    if False:
        i = 10
        return i + 15
    'Test to make sure that the GLSL tests work correctly.'
    run_glsl_test(gsg, 'assert(true);')

def test_glsl_test_fail(gsg):
    if False:
        while True:
            i = 10
    'Same as above, but making sure that the failure case works correctly.'
    with pytest.raises(Failed):
        run_glsl_test(gsg, 'assert(false);')

def test_glsl_sampler(gsg):
    if False:
        i = 10
        return i + 15
    tex1 = core.Texture('')
    tex1.setup_1d_texture(1, core.Texture.T_unsigned_byte, core.Texture.F_rgba8)
    tex1.set_clear_color((0, 2 / 255.0, 1, 1))
    tex2 = core.Texture('')
    tex2.setup_2d_texture(1, 1, core.Texture.T_float, core.Texture.F_rgba32)
    tex2.set_clear_color((1.0, 2.0, -3.14, 0.0))
    tex3 = core.Texture('')
    tex3.setup_3d_texture(1, 1, 1, core.Texture.T_float, core.Texture.F_r32)
    tex3.set_clear_color((0.5, 0.0, 0.0, 1.0))
    preamble = '\n    uniform sampler1D tex1;\n    uniform sampler2D tex2;\n    uniform sampler3D tex3;\n    '
    code = '\n    assert(texelFetch(tex1, 0, 0) == vec4(0, 2 / 255.0, 1, 1));\n    assert(texelFetch(tex2, ivec2(0, 0), 0) == vec4(1.0, 2.0, -3.14, 0.0));\n    assert(texelFetch(tex3, ivec3(0, 0, 0), 0) == vec4(0.5, 0.0, 0.0, 1.0));\n    '
    run_glsl_test(gsg, code, preamble, {'tex1': tex1, 'tex2': tex2, 'tex3': tex3})

def test_glsl_isampler(gsg):
    if False:
        print('Hello World!')
    from struct import pack
    tex1 = core.Texture('')
    tex1.setup_1d_texture(1, core.Texture.T_byte, core.Texture.F_rgba8i)
    tex1.set_ram_image(pack('bbbb', 0, 1, 2, 3))
    tex2 = core.Texture('')
    tex2.setup_2d_texture(1, 1, core.Texture.T_short, core.Texture.F_r16i)
    tex2.set_ram_image(pack('h', 4))
    tex3 = core.Texture('')
    tex3.setup_3d_texture(1, 1, 1, core.Texture.T_int, core.Texture.F_r32i)
    tex3.set_ram_image(pack('i', 5))
    preamble = '\n    uniform isampler1D tex1;\n    uniform isampler2D tex2;\n    uniform isampler3D tex3;\n    '
    code = '\n    assert(texelFetch(tex1, 0, 0) == ivec4(0, 1, 2, 3));\n    assert(texelFetch(tex2, ivec2(0, 0), 0) == ivec4(4, 0, 0, 1));\n    assert(texelFetch(tex3, ivec3(0, 0, 0), 0) == ivec4(5, 0, 0, 1));\n    '
    run_glsl_test(gsg, code, preamble, {'tex1': tex1, 'tex2': tex2, 'tex3': tex3})

def test_glsl_usampler(gsg):
    if False:
        return 10
    from struct import pack
    tex1 = core.Texture('')
    tex1.setup_1d_texture(1, core.Texture.T_unsigned_byte, core.Texture.F_rgba8i)
    tex1.set_ram_image(pack('BBBB', 0, 1, 2, 3))
    tex2 = core.Texture('')
    tex2.setup_2d_texture(1, 1, core.Texture.T_unsigned_short, core.Texture.F_r16i)
    tex2.set_ram_image(pack('H', 4))
    tex3 = core.Texture('')
    tex3.setup_3d_texture(1, 1, 1, core.Texture.T_unsigned_int, core.Texture.F_r32i)
    tex3.set_ram_image(pack('I', 5))
    preamble = '\n    uniform usampler1D tex1;\n    uniform usampler2D tex2;\n    uniform usampler3D tex3;\n    '
    code = '\n    assert(texelFetch(tex1, 0, 0) == uvec4(0, 1, 2, 3));\n    assert(texelFetch(tex2, ivec2(0, 0), 0) == uvec4(4, 0, 0, 1));\n    assert(texelFetch(tex3, ivec3(0, 0, 0), 0) == uvec4(5, 0, 0, 1));\n    '
    run_glsl_test(gsg, code, preamble, {'tex1': tex1, 'tex2': tex2, 'tex3': tex3})

def test_glsl_image(gsg):
    if False:
        i = 10
        return i + 15
    tex1 = core.Texture('')
    tex1.setup_1d_texture(1, core.Texture.T_unsigned_byte, core.Texture.F_rgba8)
    tex1.set_clear_color((0, 2 / 255.0, 1, 1))
    tex2 = core.Texture('')
    tex2.setup_2d_texture(1, 1, core.Texture.T_float, core.Texture.F_rgba32)
    tex2.set_clear_color((1.0, 2.0, -3.14, 0.0))
    preamble = '\n    layout(rgba8) uniform image1D tex1;\n    layout(rgba32f) uniform image2D tex2;\n    '
    code = '\n    assert(imageLoad(tex1, 0) == vec4(0, 2 / 255.0, 1, 1));\n    assert(imageLoad(tex2, ivec2(0, 0)) == vec4(1.0, 2.0, -3.14, 0.0));\n    '
    run_glsl_test(gsg, code, preamble, {'tex1': tex1, 'tex2': tex2})

def test_glsl_iimage(gsg):
    if False:
        print('Hello World!')
    from struct import pack
    tex1 = core.Texture('')
    tex1.setup_1d_texture(1, core.Texture.T_byte, core.Texture.F_rgba8i)
    tex1.set_ram_image(pack('bbbb', 0, 1, 2, 3))
    tex2 = core.Texture('')
    tex2.setup_2d_texture(1, 1, core.Texture.T_short, core.Texture.F_r16i)
    tex2.set_ram_image(pack('h', 4))
    tex3 = core.Texture('')
    tex3.setup_3d_texture(1, 1, 1, core.Texture.T_int, core.Texture.F_r32i)
    tex3.set_ram_image(pack('i', 5))
    preamble = '\n    layout(rgba8i) uniform iimage1D tex1;\n    layout(r16i) uniform iimage2D tex2;\n    layout(r32i) uniform iimage3D tex3;\n    '
    code = '\n    assert(imageLoad(tex1, 0) == ivec4(0, 1, 2, 3));\n    assert(imageLoad(tex2, ivec2(0, 0)) == ivec4(4, 0, 0, 1));\n    assert(imageLoad(tex3, ivec3(0, 0, 0)) == ivec4(5, 0, 0, 1));\n    '
    run_glsl_test(gsg, code, preamble, {'tex1': tex1, 'tex2': tex2, 'tex3': tex3})

def test_glsl_uimage(gsg):
    if False:
        i = 10
        return i + 15
    from struct import pack
    tex1 = core.Texture('')
    tex1.setup_1d_texture(1, core.Texture.T_unsigned_byte, core.Texture.F_rgba8i)
    tex1.set_ram_image(pack('BBBB', 0, 1, 2, 3))
    tex2 = core.Texture('')
    tex2.setup_2d_texture(1, 1, core.Texture.T_unsigned_short, core.Texture.F_r16i)
    tex2.set_ram_image(pack('H', 4))
    tex3 = core.Texture('')
    tex3.setup_3d_texture(1, 1, 1, core.Texture.T_unsigned_int, core.Texture.F_r32i)
    tex3.set_ram_image(pack('I', 5))
    preamble = '\n    layout(rgba8ui) uniform uimage1D tex1;\n    layout(r16ui) uniform uimage2D tex2;\n    layout(r32ui) uniform uimage3D tex3;\n    '
    code = '\n    assert(imageLoad(tex1, 0) == uvec4(0, 1, 2, 3));\n    assert(imageLoad(tex2, ivec2(0, 0)) == uvec4(4, 0, 0, 1));\n    assert(imageLoad(tex3, ivec3(0, 0, 0)) == uvec4(5, 0, 0, 1));\n    '
    run_glsl_test(gsg, code, preamble, {'tex1': tex1, 'tex2': tex2, 'tex3': tex3})

def test_glsl_ssbo(gsg):
    if False:
        print('Hello World!')
    from struct import pack
    num1 = pack('<i', 1234567)
    num2 = pack('<i', -1234567)
    buffer1 = core.ShaderBuffer('buffer1', num1, core.GeomEnums.UH_static)
    buffer2 = core.ShaderBuffer('buffer2', num2, core.GeomEnums.UH_static)
    preamble = '\n    layout(std430, binding=0) buffer buffer1 {\n        int value1;\n    };\n    layout(std430, binding=1) buffer buffer2 {\n        int value2;\n    };\n    '
    code = '\n    assert(value1 == 1234567);\n    assert(value2 == -1234567);\n    '
    run_glsl_test(gsg, code, preamble, {'buffer1': buffer1, 'buffer2': buffer2}, exts={'GL_ARB_shader_storage_buffer_object', 'GL_ARB_uniform_buffer_object', 'GL_ARB_shading_language_420pack'})

def test_glsl_int(gsg):
    if False:
        for i in range(10):
            print('nop')
    inputs = dict(zero=0, intmax=2147483647, intmin=-2147483647)
    preamble = '\n    uniform int zero;\n    uniform int intmax;\n    uniform int intmin;\n    '
    code = '\n    assert(zero == 0);\n    assert(intmax == 0x7fffffff);\n    assert(intmin == -0x7fffffff);\n    '
    run_glsl_test(gsg, code, preamble, inputs)

def test_glsl_uint(gsg):
    if False:
        i = 10
        return i + 15
    inputs = dict(zero=0, intmax=2147483647)
    preamble = '\n    uniform uint zero;\n    uniform uint intmax;\n    '
    code = '\n    assert(zero == 0u);\n    assert(intmax == 0x7fffffffu);\n    '
    run_glsl_test(gsg, code, preamble, inputs)

def test_glsl_bool(gsg):
    if False:
        return 10
    flags = dict(flag1=False, flag2=0, flag3=0.0, flag4=True, flag5=1, flag6=3)
    preamble = '\n    uniform bool flag1;\n    uniform bool flag2;\n    uniform bool flag3;\n    uniform bool flag4;\n    uniform bool flag5;\n    uniform bool flag6;\n    '
    code = '\n    assert(!flag1);\n    assert(!flag2);\n    assert(!flag3);\n    assert(flag4);\n    assert(flag5);\n    assert(flag6);\n    '
    run_glsl_test(gsg, code, preamble, flags)

def test_glsl_mat3(gsg):
    if False:
        while True:
            i = 10
    param1 = core.LMatrix4(core.LMatrix3(1, 2, 3, 4, 5, 6, 7, 8, 9))
    param2 = core.NodePath('param2')
    param2.set_mat(core.LMatrix3(10, 11, 12, 13, 14, 15, 16, 17, 18))
    preamble = '\n    uniform mat3 param1;\n    uniform mat3 param2;\n    '
    code = '\n    assert(param1[0] == vec3(1, 2, 3));\n    assert(param1[1] == vec3(4, 5, 6));\n    assert(param1[2] == vec3(7, 8, 9));\n    assert(param2[0] == vec3(10, 11, 12));\n    assert(param2[1] == vec3(13, 14, 15));\n    assert(param2[2] == vec3(16, 17, 18));\n    '
    run_glsl_test(gsg, code, preamble, {'param1': param1, 'param2': param2})

def test_glsl_mat4(gsg):
    if False:
        i = 10
        return i + 15
    param1 = core.LMatrix4(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16)
    param2 = core.NodePath('param2')
    param2.set_mat(core.LMatrix4(17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32))
    preamble = '\n    uniform mat4 param1;\n    uniform mat4 param2;\n    '
    code = '\n    assert(param1[0] == vec4(1, 2, 3, 4));\n    assert(param1[1] == vec4(5, 6, 7, 8));\n    assert(param1[2] == vec4(9, 10, 11, 12));\n    assert(param1[3] == vec4(13, 14, 15, 16));\n    assert(param2[0] == vec4(17, 18, 19, 20));\n    assert(param2[1] == vec4(21, 22, 23, 24));\n    assert(param2[2] == vec4(25, 26, 27, 28));\n    assert(param2[3] == vec4(29, 30, 31, 32));\n    '
    run_glsl_test(gsg, code, preamble, {'param1': param1, 'param2': param2})

def test_glsl_pta_int(gsg):
    if False:
        while True:
            i = 10
    pta = core.PTA_int((0, 1, 2, 3))
    preamble = '\n    uniform int pta[4];\n    '
    code = '\n    assert(pta[0] == 0);\n    assert(pta[1] == 1);\n    assert(pta[2] == 2);\n    assert(pta[3] == 3);\n    '
    run_glsl_test(gsg, code, preamble, {'pta': pta})

def test_glsl_pta_ivec4(gsg):
    if False:
        for i in range(10):
            print('nop')
    pta = core.PTA_LVecBase4i(((0, 1, 2, 3), (4, 5, 6, 7)))
    preamble = '\n    uniform ivec4 pta[2];\n    '
    code = '\n    assert(pta[0] == ivec4(0, 1, 2, 3));\n    assert(pta[1] == ivec4(4, 5, 6, 7));\n    '
    run_glsl_test(gsg, code, preamble, {'pta': pta})

def test_glsl_pta_mat4(gsg):
    if False:
        i = 10
        return i + 15
    pta = core.PTA_LMatrix4f(((0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15), (16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31)))
    preamble = '\n    uniform mat4 pta[2];\n    '
    code = '\n    assert(pta[0][0] == vec4(0, 1, 2, 3));\n    assert(pta[0][1] == vec4(4, 5, 6, 7));\n    assert(pta[0][2] == vec4(8, 9, 10, 11));\n    assert(pta[0][3] == vec4(12, 13, 14, 15));\n    assert(pta[1][0] == vec4(16, 17, 18, 19));\n    assert(pta[1][1] == vec4(20, 21, 22, 23));\n    assert(pta[1][2] == vec4(24, 25, 26, 27));\n    assert(pta[1][3] == vec4(28, 29, 30, 31));\n    '
    run_glsl_test(gsg, code, preamble, {'pta': pta})

def test_glsl_param_vec4(gsg):
    if False:
        while True:
            i = 10
    param = core.ParamVecBase4((0, 1, 2, 3))
    preamble = '\n    uniform vec4 param;\n    '
    code = '\n    assert(param.x == 0.0);\n    assert(param.y == 1.0);\n    assert(param.z == 2.0);\n    assert(param.w == 3.0);\n    '
    run_glsl_test(gsg, code, preamble, {'param': param})

def test_glsl_param_ivec4(gsg):
    if False:
        for i in range(10):
            print('nop')
    param = core.ParamVecBase4i((0, 1, 2, 3))
    preamble = '\n    uniform ivec4 param;\n    '
    code = '\n    assert(param.x == 0);\n    assert(param.y == 1);\n    assert(param.z == 2);\n    assert(param.w == 3);\n    '
    run_glsl_test(gsg, code, preamble, {'param': param})

def test_glsl_write_extract_image_buffer(gsg):
    if False:
        for i in range(10):
            print('nop')
    tex1 = core.Texture('tex1')
    tex1.set_clear_color(0)
    tex1.setup_buffer_texture(1, core.Texture.T_unsigned_int, core.Texture.F_r32i, core.GeomEnums.UH_static)
    tex2 = core.Texture('tex2')
    tex2.set_clear_color(0)
    tex2.setup_buffer_texture(1, core.Texture.T_int, core.Texture.F_r32i, core.GeomEnums.UH_static)
    preamble = '\n    layout(r32ui) uniform uimageBuffer tex1;\n    layout(r32i) uniform iimageBuffer tex2;\n    '
    code = '\n    assert(imageLoad(tex1, 0).r == 0u);\n    assert(imageLoad(tex2, 0).r == 0);\n    imageStore(tex1, 0, uvec4(123));\n    imageStore(tex2, 0, ivec4(-456));\n    memoryBarrier();\n    assert(imageLoad(tex1, 0).r == 123u);\n    assert(imageLoad(tex2, 0).r == -456);\n    '
    run_glsl_test(gsg, code, preamble, {'tex1': tex1, 'tex2': tex2})
    engine = core.GraphicsEngine.get_global_ptr()
    assert engine.extract_texture_data(tex1, gsg)
    assert engine.extract_texture_data(tex2, gsg)
    assert struct.unpack('I', tex1.get_ram_image()) == (123,)
    assert struct.unpack('i', tex2.get_ram_image()) == (-456,)

def test_glsl_compile_error(gsg):
    if False:
        return 10
    'Test getting compile errors from bad shaders'
    suffix = ''
    if (gsg.driver_shader_version_major, gsg.driver_shader_version_minor) < (1, 50):
        suffix = '_legacy'
    vert_path = core.Filename(SHADERS_DIR, 'glsl_bad' + suffix + '.vert')
    frag_path = core.Filename(SHADERS_DIR, 'glsl_simple' + suffix + '.frag')
    run_glsl_compile_check(gsg, vert_path, frag_path, expect_fail=True)

def test_glsl_from_file(gsg):
    if False:
        print('Hello World!')
    'Test compiling GLSL shaders from files'
    suffix = ''
    if (gsg.driver_shader_version_major, gsg.driver_shader_version_minor) < (1, 50):
        suffix = '_legacy'
    vert_path = core.Filename(SHADERS_DIR, 'glsl_simple' + suffix + '.vert')
    frag_path = core.Filename(SHADERS_DIR, 'glsl_simple' + suffix + '.frag')
    run_glsl_compile_check(gsg, vert_path, frag_path)

def test_glsl_includes(gsg):
    if False:
        for i in range(10):
            print('nop')
    'Test preprocessing includes in GLSL shaders'
    suffix = ''
    if (gsg.driver_shader_version_major, gsg.driver_shader_version_minor) < (1, 50):
        suffix = '_legacy'
    vert_path = core.Filename(SHADERS_DIR, 'glsl_include' + suffix + '.vert')
    frag_path = core.Filename(SHADERS_DIR, 'glsl_simple' + suffix + '.frag')
    run_glsl_compile_check(gsg, vert_path, frag_path)

def test_glsl_includes_angle_nodir(gsg):
    if False:
        for i in range(10):
            print('nop')
    'Test preprocessing includes with angle includes without model-path'
    suffix = ''
    if (gsg.driver_shader_version_major, gsg.driver_shader_version_minor) < (1, 50):
        suffix = '_legacy'
    vert_path = core.Filename(SHADERS_DIR, 'glsl_include_angle' + suffix + '.vert')
    frag_path = core.Filename(SHADERS_DIR, 'glsl_simple' + suffix + '.frag')
    assert core.Shader.load(core.Shader.SL_GLSL, vert_path, frag_path) is None

@pytest.fixture
def with_current_dir_on_model_path():
    if False:
        print('Hello World!')
    model_path = core.get_model_path()
    model_path.prepend_directory(core.Filename.from_os_specific(os.path.dirname(__file__)))
    yield
    model_path.clear_local_value()

def test_glsl_includes_angle_withdir(gsg, with_current_dir_on_model_path):
    if False:
        print('Hello World!')
    'Test preprocessing includes with angle includes with model-path'
    suffix = ''
    if (gsg.driver_shader_version_major, gsg.driver_shader_version_minor) < (1, 50):
        suffix = '_legacy'
    vert_path = core.Filename(SHADERS_DIR, 'glsl_include_angle' + suffix + '.vert')
    frag_path = core.Filename(SHADERS_DIR, 'glsl_simple' + suffix + '.frag')
    run_glsl_compile_check(gsg, vert_path, frag_path)