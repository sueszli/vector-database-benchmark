import json
import tempfile
from unittest import mock
from vispy import config
from vispy.app import Canvas
from vispy.gloo import glir
from vispy.testing import requires_application, requires_pyopengl, run_tests_if_main
import numpy as np

def test_queue():
    if False:
        i = 10
        return i + 15
    q = glir.GlirQueue()
    parser = glir.GlirParser()
    N = 5
    for i in range(N):
        q.command('FOO', 'BAR', i)
    cmds = q.clear()
    for i in range(N):
        assert cmds[i] == ('FOO', 'BAR', i)
    cmds1 = [('DATA', 1), ('SIZE', 1), ('FOO', 1), ('SIZE', 1), ('FOO', 1), ('DATA', 1), ('DATA', 1)]
    cmds2 = [c[0] for c in q._shared._filter(cmds1, parser)]
    assert cmds2 == ['FOO', 'SIZE', 'FOO', 'DATA', 'DATA']
    cmds1 = [('DATA', 1), ('SIZE', 1), ('FOO', 1), ('SIZE', 2), ('SIZE', 2), ('DATA', 2), ('SIZE', 1), ('FOO', 1), ('DATA', 1), ('DATA', 1)]
    cmds2 = q._shared._filter(cmds1, parser)
    assert cmds2 == [('FOO', 1), ('SIZE', 2), ('DATA', 2), ('SIZE', 1), ('FOO', 1), ('DATA', 1), ('DATA', 1)]
    shader1 = '\n        precision highp float;uniform mediump vec4 u_foo;uniform vec4 u_bar;\n        '.strip().replace(';', ';\n')
    shader2 = glir.convert_shader('desktop', shader1)
    assert 'highp' not in shader2
    assert 'mediump' not in shader2
    assert 'precision' not in shader2
    shader3 = glir.convert_shader('es2', shader2)
    assert 'precision highp float;' in shader3
    assert shader3.startswith('precision')
    shader4 = '\n        #version 100; precision highp float;uniform mediump vec4 u_foo;uniform vec4 u_bar;\n        '.strip().replace(';', ';\n')
    shader5 = glir.convert_shader('es2', shader4)
    assert 'precision highp float;' in shader5
    assert shader3.startswith('precision')

@requires_application()
def test_log_parser():
    if False:
        return 10
    'Test GLIR log parsing'
    glir_file = tempfile.TemporaryFile(mode='r+')
    config.update(glir_file=glir_file)
    with Canvas() as c:
        c.context.set_clear_color('white')
        c.context.clear()
    glir_file.seek(0)
    lines = glir_file.read().split(',\n')
    assert lines[0][0] == '['
    lines[0] = lines[0][1:]
    assert lines[-1][-1] == ']'
    lines[-1] = lines[-1][:-1]
    i = 0
    expected = json.dumps(['CURRENT', 0, 1])
    assert len(lines[i]) >= len(expected)
    expected = expected.split('1')
    assert lines[i].startswith(expected[0])
    assert lines[i].endswith(expected[1])
    assert int(lines[i][len(expected[0]):-len(expected[1])]) is not None
    while lines[i].startswith('["CURRENT",'):
        i += 1
        if lines[i] == json.dumps(['FUNC', 'colorMask', False, False, False, True]):
            i += 4
    assert lines[i] == json.dumps(['FUNC', 'clearColor', 1.0, 1.0, 1.0, 1.0])
    i += 1
    assert lines[i] == json.dumps(['FUNC', 'clear', 17664])
    i += 1
    assert lines[i] == json.dumps(['FUNC', 'finish'])
    i += 1
    config.update(glir_file='')
    glir_file.close()

@requires_application()
def test_capabilities():
    if False:
        return 10
    'Test GLIR capability reporting'
    with Canvas() as c:
        capabilities = c.context.shared.parser.capabilities
        assert capabilities['max_texture_size'] is not None
        assert capabilities['gl_version'] != 'unknown'

@requires_pyopengl()
@mock.patch('vispy.gloo.glir._check_pyopengl_3D')
@mock.patch('vispy.gloo.glir.gl')
def test_texture1d_alignment(gl, check3d):
    if False:
        return 10
    'Test that textures set unpack alignment properly.\n\n    See https://github.com/vispy/vispy/pull/1758\n\n    '
    from ..glir import GlirTexture1D
    check3d.return_value = check3d
    t = GlirTexture1D(mock.MagicMock(), 3)
    shape = (393, 1)
    t.set_size(shape, 'luminance', 'luminance')
    t.set_data((0,), np.zeros(shape, np.float32))
    gl.glPixelStorei.assert_not_called()
    gl.glPixelStorei.reset_mock()
    t.set_data((0,), np.zeros(shape, np.uint8))
    gl.glPixelStorei.assert_has_calls([mock.call(gl.GL_UNPACK_ALIGNMENT, 1), mock.call(gl.GL_UNPACK_ALIGNMENT, 4)])
    gl.glPixelStorei.reset_mock()
    shape = (394, 1)
    t.set_size(shape, 'luminance', 'luminance')
    t.set_data((0,), np.zeros(shape, np.float32))
    gl.glPixelStorei.assert_not_called()
    gl.glPixelStorei.reset_mock()
    t.set_data((0,), np.zeros(shape, np.uint8))
    gl.glPixelStorei.assert_has_calls([mock.call(gl.GL_UNPACK_ALIGNMENT, 1), mock.call(gl.GL_UNPACK_ALIGNMENT, 4)])
    gl.glPixelStorei.reset_mock()

@requires_pyopengl()
@mock.patch('vispy.gloo.glir._check_pyopengl_3D')
@mock.patch('vispy.gloo.glir.gl')
def test_texture2d_alignment(gl, check3d):
    if False:
        while True:
            i = 10
    'Test that textures set unpack alignment properly.\n\n    See https://github.com/vispy/vispy/pull/1758\n\n    '
    from ..glir import GlirTexture2D
    check3d.return_value = gl
    t = GlirTexture2D(mock.MagicMock(), 3)
    shape = (296, 393, 1)
    t.set_size(shape, 'luminance', 'luminance')
    t.set_data((0, 0), np.zeros(shape, np.float32))
    gl.glPixelStorei.assert_not_called()
    gl.glPixelStorei.reset_mock()
    t.set_data((0, 0), np.zeros(shape, np.uint8))
    gl.glPixelStorei.assert_has_calls([mock.call(gl.GL_UNPACK_ALIGNMENT, 1), mock.call(gl.GL_UNPACK_ALIGNMENT, 4)])
    gl.glPixelStorei.reset_mock()
    shape = (296, 394, 1)
    t.set_size(shape, 'luminance', 'luminance')
    t.set_data((0, 0), np.zeros(shape, np.float32))
    gl.glPixelStorei.assert_has_calls([mock.call(gl.GL_UNPACK_ALIGNMENT, 8), mock.call(gl.GL_UNPACK_ALIGNMENT, 4)])
    gl.glPixelStorei.reset_mock()
    t.set_data((0, 0), np.zeros(shape, np.uint8))
    gl.glPixelStorei.assert_has_calls([mock.call(gl.GL_UNPACK_ALIGNMENT, 2), mock.call(gl.GL_UNPACK_ALIGNMENT, 4)])
    gl.glPixelStorei.reset_mock()

@requires_pyopengl()
@mock.patch('vispy.gloo.glir._check_pyopengl_3D')
@mock.patch('vispy.gloo.glir.gl')
def test_texture3d_alignment(gl, check3d):
    if False:
        i = 10
        return i + 15
    'Test that textures set unpack alignment properly.\n\n    See https://github.com/vispy/vispy/pull/1758\n\n    '
    from ..glir import GlirTexture3D
    check3d.return_value = gl
    t = GlirTexture3D(mock.MagicMock(), 3)
    shape = (68, 296, 393, 1)
    t.set_size(shape, 'luminance', 'luminance')
    t.set_data((0, 0, 0), np.zeros(shape, np.float32))
    gl.glPixelStorei.assert_not_called()
    gl.glPixelStorei.reset_mock()
    t.set_data((0, 0, 0), np.zeros(shape, np.uint8))
    gl.glPixelStorei.assert_has_calls([mock.call(gl.GL_UNPACK_ALIGNMENT, 1), mock.call(gl.GL_UNPACK_ALIGNMENT, 4)])
    gl.glPixelStorei.reset_mock()
    shape = (68, 296, 394, 1)
    t.set_size(shape, 'luminance', 'luminance')
    t.set_data((0, 0, 0), np.zeros(shape, np.float32))
    gl.glPixelStorei.assert_has_calls([mock.call(gl.GL_UNPACK_ALIGNMENT, 8), mock.call(gl.GL_UNPACK_ALIGNMENT, 4)])
    gl.glPixelStorei.reset_mock()
    t.set_data((0, 0, 0), np.zeros(shape, np.uint8))
    gl.glPixelStorei.assert_has_calls([mock.call(gl.GL_UNPACK_ALIGNMENT, 2), mock.call(gl.GL_UNPACK_ALIGNMENT, 4)])
    gl.glPixelStorei.reset_mock()

@requires_pyopengl()
@mock.patch('vispy.gloo.glir._check_pyopengl_3D')
@mock.patch('vispy.gloo.glir.gl')
def test_texture_cube_alignment(gl, check3d):
    if False:
        i = 10
        return i + 15
    'Test that textures set unpack alignment properly.\n\n    See https://github.com/vispy/vispy/pull/1758\n\n    '
    from ..glir import GlirTextureCube
    check3d.return_value = gl
    t = GlirTextureCube(mock.MagicMock(), 3)
    shape = (68, 296, 393, 1)
    t.set_size(shape, 'luminance', 'luminance')
    t.set_data((0, 0, 0), np.zeros(shape, np.float32))
    gl.glPixelStorei.assert_not_called()
    gl.glPixelStorei.reset_mock()
    t.set_data((0, 0, 0), np.zeros(shape, np.uint8))
    gl.glPixelStorei.assert_has_calls([mock.call(gl.GL_UNPACK_ALIGNMENT, 1), mock.call(gl.GL_UNPACK_ALIGNMENT, 4)])
    gl.glPixelStorei.reset_mock()
    shape = (68, 296, 394, 1)
    t.set_size(shape, 'luminance', 'luminance')
    t.set_data((0, 0, 0), np.zeros(shape, np.float32))
    gl.glPixelStorei.assert_has_calls([mock.call(gl.GL_UNPACK_ALIGNMENT, 8), mock.call(gl.GL_UNPACK_ALIGNMENT, 4)])
    gl.glPixelStorei.reset_mock()
    t.set_data((0, 0, 0), np.zeros(shape, np.uint8))
    gl.glPixelStorei.assert_has_calls([mock.call(gl.GL_UNPACK_ALIGNMENT, 2), mock.call(gl.GL_UNPACK_ALIGNMENT, 4)])
    gl.glPixelStorei.reset_mock()
run_tests_if_main()