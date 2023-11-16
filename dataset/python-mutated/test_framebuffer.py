from vispy.testing import run_tests_if_main, assert_raises
from vispy import gloo
from vispy.gloo import FrameBuffer, RenderBuffer

def test_renderbuffer():
    if False:
        return 10
    assert_raises(ValueError, RenderBuffer)
    R = RenderBuffer((10, 20))
    assert R.shape == (10, 20)
    assert R.format is None
    R = RenderBuffer((10, 20), 'color')
    assert R.shape == (10, 20)
    assert R.format == 'color'
    glir_cmds = R._glir.clear()
    assert len(glir_cmds) == 2
    assert glir_cmds[0][0] == 'CREATE'
    assert glir_cmds[1][0] == 'SIZE'
    assert RenderBuffer((10, 20), 'depth').format == 'depth'
    assert RenderBuffer((10, 20), 'stencil').format == 'stencil'
    R.resize((9, 9), 'depth')
    assert R.shape == (9, 9)
    assert R.format == 'depth'
    R.resize((8, 8), 'stencil')
    assert R.shape == (8, 8)
    assert R.format == 'stencil'
    assert_raises(ValueError, R.resize, (9, 9), 'no_format')
    assert_raises(ValueError, R.resize, (9, 9), [])
    R = RenderBuffer((10, 20), 'color', False)
    assert_raises(RuntimeError, R.resize, (9, 9), 'color')
    F = FrameBuffer()
    R = RenderBuffer((9, 9))
    F.color_buffer = R
    assert F.color_buffer is R
    assert R.format == 'color'
    F.depth_buffer = RenderBuffer((9, 9))
    assert F.depth_buffer.format == 'depth'
    F.stencil_buffer = RenderBuffer((9, 9))
    assert F.stencil_buffer.format == 'stencil'

def test_framebuffer():
    if False:
        print('Hello World!')
    F = FrameBuffer()
    glir_cmds = F._glir.clear()
    assert len(glir_cmds) == 1
    glir_cmds[0][0] == 'CREATE'
    F.activate()
    glir_cmd = F._glir.clear()[-1]
    assert glir_cmd[0] == 'FRAMEBUFFER'
    assert glir_cmd[2] is True
    F.deactivate()
    glir_cmd = F._glir.clear()[-1]
    assert glir_cmd[0] == 'FRAMEBUFFER'
    assert glir_cmd[2] is False
    with F:
        pass
    glir_cmds = F._glir.clear()
    assert len(glir_cmds) == 2
    assert glir_cmds[0][0] == 'FRAMEBUFFER'
    assert glir_cmds[1][0] == 'FRAMEBUFFER'
    assert glir_cmds[0][2] is True and glir_cmds[1][2] is False
    R = RenderBuffer((3, 3))
    F = FrameBuffer(R)
    assert F.color_buffer is R
    R2 = RenderBuffer((3, 3))
    F.color_buffer = R2
    assert F.color_buffer is R2
    F = FrameBuffer()
    assert_raises(TypeError, FrameBuffer.color_buffer.fset, F, 'FOO')
    assert_raises(TypeError, FrameBuffer.color_buffer.fset, F, [])
    assert_raises(TypeError, FrameBuffer.depth_buffer.fset, F, 'FOO')
    assert_raises(TypeError, FrameBuffer.stencil_buffer.fset, F, 'FOO')
    color_buffer = RenderBuffer((9, 9), 'color')
    assert_raises(ValueError, FrameBuffer.depth_buffer.fset, F, color_buffer)
    F.color_buffer = None
    R1 = RenderBuffer((3, 3))
    R2 = RenderBuffer((3, 3))
    R3 = RenderBuffer((3, 3))
    F = FrameBuffer(R1, R2, R3)
    assert F.shape == R1.shape
    assert R1.format == 'color'
    assert R2.format == 'depth'
    assert R3.format == 'stencil'
    F.resize((10, 10))
    assert F.shape == (10, 10)
    assert F.shape == R1.shape
    assert F.shape == R2.shape
    assert F.shape == R3.shape
    assert R1.format == 'color'
    assert R2.format == 'depth'
    assert R3.format == 'stencil'
    F.color_buffer = None
    assert F.shape == (10, 10)
    F.depth_buffer = None
    assert F.shape == (10, 10)
    F.stencil_buffer = None
    assert_raises(RuntimeError, FrameBuffer.shape.fget, F)
    T = gloo.Texture2D((20, 30))
    R = RenderBuffer(T.shape)
    assert T.format == 'luminance'
    F = FrameBuffer(T, R)
    assert F.shape == T.shape[:2]
    assert F.shape == R.shape
    assert T.format == 'luminance'
    assert R.format == 'depth'
    F.resize((10, 10))
    assert F.shape == (10, 10)
    assert T.shape == (10, 10, 1)
    assert F.shape == R.shape
    assert T.format == 'luminance'
    assert R.format == 'depth'
    T = gloo.Texture2D((20, 30, 3))
    R = RenderBuffer(T.shape)
    assert T.format == 'rgb'
    F = FrameBuffer(T, R)
    assert F.shape == T.shape[:2]
    assert F.shape == R.shape
    assert T.format == 'rgb'
    assert R.format == 'depth'
    F.resize((10, 10))
    assert F.shape == (10, 10)
    assert T.shape == (10, 10, 3)
    assert F.shape == R.shape
    assert T.format == 'rgb'
    assert R.format == 'depth'
    T1 = gloo.Texture2D((20, 30, 3))
    T2 = gloo.Texture2D((20, 30, 1))
    assert T1.format == 'rgb'
    assert T2.format == 'luminance'
    F = FrameBuffer(T1, T2)
    assert F.shape == T1.shape[:2]
    assert F.shape == T2.shape[:2]
    assert T1.format == 'rgb'
    assert T2.format == 'luminance'
    F.resize((10, 10))
    assert F.shape == (10, 10)
    assert T1.shape == (10, 10, 3)
    assert T2.shape == (10, 10, 1)
    assert T1.format == 'rgb'
    assert T2.format == 'luminance'
    assert_raises(ValueError, F.resize, (9, 9, 1))
    assert_raises(ValueError, F.resize, (9,))
    assert_raises(ValueError, F.resize, 'FOO')
run_tests_if_main()