import unittest
import numpy as np
from vispy import gloo, app
from vispy.gloo.program import Program
from vispy.testing import run_tests_if_main, requires_application
from vispy.gloo.context import set_current_canvas, forget_canvas

class DummyParser(gloo.glir.BaseGlirParser):

    @property
    def shader_compatibility(self):
        if False:
            i = 10
            return i + 15
        return 'desktop'

    def parse(self, commands):
        if False:
            for i in range(10):
                print('nop')
        pass

class DummyCanvas:

    def __init__(self):
        if False:
            return 10
        self.context = gloo.context.GLContext()
        self.context.shared.parser = DummyParser()
        self.context.glir.flush = lambda *args: None

class ProgramTest(unittest.TestCase):

    def test_init(self):
        if False:
            while True:
                i = 10
        from vispy.gloo.program import VertexShader, FragmentShader
        program = Program()
        assert program._user_variables == {}
        assert program._code_variables == {}
        assert program._pending_variables == {}
        assert program.shaders[0] is None
        assert program.shaders[1] is None
        program = Program('A', 'B')
        assert isinstance(program.shaders[0], VertexShader)
        assert program.shaders[0].code == 'A'
        assert isinstance(program.shaders[1], FragmentShader)
        assert program.shaders[1].code == 'B'
        self.assertRaises(ValueError, Program, 'A', None)
        self.assertRaises(ValueError, Program, None, 'B')
        self.assertRaises(ValueError, Program, 3, 'B')
        self.assertRaises(ValueError, Program, 3, None)
        self.assertRaises(ValueError, Program, 'A', 3)
        self.assertRaises(ValueError, Program, None, 3)
        self.assertRaises(ValueError, Program, '', '')
        self.assertRaises(ValueError, Program, 'foo', '')
        self.assertRaises(ValueError, Program, '', 'foo')

    def test_setting_shaders(self):
        if False:
            i = 10
            return i + 15
        from vispy.gloo.program import VertexShader, FragmentShader
        program = Program('A', 'B')
        assert isinstance(program.shaders[0], VertexShader)
        assert program.shaders[0].code == 'A'
        assert isinstance(program.shaders[1], FragmentShader)
        assert program.shaders[1].code == 'B'
        program.set_shaders('C', 'D')
        assert program.shaders[0].code == 'C'
        assert program.shaders[1].code == 'D'

    @requires_application()
    def test_error(self):
        if False:
            i = 10
            return i + 15
        vert = '\n        void main() {\n            vec2 xy;\n            error on this line\n            vec2 ab;\n        }\n        '
        frag = 'void main() { glFragColor = vec4(1, 1, 1, 1); }'
        with app.Canvas() as c:
            program = Program(vert, frag)
            try:
                program._glir.flush(c.context.shared.parser)
            except Exception as err:
                assert 'error on this line' in str(err)
            else:
                raise Exception('Compile program should have failed.')

    def test_uniform(self):
        if False:
            return 10
        program = Program('uniform float A[10];', 'foo')
        assert ('uniform_array', 'float', 'A') in program.variables
        assert len(program.variables) == 11
        self.assertRaises(ValueError, program.__setitem__, 'A', np.ones((9, 1)))
        program['A'] = np.ones((10, 1))
        program['A[0]'] = 0
        assert 'A[0]' in program._user_variables
        assert 'A[0]' not in program._pending_variables
        program = Program('uniform float A;', 'uniform float A; uniform vec4 B;')
        assert ('uniform', 'float', 'A') in program.variables
        assert ('uniform', 'vec4', 'B') in program.variables
        assert len(program.variables) == 2
        program['A'] = 3.0
        assert isinstance(program['A'], np.ndarray)
        assert program['A'] == 3.0
        assert 'A' in program._user_variables
        program['B'] = (1.0, 2.0, 3.0, 4.0)
        assert isinstance(program['B'], np.ndarray)
        assert all(program['B'] == np.array((1.0, 2.0, 3.0, 4.0), np.float32))
        assert 'B' in program._user_variables
        program['C'] = (1.0, 2.0)
        assert program['C'] == (1.0, 2.0)
        assert 'C' not in program._user_variables
        assert 'C' in program._pending_variables
        program.set_shaders('uniform sampler1D T1;\n                            uniform sampler2D T2;\n                            uniform sampler3D T3;', 'f')
        program['T1'] = np.zeros((10,), np.float32)
        program['T2'] = np.zeros((10, 10), np.float32)
        program['T3'] = np.zeros((10, 10, 10), np.float32)
        assert isinstance(program['T1'], gloo.Texture1D)
        assert isinstance(program['T2'], gloo.Texture2D)
        assert isinstance(program['T3'], gloo.Texture3D)
        tex = gloo.Texture2D((10, 10))
        program['T2'] = tex
        assert program['T2'] is tex
        program['T2'] = np.zeros((10, 10), np.float32)
        assert program['T2'] is tex
        program.set_shaders('uniform float A; uniform vec2 C;', 'uniform float A; uniform vec4 B;')
        assert isinstance(program['C'], np.ndarray)
        assert all(program['C'] == np.array((1.0, 2.0), np.float32))
        assert 'C' in program._user_variables
        assert 'C' not in program._pending_variables
        self.assertRaises(ValueError, program.__setitem__, 'A', (1.0, 2.0))
        self.assertRaises(ValueError, program.__setitem__, 'B', (1.0, 2.0))
        self.assertRaises(ValueError, program.__setitem__, 'C', 1.0)
        program['D'] = (1.0, 2.0)
        self.assertRaises(ValueError, program.set_shaders, '', 'uniform vec3 D;')

    def test_attributes(self):
        if False:
            print('Hello World!')
        program = Program('attribute float A; attribute vec4 B;', 'foo')
        assert ('attribute', 'float', 'A') in program.variables
        assert ('attribute', 'vec4', 'B') in program.variables
        assert len(program.variables) == 2
        from vispy.gloo import VertexBuffer
        vbo = VertexBuffer()
        program['A'] = vbo
        assert program['A'] == vbo
        assert 'A' in program._user_variables
        assert program._user_variables['A'] is vbo
        program['A'] = np.zeros((10,), np.float32)
        assert program._user_variables['A'] is vbo
        program['B'] = np.zeros((10, 4), np.float32)
        assert isinstance(program._user_variables['B'], VertexBuffer)
        vbo = VertexBuffer()
        program['C'] = vbo
        assert program['C'] == vbo
        assert 'C' not in program._user_variables
        assert 'C' in program._pending_variables
        program.set_shaders('attribute float A; attribute vec2 C;', 'foo')
        assert program['C'] == vbo
        assert 'C' in program._user_variables
        assert 'C' not in program._pending_variables
        self.assertRaises(ValueError, program.__setitem__, 'A', 'asddas')
        program['D'] = ''
        self.assertRaises(ValueError, program.set_shaders, 'attribute vec3 D;', '')
        program.set_shaders('attribute float A; attribute vec2 C;', 'foo')
        program['A'] = 1.0
        assert program['A'] == 1.0
        program['C'] = (1.0, 2.0)
        assert all(program['C'] == np.array((1.0, 2.0), np.float32))
        self.assertRaises(ValueError, program.__setitem__, 'A', (1.0, 2.0))
        self.assertRaises(ValueError, program.__setitem__, 'C', 1.0)
        self.assertRaises(ValueError, program.bind, 'notavertexbuffer')
        program = Program('attribute vec2 C;', 'foo')
        self.assertRaises(ValueError, program.__setitem__, 'C', np.ones((2, 10), np.float32))
        program['C'] = np.ones((10, 2), np.float32)
        self.assertRaises(ValueError, program.__setitem__, 'C', np.ones((2, 10), np.float32))

    def test_vbo(self):
        if False:
            for i in range(10):
                print('nop')
        program = Program('attribute float a; attribute vec2 b;', 'foo', 10)
        assert program._count == 10
        assert ('attribute', 'float', 'a') in program.variables
        assert ('attribute', 'vec2', 'b') in program.variables
        program['a'] = np.ones((10,), np.float32)
        assert np.all(program._buffer['a'] == 1)

    def test_varyings(self):
        if False:
            while True:
                i = 10
        program = Program('varying float A; const vec4 B;', 'foo')
        assert ('varying', 'float', 'A') in program.variables
        assert ('const', 'vec4', 'B') in program.variables
        self.assertRaises(KeyError, program.__setitem__, 'A', 3.0)
        self.assertRaises(KeyError, program.__setitem__, 'B', (1.0, 2.0, 3.0))
        self.assertRaises(KeyError, program.__getitem__, 'fooo')

    def test_type_aliases(self):
        if False:
            return 10
        program = Program('in bool A; out float B;', 'foo')
        assert ('attribute', 'bool', 'A') in program.variables
        assert ('varying', 'float', 'B') in program.variables

    def test_draw(self):
        if False:
            for i in range(10):
                print('nop')
        program = Program('attribute float A;', 'uniform float foo')
        program['A'] = np.zeros((10,), np.float32)
        dummy_canvas = DummyCanvas()
        glir = dummy_canvas.context.glir
        set_current_canvas(dummy_canvas)
        try:
            program.draw('triangles')
            glir_cmd = glir.clear()[-1]
            assert glir_cmd[0] == 'DRAW'
            assert len(glir_cmd[-2]) == 2
            indices = gloo.IndexBuffer(np.zeros(10, dtype=np.uint8))
            program.draw('triangles', indices)
            glir_cmd = glir.clear()[-1]
            assert glir_cmd[0] == 'DRAW'
            assert len(glir_cmd[-2]) == 3
            self.assertRaises(ValueError, program.draw, 'nogeometricshape')
            self.assertRaises(TypeError, program.draw, 'triangles', 'notindex')
            program = Program('attribute float A;', 'uniform float foo')
            self.assertRaises(RuntimeError, program.draw, 'triangles')
            program = Program('attribute float A; attribute float B;', 'foo')
            program['A'] = np.zeros((10,), np.float32)
            program['B'] = np.zeros((11,), np.float32)
            self.assertRaises(RuntimeError, program.draw, 'triangles')
        finally:
            forget_canvas(dummy_canvas)
run_tests_if_main()