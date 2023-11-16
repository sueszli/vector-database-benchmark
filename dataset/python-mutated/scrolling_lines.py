import numpy as np
from .visual import Visual
from .. import gloo

class ScrollingLinesVisual(Visual):
    """Displays many line strips of equal length, with the option to add new
    vertex data to one end of the lines.

    Parameters
    ----------
    n_lines : int
        The number of independent line strips to draw.
    line_size : int
        The number of samples in each line strip.
    dx : float
        The x distance between samples
    color : array-like
        An array of colors to assign to each line strip.
    pos_offset : array-like
        An array of x, y position offsets to apply to each line strip.
    columns : int
        Arrange line strips into a grid with this number of columns. This
        option is not compatible with *pos_offset*.
    cell_size : tuple
        The x, y distance between cells in the grid.
    """
    vertex_code = '\n    attribute vec2 index;  // .x=line_n, .y=vertex_n\n    uniform sampler2D position;\n    uniform sampler1D pos_offset;\n    uniform sampler1D color_tex;\n    \n    uniform vec2 pos_size;  // x=n_lines, y=n_verts_per_line\n    uniform float offset;  // rolling pointer into vertexes\n    uniform float dx;  // x step per sample\n    \n    varying vec2 v_index;\n    varying vec4 v_color;\n    \n    \n    void main() {\n        v_index = vec2(mod(index.y + offset, pos_size.y), index.x);\n        vec2 uv = (v_index + 0.5) / (pos_size.yx);\n        vec4 pos = vec4(index.y * dx, texture2D(position, uv).r, 0, 1);\n        \n        // fetch starting position from texture lookup:\n        pos += vec4(texture1D(pos_offset, (index.x + 0.5) / pos_size.x).rg,\n                              0, 0); \n        \n        gl_Position = $transform(pos);\n        \n        v_color = texture1D(color_tex, (index.x + 0.5) / pos_size.x);\n    }\n    '
    fragment_code = '\n    varying vec2 v_index;\n    varying vec4 v_color;\n    \n    void main() {\n        if (v_index.y - floor(v_index.y) > 0) {\n            discard;\n        }\n        gl_FragColor = $color;\n    }\n    '

    def __init__(self, n_lines, line_size, dx, color=None, pos_offset=None, columns=None, cell_size=None):
        if False:
            while True:
                i = 10
        self._pos_data = None
        self._offset = 0
        self._dx = dx
        data = np.zeros((n_lines, line_size), dtype='float32')
        self._pos_tex = gloo.Texture2D(data, format='luminance', internalformat='r32f')
        self._index_buf = gloo.VertexBuffer()
        self._data_shape = data.shape
        Visual.__init__(self, vcode=self.vertex_code, fcode=self.fragment_code)
        self.shared_program['position'] = self._pos_tex
        self.shared_program['index'] = self._index_buf
        self.shared_program['dx'] = dx
        self.shared_program['pos_size'] = data.shape
        self.shared_program['offset'] = self._offset
        if pos_offset is None:
            rows = int(np.ceil(n_lines / columns))
            pos_offset = np.empty((rows, columns, 3), dtype='float32')
            pos_offset[..., 0] = np.arange(columns)[np.newaxis, :] * cell_size[0]
            pos_offset[..., 1] = np.arange(rows)[:, np.newaxis] * cell_size[1]
            pos_offset = pos_offset.reshape(rows * columns, 3)[:n_lines, :]
        self._pos_offset = gloo.Texture1D(pos_offset, internalformat='rgb32f', interpolation='nearest')
        self.shared_program['pos_offset'] = self._pos_offset
        if color is None:
            self._color_tex = gloo.Texture1D(np.ones((n_lines, 4), dtype=np.float32))
            self.shared_program['color_tex'] = self._color_tex
            self.shared_program.frag['color'] = 'v_color'
        else:
            self._color_tex = gloo.Texture1D(color)
            self.shared_program['color_tex'] = self._color_tex
            self.shared_program.frag['color'] = 'v_color'
        index = np.empty((data.shape[0], data.shape[1], 2), dtype='float32')
        index[..., 0] = np.arange(data.shape[0])[:, np.newaxis]
        index[..., 1] = np.arange(data.shape[1])[np.newaxis, :]
        index = index.reshape((index.shape[0] * index.shape[1], index.shape[2]))
        self._index_buf.set_data(index)
        self._draw_mode = 'line_strip'
        self.set_gl_state('translucent', line_width=1)
        self.freeze()

    def set_pos_offset(self, po):
        if False:
            i = 10
            return i + 15
        'Set the array of position offsets for each line strip.\n\n        Parameters\n        ----------\n        po : array-like\n            An array of xy offset values.\n        '
        self._pos_offset.set_data(po)

    def set_color(self, color):
        if False:
            print('Hello World!')
        'Set the array of colors for each line strip.\n\n        Parameters\n        ----------\n        color : array-like\n            An array of rgba values.\n        '
        self._color_tex.set_data(color)

    def _prepare_transforms(self, view):
        if False:
            return 10
        view.view_program.vert['transform'] = view.get_transform().simplified

    def _prepare_draw(self, view):
        if False:
            for i in range(10):
                print('nop')
        pass

    def _compute_bounds(self, axis, view):
        if False:
            for i in range(10):
                print('nop')
        if self._pos_data is None:
            return None
        return (self._pos_data[..., axis].min(), self.pos_data[..., axis].max())

    def roll_data(self, data):
        if False:
            i = 10
            return i + 15
        'Append new data to the right side of every line strip and remove\n        as much data from the left.\n\n        Parameters\n        ----------\n        data : array-like\n            A data array to append.\n        '
        data = data.astype('float32')[..., np.newaxis]
        s1 = self._data_shape[1] - self._offset
        if data.shape[1] > s1:
            self._pos_tex[:, self._offset:] = data[:, :s1]
            self._pos_tex[:, :data.shape[1] - s1] = data[:, s1:]
            self._offset = (self._offset + data.shape[1]) % self._data_shape[1]
        else:
            self._pos_tex[:, self._offset:self._offset + data.shape[1]] = data
            self._offset += data.shape[1]
        self.shared_program['offset'] = self._offset
        self.update()

    def set_data(self, index, data):
        if False:
            return 10
        'Set the complete data for a single line strip.\n\n        Parameters\n        ----------\n        index : int\n            The index of the line strip to be replaced.\n        data : array-like\n            The data to assign to the selected line strip.\n        '
        self._pos_tex[index, :] = data
        self.update()