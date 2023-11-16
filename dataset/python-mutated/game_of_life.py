"""
Conway game of life.
"""
import numpy as np
from vispy.gloo import Program, FrameBuffer, RenderBuffer, clear, set_viewport, set_state
from vispy import app
render_vertex = '\nattribute vec2 position;\nattribute vec2 texcoord;\nvarying vec2 v_texcoord;\nvoid main()\n{\n    gl_Position = vec4(position, 0.0, 1.0);\n    v_texcoord = texcoord;\n}\n'
render_fragment = '\nuniform int pingpong;\nuniform sampler2D texture;\nvarying vec2 v_texcoord;\nvoid main()\n{\n    float v;\n    v = texture2D(texture, v_texcoord)[pingpong];\n    gl_FragColor = vec4(1.0-v, 1.0-v, 1.0-v, 1.0);\n}\n'
compute_vertex = '\nattribute vec2 position;\nattribute vec2 texcoord;\nvarying vec2 v_texcoord;\nvoid main()\n{\n    gl_Position = vec4(position, 0.0, 1.0);\n    v_texcoord = texcoord;\n}\n'
compute_fragment = '\nuniform int pingpong;\nuniform sampler2D texture;\nuniform float dx;          // horizontal distance between texels\nuniform float dy;          // vertical distance between texels\nvarying vec2 v_texcoord;\nvoid main(void)\n{\n    vec2  p = v_texcoord;\n    float old_state, new_state, count;\n\n    old_state = texture2D(texture, p)[pingpong];\n    count = texture2D(texture, p + vec2(-dx,-dy))[pingpong]\n            + texture2D(texture, p + vec2( dx,-dy))[pingpong]\n            + texture2D(texture, p + vec2(-dx, dy))[pingpong]\n            + texture2D(texture, p + vec2( dx, dy))[pingpong]\n            + texture2D(texture, p + vec2(-dx, 0.0))[pingpong]\n            + texture2D(texture, p + vec2( dx, 0.0))[pingpong]\n            + texture2D(texture, p + vec2(0.0,-dy))[pingpong]\n            + texture2D(texture, p + vec2(0.0, dy))[pingpong];\n\n    new_state = old_state;\n    if( old_state > 0.5 ) {\n        // Any live cell with fewer than two live neighbours dies\n        // as if caused by under-population.\n        if( count  < 1.5 )\n            new_state = 0.0;\n\n        // Any live cell with two or three live neighbours\n        // lives on to the next generation.\n\n        // Any live cell with more than three live neighbours dies,\n        //  as if by overcrowding.\n        else if( count > 3.5 )\n            new_state = 0.0;\n    } else {\n        // Any dead cell with exactly three live neighbours becomes\n        //  a live cell, as if by reproduction.\n       if( (count > 2.5) && (count < 3.5) )\n           new_state = 1.0;\n    }\n\n    if( pingpong == 0) {\n        gl_FragColor[1] = new_state;\n        gl_FragColor[0] = old_state;\n    } else {\n        gl_FragColor[1] = old_state;\n        gl_FragColor[0] = new_state;\n    }\n}\n'

class Canvas(app.Canvas):

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        app.Canvas.__init__(self, title='Conway game of life', size=(512, 512), keys='interactive')
        self.comp_size = self.size
        size = self.comp_size + (4,)
        Z = np.zeros(size, dtype=np.float32)
        Z[...] = np.random.randint(0, 2, size)
        Z[:256, :256, :] = 0
        gun = '\n        ........................O...........\n        ......................O.O...........\n        ............OO......OO............OO\n        ...........O...O....OO............OO\n        OO........O.....O...OO..............\n        OO........O...O.OO....O.O...........\n        ..........O.....O.......O...........\n        ...........O...O....................\n        ............OO......................'
        (x, y) = (0, 0)
        for i in range(len(gun)):
            if gun[i] == '\n':
                y += 1
                x = 0
            elif gun[i] == 'O':
                Z[y, x] = 1
            x += 1
        self.pingpong = 1
        self.compute = Program(compute_vertex, compute_fragment, 4)
        self.compute['texture'] = Z
        self.compute['position'] = [(-1, -1), (-1, +1), (+1, -1), (+1, +1)]
        self.compute['texcoord'] = [(0, 0), (0, 1), (1, 0), (1, 1)]
        self.compute['dx'] = 1.0 / size[1]
        self.compute['dy'] = 1.0 / size[0]
        self.compute['pingpong'] = self.pingpong
        self.render = Program(render_vertex, render_fragment, 4)
        self.render['position'] = [(-1, -1), (-1, +1), (+1, -1), (+1, +1)]
        self.render['texcoord'] = [(0, 0), (0, 1), (1, 0), (1, 1)]
        self.render['texture'] = self.compute['texture']
        self.render['pingpong'] = self.pingpong
        self.fbo = FrameBuffer(self.compute['texture'], RenderBuffer(self.comp_size))
        set_state(depth_test=False, clear_color='black')
        self._timer = app.Timer('auto', connect=self.update, start=True)
        self.show()

    def on_draw(self, event):
        if False:
            print('Hello World!')
        with self.fbo:
            set_viewport(0, 0, *self.comp_size)
            self.compute['texture'].interpolation = 'nearest'
            self.compute.draw('triangle_strip')
        clear()
        set_viewport(0, 0, *self.physical_size)
        self.render['texture'].interpolation = 'linear'
        self.render.draw('triangle_strip')
        self.pingpong = 1 - self.pingpong
        self.compute['pingpong'] = self.pingpong
        self.render['pingpong'] = self.pingpong
if __name__ == '__main__':
    canvas = Canvas()
    app.run()