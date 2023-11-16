from vispy import app, gloo
vertex = '\nattribute vec2 position;\n\nvoid main()\n{\n    gl_Position = vec4(position, 0, 1.0);\n}\n'
fragment = '\nuniform vec2 resolution;\nuniform vec2 center;\nuniform float scale;\n\nvec3 hot(float t)\n{\n    return vec3(smoothstep(0.00,0.33,t),\n                smoothstep(0.33,0.66,t),\n                smoothstep(0.66,1.00,t));\n}\n\nvoid main()\n{\n    \n    const int n = 300;\n    const float log_2 = 0.6931471805599453;\n\n    vec2 c;\n\n    // Recover coordinates from pixel coordinates\n    c.x = (gl_FragCoord.x / resolution.x - 0.5) * scale + center.x;\n    c.y = (gl_FragCoord.y / resolution.y - 0.5) * scale + center.y;\n\n    float x, y, d;\n    int i;\n    vec2 z = c;\n    for(i = 0; i < n; ++i)\n    {\n        x = (z.x*z.x - z.y*z.y) + c.x;\n        y = (z.y*z.x + z.x*z.y) + c.y;\n        d = x*x + y*y;\n        if (d > 4.0) break;\n        z = vec2(x,y);\n    }\n    if ( i < n ) {\n        float nu = log(log(sqrt(d))/log_2)/log_2;\n        float index = float(i) + 1.0 - nu;\n        float v = pow(index/float(n),0.5);\n        gl_FragColor = vec4(hot(v),1.0);\n    } else {\n        gl_FragColor = vec4(hot(0.0),1.0);\n    }\n}\n\n'

class Canvas(app.Canvas):

    def __init__(self, *args, **kwargs):
        if False:
            print('Hello World!')
        app.Canvas.__init__(self, *args, **kwargs)
        self.program = gloo.Program(vertex, fragment)
        self.program['position'] = [(-1, -1), (-1, 1), (1, 1), (-1, -1), (1, 1), (1, -1)]
        self.scale = self.program['scale'] = 3
        self.center = self.program['center'] = [-0.5, 0]
        self.apply_zoom()
        self.bounds = [-2, 2]
        self.min_scale = 5e-05
        self.max_scale = 4
        gloo.set_clear_color(color='black')
        self._timer = app.Timer('auto', connect=self.update, start=True)
        self.show()

    def on_draw(self, event):
        if False:
            while True:
                i = 10
        self.program.draw()

    def on_resize(self, event):
        if False:
            return 10
        self.apply_zoom()

    def apply_zoom(self):
        if False:
            i = 10
            return i + 15
        (width, height) = self.physical_size
        gloo.set_viewport(0, 0, width, height)
        self.program['resolution'] = [width, height]

    def on_mouse_move(self, event):
        if False:
            print('Hello World!')
        'Pan the view based on the change in mouse position.'
        if event.is_dragging and event.buttons[0] == 1:
            (x0, y0) = (event.last_event.pos[0], event.last_event.pos[1])
            (x1, y1) = (event.pos[0], event.pos[1])
            (X0, Y0) = self.pixel_to_coords(float(x0), float(y0))
            (X1, Y1) = self.pixel_to_coords(float(x1), float(y1))
            self.translate_center(X1 - X0, Y1 - Y0)

    def translate_center(self, dx, dy):
        if False:
            print('Hello World!')
        'Translates the center point, and keeps it in bounds.'
        center = self.center
        center[0] -= dx
        center[1] -= dy
        center[0] = min(max(center[0], self.bounds[0]), self.bounds[1])
        center[1] = min(max(center[1], self.bounds[0]), self.bounds[1])
        self.program['center'] = self.center = center

    def pixel_to_coords(self, x, y):
        if False:
            while True:
                i = 10
        'Convert pixel coordinates to Mandelbrot set coordinates.'
        (rx, ry) = self.size
        nx = (x / rx - 0.5) * self.scale + self.center[0]
        ny = ((ry - y) / ry - 0.5) * self.scale + self.center[1]
        return [nx, ny]

    def on_mouse_wheel(self, event):
        if False:
            i = 10
            return i + 15
        'Use the mouse wheel to zoom.'
        delta = event.delta[1]
        if delta > 0:
            factor = 0.9
        elif delta < 0:
            factor = 1 / 0.9
        for _ in range(int(abs(delta))):
            self.zoom(factor, event.pos)

    def on_key_press(self, event):
        if False:
            while True:
                i = 10
        "Use + or - to zoom in and out.\n\n        The mouse wheel can be used to zoom, but some people don't have mouse\n        wheels :)\n\n        "
        if event.text == '+' or event.text == '=':
            self.zoom(0.9)
        elif event.text == '-':
            self.zoom(1 / 0.9)

    def zoom(self, factor, mouse_coords=None):
        if False:
            return 10
        'Factors less than zero zoom in, and greater than zero zoom out.\n\n        If mouse_coords is given, the point under the mouse stays stationary\n        while zooming. mouse_coords should come from MouseEvent.pos.\n\n        '
        if mouse_coords is not None:
            (x, y) = (float(mouse_coords[0]), float(mouse_coords[1]))
            (x0, y0) = self.pixel_to_coords(x, y)
        self.scale *= factor
        self.scale = max(min(self.scale, self.max_scale), self.min_scale)
        self.program['scale'] = self.scale
        if mouse_coords is not None:
            (x1, y1) = self.pixel_to_coords(x, y)
            self.translate_center(x1 - x0, y1 - y0)
if __name__ == '__main__':
    canvas = Canvas(size=(800, 800), keys='interactive')
    app.run()