"""
Example demonstrating the use of emulated double-precision floating point
numbers. Based off of mandelbrot.py.

The shader program emulates double-precision variables using a vec2 instead of
single-precision floats. Any function starting with ds_* operates on these
variables. See http://www.thasler.org/blog/?p=93.

NOTE: Some NVIDIA cards optimize the double-precision code away. Results are
therefore hardware dependent.

"""
from __future__ import division
import numpy as np
from vispy import app, gloo
vertex = '\nattribute vec2 position;\n\nvoid main()\n{\n    gl_Position = vec4(position, 0, 1.0);\n}\n'
fragment = '\n#pragma optionNV(fastmath off)\n#pragma optionNV(fastprecision off)\n\nuniform vec2 inv_resolution_x;  // Inverse resolutions\nuniform vec2 inv_resolution_y;\nuniform vec2 center_x;\nuniform vec2 center_y;\nuniform vec2 scale;\nuniform int iter;\n\n// Jet color scheme\nvec4 color_scheme(float x) {\n    vec3 a, b;\n    float c;\n    if (x < 0.34) {\n        a = vec3(0, 0, 0.5);\n        b = vec3(0, 0.8, 0.95);\n        c = (x - 0.0) / (0.34 - 0.0);\n    } else if (x < 0.64) {\n        a = vec3(0, 0.8, 0.95);\n        b = vec3(0.85, 1, 0.04);\n        c = (x - 0.34) / (0.64 - 0.34);\n    } else if (x < 0.89) {\n        a = vec3(0.85, 1, 0.04);\n        b = vec3(0.96, 0.7, 0);\n        c = (x - 0.64) / (0.89 - 0.64);\n    } else {\n        a = vec3(0.96, 0.7, 0);\n        b = vec3(0.5, 0, 0);\n        c = (x - 0.89) / (1.0 - 0.89);\n    }\n    return vec4(mix(a, b, c), 1.0);\n}\n\nvec2 ds_set(float a) {\n    // Create an emulated double by storing first part of float in first half\n    // of vec2\n    vec2 z;\n    z.x = a;\n    z.y = 0.0;\n    return z;\n}\n\nvec2 ds_add (vec2 dsa, vec2 dsb)\n{\n    // Add two emulated doubles. Complexity comes from carry-over.\n    vec2 dsc;\n    float t1, t2, e;\n\n    t1 = dsa.x + dsb.x;\n    e = t1 - dsa.x;\n    t2 = ((dsb.x - e) + (dsa.x - (t1 - e))) + dsa.y + dsb.y;\n\n    dsc.x = t1 + t2;\n    dsc.y = t2 - (dsc.x - t1);\n    return dsc;\n}\n\nvec2 ds_mul (vec2 dsa, vec2 dsb)\n{\n    vec2 dsc;\n    float c11, c21, c2, e, t1, t2;\n    float a1, a2, b1, b2, cona, conb, split = 8193.;\n\n    cona = dsa.x * split;\n    conb = dsb.x * split;\n    a1 = cona - (cona - dsa.x);\n    b1 = conb - (conb - dsb.x);\n    a2 = dsa.x - a1;\n    b2 = dsb.x - b1;\n\n    c11 = dsa.x * dsb.x;\n    c21 = a2 * b2 + (a2 * b1 + (a1 * b2 + (a1 * b1 - c11)));\n\n    c2 = dsa.x * dsb.y + dsa.y * dsb.x;\n\n    t1 = c11 + c2;\n    e = t1 - c11;\n    t2 = dsa.y * dsb.y + ((c2 - e) + (c11 - (t1 - e))) + c21;\n\n    dsc.x = t1 + t2;\n    dsc.y = t2 - (dsc.x - t1);\n\n    return dsc;\n}\n\n// Compare: res = -1 if a < b\n//              = 0 if a == b\n//              = 1 if a > b\nfloat ds_compare(vec2 dsa, vec2 dsb)\n{\n    if (dsa.x < dsb.x) return -1.;\n    else if (dsa.x == dsb.x) {\n        if (dsa.y < dsb.y) return -1.;\n        else if (dsa.y == dsb.y) return 0.;\n        else return 1.;\n    }\n    else return 1.;\n}\n\nvoid main() {\n    vec2 z_x, z_y, c_x, c_y, x, y, frag_x, frag_y;\n    vec2 four = ds_set(4.0);\n    vec2 point5 = ds_set(0.5);\n\n    // Recover coordinates from pixel coordinates\n    frag_x = ds_set(gl_FragCoord.x);\n    frag_y = ds_set(gl_FragCoord.y);\n\n    c_x = ds_add(ds_mul(frag_x, inv_resolution_x), -point5);\n    c_x = ds_add(ds_mul(c_x, scale), center_x);\n    c_y = ds_add(ds_mul(frag_y, inv_resolution_y), -point5);\n    c_y = ds_add(ds_mul(c_y, scale), center_y);\n\n\n    // Main Mandelbrot computation\n    int i;\n    z_x = c_x;\n    z_y = c_y;\n    for(i = 0; i < iter; i++) {\n        x = ds_add(ds_add(ds_mul(z_x, z_x), -ds_mul(z_y, z_y)), c_x);\n        y = ds_add(ds_add(ds_mul(z_y, z_x), ds_mul(z_x, z_y)), c_y);\n\n        if(ds_compare(ds_add(ds_mul(x, x), ds_mul(y, y)), four) > 0.) break;\n        z_x = x;\n        z_y = y;\n    }\n\n    // Convert iterations to color\n    float color = 1.0 - float(i) / float(iter);\n    gl_FragColor = color_scheme(color);\n\n}\n'

class Canvas(app.Canvas):

    def __init__(self, *args, **kwargs):
        if False:
            i = 10
            return i + 15
        app.Canvas.__init__(self, *args, **kwargs)
        self.program = gloo.Program(vertex, fragment)
        self.program['position'] = [(-1, -1), (-1, 1), (1, 1), (-1, -1), (1, 1), (1, -1)]
        self.scale = 3
        self.program['scale'] = set_emulated_double(self.scale)
        self.center = [-0.5, 0]
        self.bounds = [-2, 2]
        self.translate_center(0, 0)
        self.iterations = self.program['iter'] = 300
        self.apply_zoom()
        self.min_scale = 1e-12
        self.max_scale = 4
        gloo.set_clear_color(color='black')
        self.show()

    def on_draw(self, event):
        if False:
            i = 10
            return i + 15
        self.program.draw()

    def on_resize(self, event):
        if False:
            while True:
                i = 10
        self.apply_zoom()

    def apply_zoom(self):
        if False:
            return 10
        (width, height) = self.physical_size
        gloo.set_viewport(0, 0, width, height)
        self.program['inv_resolution_x'] = set_emulated_double(1 / width)
        self.program['inv_resolution_y'] = set_emulated_double(1 / height)

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
            self.update()

    def translate_center(self, dx, dy):
        if False:
            while True:
                i = 10
        'Translates the center point, and keeps it in bounds.'
        center = self.center
        center[0] -= dx
        center[1] -= dy
        center[0] = min(max(center[0], self.bounds[0]), self.bounds[1])
        center[1] = min(max(center[1], self.bounds[0]), self.bounds[1])
        self.center = center
        center_x = set_emulated_double(center[0])
        center_y = set_emulated_double(center[1])
        self.program['center_x'] = center_x
        self.program['center_y'] = center_y

    def pixel_to_coords(self, x, y):
        if False:
            for i in range(10):
                print('nop')
        'Convert pixel coordinates to Mandelbrot set coordinates.'
        (rx, ry) = self.size
        nx = (x / rx - 0.5) * self.scale + self.center[0]
        ny = ((ry - y) / ry - 0.5) * self.scale + self.center[1]
        return [nx, ny]

    def on_mouse_wheel(self, event):
        if False:
            return 10
        'Use the mouse wheel to zoom.'
        delta = event.delta[1]
        if delta > 0:
            factor = 0.9
        elif delta < 0:
            factor = 1 / 0.9
        for _ in range(int(abs(delta))):
            self.zoom(factor, event.pos)
        self.update()

    def on_key_press(self, event):
        if False:
            print('Hello World!')
        "Use + or - to zoom in and out.\n\n        The mouse wheel can be used to zoom, but some people don't have mouse\n        wheels :)\n\n        "
        if event.text == '+' or event.text == '=':
            self.zoom(0.9)
        elif event.text == '-':
            self.zoom(1 / 0.9)
        self.update()

    def zoom(self, factor, mouse_coords=None):
        if False:
            print('Hello World!')
        'Factors less than zero zoom in, and greater than zero zoom out.\n\n        If mouse_coords is given, the point under the mouse stays stationary\n        while zooming. mouse_coords should come from MouseEvent.pos.\n\n        '
        if mouse_coords is not None:
            (x, y) = (float(mouse_coords[0]), float(mouse_coords[1]))
            (x0, y0) = self.pixel_to_coords(x, y)
        self.scale *= factor
        self.scale = max(min(self.scale, self.max_scale), self.min_scale)
        self.program['scale'] = set_emulated_double(self.scale)
        if mouse_coords is not None:
            (x1, y1) = self.pixel_to_coords(x, y)
            self.translate_center(x1 - x0, y1 - y0)

def set_emulated_double(number):
    if False:
        i = 10
        return i + 15
    'Emulate a double using two numbers of type float32.'
    double = np.array([number, 0], dtype=np.float32)
    double[1] = number - double[0]
    return double
if __name__ == '__main__':
    canvas = Canvas(size=(800, 800), keys='interactive')
    app.run()