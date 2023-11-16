"""
Plotting Examples

Suggested Usage:    python -i pyglet_plotting.py
"""
from sympy import symbols, sin, cos, pi, sqrt
from sympy.plotting.pygletplot import PygletPlot
from time import sleep, perf_counter

def main():
    if False:
        while True:
            i = 10
    (x, y, z) = symbols('x,y,z')
    axes_options = 'visible=false; colored=true; label_ticks=true; label_axes=true; overlay=true; stride=0.5'
    p = PygletPlot(width=600, height=500, ortho=False, invert_mouse_zoom=False, axes=axes_options, antialiasing=True)
    examples = []

    def example_wrapper(f):
        if False:
            print('Hello World!')
        examples.append(f)
        return f

    @example_wrapper
    def mirrored_saddles():
        if False:
            while True:
                i = 10
        p[5] = (x ** 2 - y ** 2, [20], [20])
        p[6] = (y ** 2 - x ** 2, [20], [20])

    @example_wrapper
    def mirrored_saddles_saveimage():
        if False:
            while True:
                i = 10
        p[5] = (x ** 2 - y ** 2, [20], [20])
        p[6] = (y ** 2 - x ** 2, [20], [20])
        p.wait_for_calculations()
        sleep(1)
        p.saveimage('plot_example.png')

    @example_wrapper
    def mirrored_ellipsoids():
        if False:
            print('Hello World!')
        p[2] = (x ** 2 + y ** 2, [40], [40], 'color=zfade')
        p[3] = (-x ** 2 - y ** 2, [40], [40], 'color=zfade')

    @example_wrapper
    def saddle_colored_by_derivative():
        if False:
            while True:
                i = 10
        f = x ** 2 - y ** 2
        p[1] = (f, 'style=solid')
        p[1].color = (abs(f.diff(x)), abs(f.diff(x) + f.diff(y)), abs(f.diff(y)))

    @example_wrapper
    def ding_dong_surface():
        if False:
            while True:
                i = 10
        f = sqrt(1.0 - y) * y
        p[1] = (f, [x, 0, 2 * pi, 40], [y, -1, 4, 100], 'mode=cylindrical; style=solid; color=zfade4')

    @example_wrapper
    def polar_circle():
        if False:
            return 10
        p[7] = (1, 'mode=polar')

    @example_wrapper
    def polar_flower():
        if False:
            while True:
                i = 10
        p[8] = (1.5 * sin(4 * x), [160], 'mode=polar')
        p[8].color = (z, x, y, (0.5, 0.5, 0.5), (0.8, 0.8, 0.8), (x, y, None, z))

    @example_wrapper
    def simple_cylinder():
        if False:
            return 10
        p[9] = (1, 'mode=cylindrical')

    @example_wrapper
    def cylindrical_hyperbola():
        if False:
            print('Hello World!')
        p[10] = (1 / y, 'mode=polar', [x], [y, -2, 2, 20])

    @example_wrapper
    def extruded_hyperbolas():
        if False:
            i = 10
            return i + 15
        p[11] = (1 / x, [x, -10, 10, 100], [1], 'style=solid')
        p[12] = (-1 / x, [x, -10, 10, 100], [1], 'style=solid')

    @example_wrapper
    def torus():
        if False:
            return 10
        (a, b) = (1, 0.5)
        p[13] = ((a + b * cos(x)) * cos(y), (a + b * cos(x)) * sin(y), b * sin(x), [x, 0, pi * 2, 40], [y, 0, pi * 2, 40])

    @example_wrapper
    def warped_torus():
        if False:
            for i in range(10):
                print('nop')
        (a, b) = (2, 1)
        p[13] = ((a + b * cos(x)) * cos(y), (a + b * cos(x)) * sin(y), b * sin(x) + 0.5 * sin(4 * y), [x, 0, pi * 2, 40], [y, 0, pi * 2, 40])

    @example_wrapper
    def parametric_spiral():
        if False:
            return 10
        p[14] = (cos(y), sin(y), y / 10.0, [y, -4 * pi, 4 * pi, 100])
        p[14].color = (x, (0.1, 0.9), y, (0.1, 0.9), z, (0.1, 0.9))

    @example_wrapper
    def multistep_gradient():
        if False:
            for i in range(10):
                print('nop')
        p[1] = (1, 'mode=spherical', 'style=both')
        gradient = [0.0, (0.3, 0.3, 1.0), 0.3, (0.3, 1.0, 0.3), 0.55, (0.95, 1.0, 0.2), 0.65, (1.0, 0.95, 0.2), 0.85, (1.0, 0.7, 0.2), 1.0, (1.0, 0.3, 0.2)]
        p[1].color = (z, [None, None, z], gradient)

    @example_wrapper
    def lambda_vs_sympy_evaluation():
        if False:
            return 10
        start = perf_counter()
        p[4] = (x ** 2 + y ** 2, [100], [100], 'style=solid')
        p.wait_for_calculations()
        print('lambda-based calculation took %s seconds.' % (perf_counter() - start))
        start = perf_counter()
        p[4] = (x ** 2 + y ** 2, [100], [100], 'style=solid; use_sympy_eval')
        p.wait_for_calculations()
        print('sympy substitution-based calculation took %s seconds.' % (perf_counter() - start))

    @example_wrapper
    def gradient_vectors():
        if False:
            return 10

        def gradient_vectors_inner(f, i):
            if False:
                print('Hello World!')
            from sympy import lambdify
            from sympy.plotting.plot_interval import PlotInterval
            from pyglet.gl import glBegin, glColor3f
            from pyglet.gl import glVertex3f, glEnd, GL_LINES

            def draw_gradient_vectors(f, iu, iv):
                if False:
                    i = 10
                    return i + 15
                '\n                Create a function which draws vectors\n                representing the gradient of f.\n                '
                (dx, dy, dz) = (f.diff(x), f.diff(y), 0)
                FF = lambdify([x, y], [x, y, f])
                FG = lambdify([x, y], [dx, dy, dz])
                iu.v_steps /= 5
                iv.v_steps /= 5
                Gvl = [[[FF(u, v), FG(u, v)] for v in iv.frange()] for u in iu.frange()]

                def draw_arrow(p1, p2):
                    if False:
                        print('Hello World!')
                    '\n                    Draw a single vector.\n                    '
                    glColor3f(0.4, 0.4, 0.9)
                    glVertex3f(*p1)
                    glColor3f(0.9, 0.4, 0.4)
                    glVertex3f(*p2)

                def draw():
                    if False:
                        print('Hello World!')
                    '\n                    Iterate through the calculated\n                    vectors and draw them.\n                    '
                    glBegin(GL_LINES)
                    for u in Gvl:
                        for v in u:
                            point = [[v[0][0], v[0][1], v[0][2]], [v[0][0] + v[1][0], v[0][1] + v[1][1], v[0][2] + v[1][2]]]
                            draw_arrow(point[0], point[1])
                    glEnd()
                return draw
            p[i] = (f, [-0.5, 0.5, 25], [-0.5, 0.5, 25], 'style=solid')
            iu = PlotInterval(p[i].intervals[0])
            iv = PlotInterval(p[i].intervals[1])
            p[i].postdraw.append(draw_gradient_vectors(f, iu, iv))
        gradient_vectors_inner(x ** 2 + y ** 2, 1)
        gradient_vectors_inner(-x ** 2 - y ** 2, 2)

    def help_str():
        if False:
            i = 10
            return i + 15
        s = '\nPlot p has been created. Useful commands: \n    help(p), p[1] = x**2, print(p), p.clear() \n\nAvailable examples (see source in plotting.py):\n\n'
        for i in range(len(examples)):
            s += '(%i) %s\n' % (i, examples[i].__name__)
        s += '\n'
        s += 'e.g. >>> example(2)\n'
        s += '     >>> ding_dong_surface()\n'
        return s

    def example(i):
        if False:
            return 10
        if callable(i):
            p.clear()
            i()
        elif i >= 0 and i < len(examples):
            p.clear()
            examples[i]()
        else:
            print('Not a valid example.\n')
        print(p)
    example(0)
    print(help_str())
if __name__ == '__main__':
    main()