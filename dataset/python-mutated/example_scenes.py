from manimlib import *
import numpy as np

class OpeningManimExample(Scene):

    def construct(self):
        if False:
            for i in range(10):
                print('nop')
        intro_words = Text('\n            The original motivation for manim was to\n            better illustrate mathematical functions\n            as transformations.\n        ')
        intro_words.to_edge(UP)
        self.play(Write(intro_words))
        self.wait(2)
        grid = NumberPlane((-10, 10), (-5, 5))
        matrix = [[1, 1], [0, 1]]
        linear_transform_words = VGroup(Text('This is what the matrix'), IntegerMatrix(matrix, include_background_rectangle=True), Text('looks like'))
        linear_transform_words.arrange(RIGHT)
        linear_transform_words.to_edge(UP)
        linear_transform_words.set_backstroke(width=5)
        self.play(ShowCreation(grid), FadeTransform(intro_words, linear_transform_words))
        self.wait()
        self.play(grid.animate.apply_matrix(matrix), run_time=3)
        self.wait()
        c_grid = ComplexPlane()
        moving_c_grid = c_grid.copy()
        moving_c_grid.prepare_for_nonlinear_transform()
        c_grid.set_stroke(BLUE_E, 1)
        c_grid.add_coordinate_labels(font_size=24)
        complex_map_words = TexText('\n            Or thinking of the plane as $\\mathds{C}$,\\\\\n            this is the map $z \\rightarrow z^2$\n        ')
        complex_map_words.to_corner(UR)
        complex_map_words.set_backstroke(width=5)
        self.play(FadeOut(grid), Write(c_grid, run_time=3), FadeIn(moving_c_grid), FadeTransform(linear_transform_words, complex_map_words))
        self.wait()
        self.play(moving_c_grid.animate.apply_complex_function(lambda z: z ** 2), run_time=6)
        self.wait(2)

class AnimatingMethods(Scene):

    def construct(self):
        if False:
            return 10
        grid = Tex('\\pi').get_grid(10, 10, height=4)
        self.add(grid)
        self.play(grid.animate.shift(LEFT))
        self.play(grid.animate.set_color(YELLOW))
        self.wait()
        self.play(grid.animate.set_submobject_colors_by_gradient(BLUE, GREEN))
        self.wait()
        self.play(grid.animate.set_height(TAU - MED_SMALL_BUFF))
        self.wait()
        self.play(grid.animate.apply_complex_function(np.exp), run_time=5)
        self.wait()
        self.play(grid.animate.apply_function(lambda p: [p[0] + 0.5 * math.sin(p[1]), p[1] + 0.5 * math.sin(p[0]), p[2]]), run_time=5)
        self.wait()

class TextExample(Scene):

    def construct(self):
        if False:
            print('Hello World!')
        text = Text('Here is a text', font='Consolas', font_size=90)
        difference = Text("\n            The most important difference between Text and TexText is that\n\n            you can change the font more easily, but can't use the LaTeX grammar\n            ", font='Arial', font_size=24, t2c={'Text': BLUE, 'TexText': BLUE, 'LaTeX': ORANGE})
        VGroup(text, difference).arrange(DOWN, buff=1)
        self.play(Write(text))
        self.play(FadeIn(difference, UP))
        self.wait(3)
        fonts = Text('And you can also set the font according to different words', font='Arial', t2f={'font': 'Consolas', 'words': 'Consolas'}, t2c={'font': BLUE, 'words': GREEN})
        fonts.set_width(FRAME_WIDTH - 1)
        slant = Text('And the same as slant and weight', font='Consolas', t2s={'slant': ITALIC}, t2w={'weight': BOLD}, t2c={'slant': ORANGE, 'weight': RED})
        VGroup(fonts, slant).arrange(DOWN, buff=0.8)
        self.play(FadeOut(text), FadeOut(difference, shift=DOWN))
        self.play(Write(fonts))
        self.wait()
        self.play(Write(slant))
        self.wait()

class TexTransformExample(Scene):

    def construct(self):
        if False:
            while True:
                i = 10
        t2c = {'A': BLUE, 'B': TEAL, 'C': GREEN}
        kw = dict(font_size=72, t2c=t2c)
        lines = VGroup(Tex('A^2 + B^2 = C^2', **kw), Tex('A^2 = C^2 - B^2', **kw), Tex('A^2 = (C + B)(C - B)', **kw), Tex('A = \\sqrt{(C + B)(C - B)}', **kw))
        lines.arrange(DOWN, buff=LARGE_BUFF)
        self.add(lines[0])
        self.play(TransformMatchingStrings(lines[0].copy(), lines[1], matched_keys=['A^2', 'B^2', 'C^2'], key_map={'+': '-'}, path_arc=90 * DEGREES))
        self.wait()
        self.play(TransformMatchingStrings(lines[1].copy(), lines[2], matched_keys=['A^2']))
        self.wait()
        self.play(TransformMatchingStrings(lines[2].copy(), lines[3], key_map={'2': '\\sqrt'}, path_arc=-30 * DEGREES))
        self.wait(2)
        self.play(LaggedStartMap(FadeOut, lines, shift=2 * RIGHT))
        source = Text('the morse code', height=1)
        target = Text('here come dots', height=1)
        saved_source = source.copy()
        self.play(Write(source))
        self.wait()
        kw = dict(run_time=3, path_arc=PI / 2)
        self.play(TransformMatchingShapes(source, target, **kw))
        self.wait()
        self.play(TransformMatchingShapes(target, saved_source, **kw))
        self.wait()

class TexIndexing(Scene):

    def construct(self):
        if False:
            for i in range(10):
                print('nop')
        equation = Tex('e^{\\pi i} = -1', font_size=144)
        self.add(equation)
        self.play(FlashAround(equation['e']))
        self.wait()
        self.play(Indicate(equation['\\pi']))
        self.wait()
        self.play(TransformFromCopy(equation['e^{\\pi i}'].copy().set_opacity(0.5), equation['-1'], path_arc=-PI / 2, run_time=3))
        self.play(FadeOut(equation))
        equation = Tex('A^2 + B^2 = C^2', font_size=144)
        self.play(Write(equation))
        for part in equation[re.compile('\\w\\^2')]:
            self.play(FlashAround(part))
        self.wait()
        self.play(FadeOut(equation))
        equation = Tex('\\sum_{n = 1}^\\infty \\frac{1}{n^2} = \\frac{\\pi^2}{6}', font_size=72)
        self.play(FadeIn(equation))
        self.play(equation['\\infty'].animate.set_color(RED))
        self.wait()
        self.play(FadeOut(equation))
        equation = Tex('\\sum_{n = 1}^\\infty {1 \\over n^2} = {\\pi^2 \\over 6}', isolate=['\\infty'], font_size=72)
        self.play(FadeIn(equation))
        self.play(equation['\\infty'].animate.set_color(RED))
        self.wait()
        self.play(FadeOut(equation))

class UpdatersExample(Scene):

    def construct(self):
        if False:
            i = 10
            return i + 15
        square = Square()
        square.set_fill(BLUE_E, 1)
        brace = always_redraw(Brace, square, UP)
        label = TexText('Width = 0.00')
        number = label.make_number_changable('0.00')
        always(label.next_to, brace, UP)
        f_always(number.set_value, square.get_width)
        self.add(square, brace, label)
        self.play(square.animate.scale(2), rate_func=there_and_back, run_time=2)
        self.wait()
        self.play(square.animate.set_width(5, stretch=True), run_time=3)
        self.wait()
        self.play(square.animate.set_width(2), run_time=3)
        self.wait()
        now = self.time
        w0 = square.get_width()
        square.add_updater(lambda m: m.set_width(w0 * math.sin(self.time - now) + w0))
        self.wait(4 * PI)

class CoordinateSystemExample(Scene):

    def construct(self):
        if False:
            i = 10
            return i + 15
        axes = Axes(x_range=(-1, 10), y_range=(-2, 2, 0.5), height=6, width=10, axis_config=dict(stroke_color=GREY_A, stroke_width=2, numbers_to_exclude=[0]), y_axis_config=dict(numbers_with_elongated_ticks=[-2, 2]))
        axes.add_coordinate_labels(font_size=20, num_decimal_places=1)
        self.add(axes)
        dot = Dot(color=RED)
        dot.move_to(axes.c2p(0, 0))
        self.play(FadeIn(dot, scale=0.5))
        self.play(dot.animate.move_to(axes.c2p(3, 2)))
        self.wait()
        self.play(dot.animate.move_to(axes.c2p(5, 0.5)))
        self.wait()
        h_line = always_redraw(lambda : axes.get_h_line(dot.get_left()))
        v_line = always_redraw(lambda : axes.get_v_line(dot.get_bottom()))
        self.play(ShowCreation(h_line), ShowCreation(v_line))
        self.play(dot.animate.move_to(axes.c2p(3, -2)))
        self.wait()
        self.play(dot.animate.move_to(axes.c2p(1, 1)))
        self.wait()
        f_always(dot.move_to, lambda : axes.c2p(1, 1))
        self.play(axes.animate.scale(0.75).to_corner(UL), run_time=2)
        self.wait()
        self.play(FadeOut(VGroup(axes, dot, h_line, v_line)))

class GraphExample(Scene):

    def construct(self):
        if False:
            for i in range(10):
                print('nop')
        axes = Axes((-3, 10), (-1, 8), height=6)
        axes.add_coordinate_labels()
        self.play(Write(axes, lag_ratio=0.01, run_time=1))
        sin_graph = axes.get_graph(lambda x: 2 * math.sin(x), color=BLUE)
        relu_graph = axes.get_graph(lambda x: max(x, 0), use_smoothing=False, color=YELLOW)
        step_graph = axes.get_graph(lambda x: 2.0 if x > 3 else 1.0, discontinuities=[3], color=GREEN)
        sin_label = axes.get_graph_label(sin_graph, '\\sin(x)')
        relu_label = axes.get_graph_label(relu_graph, Text('ReLU'))
        step_label = axes.get_graph_label(step_graph, Text('Step'), x=4)
        self.play(ShowCreation(sin_graph), FadeIn(sin_label, RIGHT))
        self.wait(2)
        self.play(ReplacementTransform(sin_graph, relu_graph), FadeTransform(sin_label, relu_label))
        self.wait()
        self.play(ReplacementTransform(relu_graph, step_graph), FadeTransform(relu_label, step_label))
        self.wait()
        parabola = axes.get_graph(lambda x: 0.25 * x ** 2)
        parabola.set_stroke(BLUE)
        self.play(FadeOut(step_graph), FadeOut(step_label), ShowCreation(parabola))
        self.wait()
        dot = Dot(color=RED)
        dot.move_to(axes.i2gp(2, parabola))
        self.play(FadeIn(dot, scale=0.5))
        x_tracker = ValueTracker(2)
        f_always(dot.move_to, lambda : axes.i2gp(x_tracker.get_value(), parabola))
        self.play(x_tracker.animate.set_value(4), run_time=3)
        self.play(x_tracker.animate.set_value(-2), run_time=3)
        self.wait()

class TexAndNumbersExample(Scene):

    def construct(self):
        if False:
            return 10
        axes = Axes((-3, 3), (-3, 3), unit_size=1)
        axes.to_edge(DOWN)
        axes.add_coordinate_labels(font_size=16)
        circle = Circle(radius=2)
        circle.set_stroke(YELLOW, 3)
        circle.move_to(axes.get_origin())
        self.add(axes, circle)
        tex = Tex('x^2 + y^2 = 4.00')
        tex.next_to(axes, UP, buff=0.5)
        value = tex.make_number_changable('4.00')
        value.add_updater(lambda v: v.set_value(circle.get_radius() ** 2))
        self.add(tex)
        text = Text('\n            You can manipulate numbers\n            in Tex mobjects\n        ', font_size=30)
        text.next_to(tex, RIGHT, buff=1.5)
        arrow = Arrow(text, tex)
        self.add(text, arrow)
        self.play(circle.animate.set_height(2.0), run_time=4, rate_func=there_and_back)
        exponents = tex.make_number_changable('2', replace_all=True)
        self.play(LaggedStartMap(FlashAround, exponents, lag_ratio=0.2, buff=0.1, color=RED), exponents.animate.set_color(RED))

        def func(x, y):
            if False:
                while True:
                    i = 10
            (xa, ya) = axes.point_to_coords(np.array([x, y, 0]))
            return xa ** 4 + ya ** 4 - 4
        new_curve = ImplicitFunction(func)
        new_curve.match_style(circle)
        circle.rotate(angle_of_vector(new_curve.get_start()))
        value.clear_updaters()
        self.play(*(ChangeDecimalToValue(exp, 4) for exp in exponents), ReplacementTransform(circle.copy(), new_curve), circle.animate.set_stroke(width=1, opacity=0.5))

class SurfaceExample(ThreeDScene):

    def construct(self):
        if False:
            while True:
                i = 10
        surface_text = Text('For 3d scenes, try using surfaces')
        surface_text.fix_in_frame()
        surface_text.to_edge(UP)
        self.add(surface_text)
        self.wait(0.1)
        torus1 = Torus(r1=1, r2=1)
        torus2 = Torus(r1=3, r2=1)
        sphere = Sphere(radius=3, resolution=torus1.resolution)
        day_texture = 'https://upload.wikimedia.org/wikipedia/commons/thumb/4/4d/Whole_world_-_land_and_oceans.jpg/1280px-Whole_world_-_land_and_oceans.jpg'
        night_texture = 'https://upload.wikimedia.org/wikipedia/commons/thumb/b/ba/The_earth_at_night.jpg/1280px-The_earth_at_night.jpg'
        surfaces = [TexturedSurface(surface, day_texture, night_texture) for surface in [sphere, torus1, torus2]]
        for mob in surfaces:
            mob.shift(IN)
            mob.mesh = SurfaceMesh(mob)
            mob.mesh.set_stroke(BLUE, 1, opacity=0.5)
        surface = surfaces[0]
        self.play(FadeIn(surface), ShowCreation(surface.mesh, lag_ratio=0.01, run_time=3))
        for mob in surfaces:
            mob.add(mob.mesh)
        surface.save_state()
        self.play(Rotate(surface, PI / 2), run_time=2)
        for mob in surfaces[1:]:
            mob.rotate(PI / 2)
        self.play(Transform(surface, surfaces[1]), run_time=3)
        self.play(Transform(surface, surfaces[2]), self.frame.animate.increment_phi(-10 * DEGREES), self.frame.animate.increment_theta(-20 * DEGREES), run_time=3)
        self.frame.add_updater(lambda m, dt: m.increment_theta(-0.1 * dt))
        light_text = Text('You can move around the light source')
        light_text.move_to(surface_text)
        light_text.fix_in_frame()
        self.play(FadeTransform(surface_text, light_text))
        light = self.camera.light_source
        self.add(light)
        light.save_state()
        self.play(light.animate.move_to(3 * IN), run_time=5)
        self.play(light.animate.shift(10 * OUT), run_time=5)
        drag_text = Text('Try moving the mouse while pressing d or f')
        drag_text.move_to(light_text)
        drag_text.fix_in_frame()
        self.play(FadeTransform(light_text, drag_text))
        self.wait()

class InteractiveDevelopment(Scene):

    def construct(self):
        if False:
            print('Hello World!')
        circle = Circle()
        circle.set_fill(BLUE, opacity=0.5)
        circle.set_stroke(BLUE_E, width=4)
        square = Square()
        self.play(ShowCreation(square))
        self.wait()
        self.embed()
        self.play(ReplacementTransform(square, circle))
        self.wait()
        self.play(circle.animate.stretch(4, 0))
        self.play(Rotate(circle, 90 * DEGREES))
        self.play(circle.animate.shift(2 * RIGHT).scale(0.25))
        text = Text('\n            In general, using the interactive shell\n            is very helpful when developing new scenes\n        ')
        self.play(Write(text))
        always(circle.move_to, self.mouse_point)

class ControlsExample(Scene):
    drag_to_pan = False

    def setup(self):
        if False:
            print('Hello World!')
        self.textbox = Textbox()
        self.checkbox = Checkbox()
        self.color_picker = ColorSliders()
        self.panel = ControlPanel(Text('Text', font_size=24), self.textbox, Line(), Text('Show/Hide Text', font_size=24), self.checkbox, Line(), Text('Color of Text', font_size=24), self.color_picker)
        self.add(self.panel)

    def construct(self):
        if False:
            return 10
        text = Text('text', font_size=96)

        def text_updater(old_text):
            if False:
                while True:
                    i = 10
            assert isinstance(old_text, Text)
            new_text = Text(self.textbox.get_value(), font_size=old_text.font_size)
            new_text.move_to(old_text)
            if self.checkbox.get_value():
                new_text.set_fill(color=self.color_picker.get_picked_color(), opacity=self.color_picker.get_picked_opacity())
            else:
                new_text.set_opacity(0)
            old_text.become(new_text)
        text.add_updater(text_updater)
        self.add(MotionMobject(text))
        self.textbox.set_value('Manim')