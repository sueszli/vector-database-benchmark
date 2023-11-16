from math import cos, sin, pi, sqrt
from random import random, randint
from itertools import combinations
from kivy.app import App
from kivy.clock import Clock
from kivy.uix.label import Label
from kivy.uix.widget import Widget
from kivy.core.window import Window
from kivy.graphics import Color, Mesh, Point
from kivy.uix.floatlayout import FloatLayout
from kivy.properties import ListProperty, StringProperty, ObjectProperty, NumericProperty
cloud_poly = [150, 50, 109.7597, 112.96, 115.4326, 113.0853, 120.1966, 111.9883, 126.0889, 111.957, 135.0841, 111.957, 138.5944, 112.5525, 145.7403, 115.5301, 150.5357, 120.3256, 155.5313, 125.5938, 160.8438, 130.5, 165.7813, 132.5, 171.8125, 132.3438, 177.5, 128.4688, 182.1531, 121.499, 185.1438, 114.0406, 185.9181, 108.5649, 186.2226, 102.5978, 187.8059, 100.2231, 193.2257, 100.1622, 197.6712, 101.8671, 202.6647, 104.1809, 207.1102, 105.8858, 214.2351, 105.0333, 219.3747, 102.8301, 224.0413, 98.7589, 225.7798, 93.7272, 226.0, 86.875, 222.9375, 81.0625, 218.3508, 76.0867, 209.8301, 70.809, 198.7806, 66.136, 189.7651, 62.2327, 183.6082, 56.6252, 183.2784, 50.5778, 190.9155, 42.7294, 196.847, 36.1343, 197.7339, 29.9272, 195.572, 23.443, 191.25, 15.9803, 184.0574, 9.5882, 175.8811, 3.9951, 165.7992, 3.4419, 159.0369, 7.437, 152.5205, 14.8125, 147.4795, 24.2162, 142.4385, 29.0103, 137.0287, 30.9771, 127.156, 27.4818, 119.1371, 20.0388, 112.182, 11.369, 104.6541, 7.1976, 97.208, 6.2979, 88.9437, 9.8149, 80.3433, 17.3218, 76.5924, 26.5452, 78.1678, 37.0432, 83.5068, 47.1104, 92.8529, 58.3561, 106.3021, 69.2978, 108.9615, 73.9329, 109.0375, 80.6955, 104.4713, 88.6708, 100.6283, 95.7483, 100.1226, 101.5114, 102.8532, 107.2745, 105.685, 110.9144, 109.7597, 112.96]

class BaseShape(Widget):
    """(internal) Base class for moving with touches or calls."""
    _old_pos = ListProperty([0, 0])
    _old_touch = ListProperty([0, 0])
    _new_touch = ListProperty([0, 0])
    name = StringProperty('')
    poly = ListProperty([])
    shape = ObjectProperty()
    poly_len = NumericProperty(0)
    shape_len = NumericProperty(0)
    debug_collider = ObjectProperty()
    debug_collider_len = NumericProperty(0)

    def __init__(self, **kwargs):
        if False:
            while True:
                i = 10
        "Create a shape with size [100, 100]\n        and give it a label if it's named.\n        "
        super(BaseShape, self).__init__(**kwargs)
        self.size_hint = (None, None)
        self.add_widget(Label(text=self.name))

    def move_label(self, x, y, *args):
        if False:
            i = 10
            return i + 15
        'Move label with shape name as the only child.'
        self.children[0].pos = [x, y]

    def move_collider(self, offset_x, offset_y, *args):
        if False:
            return 10
        'Move debug collider when the shape moves.'
        points = self.debug_collider.points[:]
        for i in range(0, self.debug_collider_len, 2):
            points[i] += offset_x
            points[i + 1] += offset_y
        self.debug_collider.points = points

    def on_debug_collider(self, instance, value):
        if False:
            i = 10
            return i + 15
        "Recalculate length of collider points' array."
        self.debug_collider_len = len(value.points)

    def on_poly(self, instance, value):
        if False:
            print('Hello World!')
        "Recalculate length of polygon points' array."
        self.poly_len = len(value)

    def on_shape(self, instance, value):
        if False:
            print('Hello World!')
        "Recalculate length of Mesh vertices' array."
        self.shape_len = len(value.vertices)

    def on_pos(self, instance, pos):
        if False:
            while True:
                i = 10
        'Move polygon and its Mesh on each position change.\n        This event is above all and changes positions of the other\n        children-like components, so that a simple::\n\n            shape.pos = (100, 200)\n\n        would move everything, not just the widget itself.\n        '
        offset_x = self._new_touch[0] - self._old_touch[0]
        offset_y = self._new_touch[1] - self._old_touch[1]
        if not offset_x and (not offset_y):
            offset_x = pos[0] - self._old_pos[0]
            offset_y = pos[1] - self._old_pos[1]
            self._old_pos = pos
        for i in range(0, self.poly_len, 2):
            self.poly[i] += offset_x
            self.poly[i + 1] += offset_y
        if self.name:
            self.move_label(*pos)
        if self.debug_collider is not None:
            self.move_collider(offset_x, offset_y)
        if self.shape is None:
            return
        points = self.shape.vertices[:]
        for i in range(0, self.shape_len, 2):
            points[i] += offset_x
            points[i + 1] += offset_y
        self.shape.vertices = points

    def on_touch_move(self, touch, *args):
        if False:
            for i in range(10):
                print('nop')
        'Move shape with dragging.'
        if touch.grab_current is not self:
            return
        (x, y) = touch.pos
        new_pos = [x, y]
        self._new_touch = new_pos
        self._old_touch = [touch.px, touch.py]
        offset_x = self._new_touch[0] - self._old_touch[0]
        offset_y = self._new_touch[1] - self._old_touch[1]
        self.pos = [self.x + offset_x, self.y + offset_y]

    def shape_collide(self, x, y, *args):
        if False:
            for i in range(10):
                print('nop')
        'Point to polygon collision through a list of points.'
        poly = self.poly
        if not poly:
            return False
        n = self.poly_len
        inside = False
        p1x = poly[0]
        p1y = poly[1]
        for i in range(0, n + 2, 2):
            p2x = poly[i % n]
            p2y = poly[(i + 1) % n]
            if y > min(p1y, p2y) and y <= max(p1y, p2y) and (x <= max(p1x, p2x)):
                if p1y != p2y:
                    xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                if p1x == p2x or x <= xinters:
                    inside = not inside
            (p1x, p1y) = (p2x, p2y)
        return inside

class RegularShape(BaseShape):
    """Starting from center and creating edges around for i.e.:
    regular triangles, squares, regular pentagons, up to "circle".
    """

    def __init__(self, edges=3, color=None, **kwargs):
        if False:
            while True:
                i = 10
        super(RegularShape, self).__init__(**kwargs)
        if edges < 3:
            raise Exception('Not enough edges! (3+ only)')
        color = color or [random() for i in range(3)]
        rad_edge = pi * 2 / float(edges)
        r_x = self.width / 2.0
        r_y = self.height / 2.0
        poly = []
        vertices = []
        for i in range(edges):
            x = cos(rad_edge * i) * r_x + self.center_x
            y = sin(rad_edge * i) * r_y + self.center_y
            poly.extend([x, y])
            vertices.extend([x, y, 0, 0])
        with self.canvas:
            Color(rgba=(color[0], color[1], color[2], 0.6))
            self.shape = Mesh(pos=self.pos, vertices=vertices, indices=list(range(edges)), mode='triangle_fan')
        self.poly = poly

    def on_touch_down(self, touch, *args):
        if False:
            for i in range(10):
                print('nop')
        if self.shape_collide(*touch.pos):
            touch.grab(self)

class MeshShape(BaseShape):
    """Starting from a custom origin and custom points, draw
    a convex Mesh shape with both touch and shape collisions.

    .. note::

        To get the points, use e.g. Pen tool from your favorite
        graphics editor and export it to a human readable format.
    """

    def __init__(self, color=None, **kwargs):
        if False:
            i = 10
            return i + 15
        super(MeshShape, self).__init__(**kwargs)
        color = color or [random() for i in range(3)]
        min_x = 10000
        min_y = 10000
        max_x = 0
        max_y = 0
        poly = [50, 50, 0, 0, 100, 0, 100, 100, 0, 100] if not self.poly else self.poly
        poly = [round(p / 1.5, 4) for p in poly]
        poly_len = len(poly)
        vertices = []
        vertices_len = 0
        for i in range(0, poly_len, 2):
            min_x = poly[i] if poly[i] < min_x else min_x
            min_y = poly[i + 1] if poly[i + 1] < min_y else min_y
            max_x = poly[i] if poly[i] > max_x else max_x
            max_y = poly[i + 1] if poly[i + 1] > max_y else max_y
            vertices_len += 4
            vertices.extend([poly[i], poly[i + 1], 0, 0])
        (poly_center_x, poly_center_y) = [(max_x - min_x) / 2.0, (max_y - min_y) / 2.0]
        dec_x = self.center_x - poly_center_x - min_x
        dec_y = self.center_y - poly_center_y - min_y
        for i in range(0, poly_len, 2):
            poly[i] += dec_x
            poly[i + 1] += dec_y
        for i in range(0, vertices_len, 4):
            vertices[i] += dec_x
            vertices[i + 1] += dec_y
        with self.canvas:
            Color(rgba=(color[0], color[1], color[2], 0.6))
            self.shape = Mesh(pos=self.pos, vertices=vertices, indices=list(range(int(poly_len / 2.0))), mode='triangle_fan')
        self.poly = poly

    def on_touch_down(self, touch, *args):
        if False:
            for i in range(10):
                print('nop')
        if self.shape_collide(*touch.pos):
            touch.grab(self)

class Collisions(App):

    def __init__(self, **kwargs):
        if False:
            i = 10
            return i + 15
        super(Collisions, self).__init__(**kwargs)
        self.register_event_type('on_collision')

    def collision_circles(self, shapes=None, distance=100, debug=False, *args):
        if False:
            i = 10
            return i + 15
        "Simple circle <-> circle collision between the shapes i.e. there's\n        a simple line between the centers of the two shapes and the collision\n        is only about measuring distance -> 1+ radii intersections.\n        "
        if not hasattr(self, 'combins'):
            self.combins = list(combinations(shapes, 2))
        for com in self.combins:
            x = (com[0].center_x - com[1].center_x) ** 2
            y = (com[0].center_y - com[1].center_y) ** 2
            if sqrt(x + y) <= distance:
                self.dispatch('on_collision', (com[0], com[1]))
        if not debug:
            return
        for shape in shapes:
            if shape.debug_collider is not None:
                continue
            d = distance / 2.0
            (cx, cy) = shape.center
            points = [(cx + d * cos(i), cy + d * sin(i)) for i in range(44)]
            points = [p for ps in points for p in ps]
            with shape.canvas:
                Color(rgba=(0, 1, 0, 1))
                shape.debug_collider = Point(points=points)

    def on_collision(self, pair, *args):
        if False:
            return 10
        'Dispatched when objects collide, gives back colliding objects\n        as a "pair" argument holding their instances.\n        '
        print('Collision {} x {}'.format(pair[0].name, pair[1].name))

    def build(self):
        if False:
            for i in range(10):
                print('nop')
        scene = FloatLayout()
        shapes = [RegularShape(name='Shape {}'.format(x), edges=x) for x in range(3, 13)]
        shapes.append(MeshShape(name='DefaultMesh'))
        shapes.append(MeshShape(name='Cloud', poly=cloud_poly))
        shapes.append(MeshShape(name='3QuarterCloud', poly=cloud_poly[:110]))
        for shape in shapes:
            shape.pos = [randint(50, i - 50) for i in Window.size]
            scene.add_widget(shape)
        Clock.schedule_interval(lambda *t: self.collision_circles(shapes, debug=True), 0.1)
        return scene
if __name__ == '__main__':
    Collisions().run()