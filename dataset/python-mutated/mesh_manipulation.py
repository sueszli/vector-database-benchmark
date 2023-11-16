"""
Mesh Manipulation Example
=========================

This demonstrates creating a mesh and using it to deform the texture (the
kivy log). You should see the kivy logo with a five sliders to right.
The sliders change the mesh points' x and y offsets, radius, and a
'wobble' deformation's magnitude and speed.

This example is developed in gabriel's blog post at
http://kivy.org/planet/2014/01/kivy-image-manipulations-with-mesh-and-textures/
"""
from kivy.app import App
from kivy.lang import Builder
from kivy.core.image import Image as CoreImage
from kivy.properties import ListProperty, ObjectProperty, NumericProperty
from kivy.clock import Clock
from kivy.core.window import Window
from math import sin, cos, pi
kv = "\nBoxLayout:\n    Widget:\n        canvas:\n            Color:\n                rgba: 1, 1, 1, 1\n            Mesh:\n                vertices: app.mesh_points\n                indices: range(len(app.mesh_points) // 4)\n                texture: app.mesh_texture\n                mode: 'triangle_fan'\n    BoxLayout:\n        orientation: 'vertical'\n        size_hint_x: None\n        width: 100\n        Slider:\n            value: app.offset_x\n            on_value: app.offset_x = args[1]\n            min: -1\n            max: 1\n        Slider:\n            value: app.offset_y\n            on_value: app.offset_y = args[1]\n            min: -1\n            max: 1\n        Slider:\n            value: app.radius\n            on_value: app.radius = args[1]\n            min: 10\n            max: 1000\n        Slider:\n            value: app.sin_wobble\n            on_value: app.sin_wobble = args[1]\n            min: -50\n            max: 50\n        Slider:\n            value: app.sin_wobble_speed\n            on_value: app.sin_wobble_speed = args[1]\n            min: 0\n            max: 50\n            step: 1\n"

class MeshBallApp(App):
    mesh_points = ListProperty([])
    mesh_texture = ObjectProperty(None)
    radius = NumericProperty(500)
    offset_x = NumericProperty(0.5)
    offset_y = NumericProperty(0.5)
    sin_wobble = NumericProperty(0)
    sin_wobble_speed = NumericProperty(0)

    def build(self):
        if False:
            for i in range(10):
                print('nop')
        self.mesh_texture = CoreImage('data/logo/kivy-icon-512.png').texture
        Clock.schedule_interval(self.update_points, 0)
        return Builder.load_string(kv)

    def update_points(self, *args):
        if False:
            return 10
        ' replace self.mesh_points based on current slider positions.\n        Called continuously by a timer because this only sample code.\n        '
        points = [Window.width / 2, Window.height / 2, 0.5, 0.5]
        i = 0
        while i < 2 * pi:
            i += 0.01 * pi
            points.extend([Window.width / 2 + cos(i) * (self.radius + self.sin_wobble * sin(i * self.sin_wobble_speed)), Window.height / 2 + sin(i) * (self.radius + self.sin_wobble * sin(i * self.sin_wobble_speed)), self.offset_x + sin(i), self.offset_y + cos(i)])
        self.mesh_points = points
if __name__ == '__main__':
    MeshBallApp().run()