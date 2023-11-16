"""
Widget animation
================

This example demonstrates creating and applying a multi-part animation to
a button widget. You should see a button labelled 'plop' that will move with
an animation when clicked.
"""
import kivy
kivy.require('1.0.7')
from kivy.animation import Animation
from kivy.app import App
from kivy.uix.button import Button

class TestApp(App):

    def animate(self, instance):
        if False:
            while True:
                i = 10
        animation = Animation(pos=(100, 100), t='out_bounce')
        animation += Animation(pos=(200, 100), t='out_bounce')
        animation &= Animation(size=(500, 500))
        animation += Animation(size=(100, 50))
        animation.start(instance)

    def build(self):
        if False:
            print('Hello World!')
        button = Button(size_hint=(None, None), text='plop', on_press=self.animate)
        return button
if __name__ == '__main__':
    TestApp().run()