from kivy.app import App
from kivy.uix.screenmanager import ScreenManager, Screen
from kivy.properties import NumericProperty
from kivy.lang import Builder
Builder.load_string('\n#:import random random.random\n#:import SlideTransition kivy.uix.screenmanager.SlideTransition\n#:import SwapTransition kivy.uix.screenmanager.SwapTransition\n#:import WipeTransition kivy.uix.screenmanager.WipeTransition\n#:import FadeTransition kivy.uix.screenmanager.FadeTransition\n#:import RiseInTransition kivy.uix.screenmanager.RiseInTransition\n#:import FallOutTransition kivy.uix.screenmanager.FallOutTransition\n#:import NoTransition kivy.uix.screenmanager.NoTransition\n\n<CustomScreen>:\n    hue: random()\n    canvas:\n        Color:\n            hsv: self.hue, .5, .3\n        Rectangle:\n            size: self.size\n\n    Label:\n        font_size: 42\n        text: root.name\n\n    Button:\n        text: \'Next screen\'\n        size_hint: None, None\n        pos_hint: {\'right\': 1}\n        size: 150, 50\n        on_release: root.manager.current = root.manager.next()\n\n    Button:\n        text: \'Previous screen\'\n        size_hint: None, None\n        size: 150, 50\n        on_release: root.manager.current = root.manager.previous()\n\n    BoxLayout:\n        size_hint: .5, None\n        height: 250\n        pos_hint: {\'center_x\': .5}\n        orientation: \'vertical\'\n\n        Button:\n            text: \'Use SlideTransition with "up" direction\'\n            on_release: root.manager.transition =                         SlideTransition(direction="up")\n\n        Button:\n            text: \'Use SlideTransition with "down" direction\'\n            on_release: root.manager.transition =                         SlideTransition(direction="down")\n\n        Button:\n            text: \'Use SlideTransition with "left" direction\'\n            on_release: root.manager.transition =                         SlideTransition(direction="left")\n\n        Button:\n            text: \'Use SlideTransition with "right" direction\'\n            on_release: root.manager.transition =                         SlideTransition(direction="right")\n\n        Button:\n            text: \'Use SwapTransition\'\n            on_release: root.manager.transition = SwapTransition()\n\n        Button:\n            text: \'Use WipeTransition\'\n            on_release: root.manager.transition = WipeTransition()\n\n        Button:\n            text: \'Use FadeTransition\'\n            on_release: root.manager.transition = FadeTransition()\n\n        Button:\n            text: \'Use FallOutTransition\'\n            on_release: root.manager.transition = FallOutTransition()\n\n        Button:\n            text: \'Use RiseInTransition\'\n            on_release: root.manager.transition = RiseInTransition()\n        Button:\n            text: \'Use NoTransition\'\n            on_release: root.manager.transition = NoTransition(duration=0)\n')

class CustomScreen(Screen):
    hue = NumericProperty(0)

class ScreenManagerApp(App):

    def build(self):
        if False:
            print('Hello World!')
        root = ScreenManager()
        for x in range(4):
            root.add_widget(CustomScreen(name='Screen %d' % x))
        return root
if __name__ == '__main__':
    ScreenManagerApp().run()