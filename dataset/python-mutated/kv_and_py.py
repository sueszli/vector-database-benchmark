"""
Referring on ids from Python
=============================

This example shows how to refer to an id from a Python file.
"""
import kivy
kivy.require('1.8.0')
from kivy.app import App
from kivy.uix.boxlayout import BoxLayout

class RootWidget(BoxLayout):

    def first_function(self, status):
        if False:
            for i in range(10):
                print('nop')
        print(status)
        if self.ids.my_switch.active is True:
            self.ids.my_label.text = 'Switch is ON'
        else:
            self.ids.my_label.text = 'Switch is OFF'

class TestApp(App):
    pass
if __name__ == '__main__':
    TestApp().run()