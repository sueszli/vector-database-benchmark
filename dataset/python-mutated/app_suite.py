"""
Suite of Application Builders
=============================

This explores different methods of starting an application. If you run
this without a command line parameter, you should see a menu in your terminal.
You can also run this with a 'r' parameter to pick a random method.
There are lots of logging options to make this easier to debug: the execution
order may not be obvious. Each time you run the command, only one kivy
application is created.

This uses the file testkvfile.kv and the file app_suite_data/testkvdir.kv.

"""
from __future__ import print_function
import sys
import re
from random import choice
import kivy
kivy.require('1.8.0')
from kivy.app import App
from kivy.uix.button import Button
from kivy.lang import Builder
from kivy.uix.floatlayout import FloatLayout
print('** In main program, done with imports')

class TestBuildApp(App):
    """ Use build() function to return a widget. """

    def build(self):
        if False:
            for i in range(10):
                print('nop')
        '   Build called by kivy when an App is started.\n              Called after trying to load a .kv file.\n              Returns a new Button as a root widget.\n              '
        print('** inside build()')
        return Button(text='hello from TestBuildApp')

class TestKVFileApp(App):
    """
    Empty class, but name used to find .kv file. The filename is the lowercase
    version of the class, i.e. 'testkvfileapp.kv'. If not found, it strips
    off the final 'app', i.e. 'testkvfile.kv'. If not file is found, and no
    other method sets the self.root, the program will run with an empty screen.
    """
    pass

class TestKVDirApp(App):
    """
    Empty class except for setting class variable kv_directory.
    This directory sets the directory in which to search for the .kv file.
    The name of the kv file is still governed by the class name and the .kv
    file should still have one root widget.
    """
    kv_directory = 'app_suite_data'

class TestKVStringApp(App):
    """
    Use a build() function and use the kivy.lang.Builder function to parse up a
    Kivy language string.
    """

    def build(self):
        if False:
            while True:
                i = 10
        '   Called by kivy run().  '
        print('** inside build()')
        widget = Builder.load_string("Button:\n  text: 'hello from TestKVStringApp'")
        print('** widget built')
        return widget

class TestPrebuiltApp(App):
    """ Use the Builder to create a top level widget at the beginning
    of the Python program, then use a dummy class for that widget.
    This costs a bit more in start-up time. """
    kv = "<Prebuilt>\n  Button:\n    text:'hello from TestPrebuiltApp'"
    Builder.load_string(kv)
    print('** in TestPrebuiltApp, class initialization built <Prebuilt>')

    class Prebuilt(FloatLayout):
        """ Empty class to cause setting root to <Prebuilt> tag and
        set inheritance """
        pass

    def build(self):
        if False:
            while True:
                i = 10
        ' called, returns instance matching tag . '
        return self.Prebuilt()

def print_class(class_name):
    if False:
        while True:
            i = 10
    ' Read this file and print the section with the class name specified.)'
    filename = sys.argv[0]
    with open(filename) as f:
        data = f.read()
        regex = '^(class ' + class_name + '\\b.*?)^\\S'
        match = re.search(regex, data, flags=re.MULTILINE | re.DOTALL)
        if match:
            print(match.group(1))
if __name__ == '__main__':
    dash = '-' * 40
    arg = sys.argv[1][0].lower() if len(sys.argv) > 1 else 'h'
    print(dash)
    if arg == 'r':
        arg = choice('bfds')
    if arg == 'b':
        print_class('TestBuildApp')
        TestBuildApp().run()
    elif arg == 'f':
        print_class('TestKVFileApp')
        TestKVFileApp().run()
    elif arg == 'd':
        print_class('TestKVDirApp')
        TestKVDirApp().run()
    elif arg == 's':
        print_class('TestKVStringApp')
        TestKVStringApp().run()
    elif arg == 'p':
        print_class('TestPrebuiltApp')
        TestPrebuiltApp().run()
    else:
        print('\nThis demo runs different application windows based on a command line argument.\n\nTry using one of these:\n   b - Use build() method to return a widget\n   d - Use a kv file from a different directory\n   f - Use a kv file with the widget object\n   p - Use prebuilt widget inside a layout\n   s - Use a kivy language string to create the widget\n   r - pick one of the options at random.\n\n   h - show this help message.\n\n   After closing the application window, this program will exit.\n   While the run() method does return, kivy cannot run another\n   application window after one has been closed.\n ')
    print(dash)
    print('This program is gratified to be of use.')