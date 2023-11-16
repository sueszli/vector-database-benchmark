"""
   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
"""
import remi.gui as gui
from remi import start, App
import os

class MyApp(App):

    def __init__(self, *args):
        if False:
            print('Hello World!')
        res_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'res')
        super(MyApp, self).__init__(*args, static_file_path={'myres': res_path})

    def idle(self):
        if False:
            while True:
                i = 10
        ' Idle loop, you can place here custom code,\n             avoid to use infinite iterations, it would stop gui update.\n            This is a Thread safe method where you can update the \n             gui with information from external Threads.\n        '
        pass

    def main(self):
        if False:
            for i in range(10):
                print('nop')
        my_html_head = '\n            '
        my_css_head = '\n            <link rel="stylesheet" href="" type="text/css">\n            '
        my_js_head = '\n            <script></script>\n            '
        self.page.children['head'].add_child('myhtml', my_html_head)
        self.page.children['head'].add_child('mycss', my_css_head)
        self.page.children['head'].add_child('myjs', my_js_head)
        self.page.children['head'].set_icon_file('/res:icon.png')
        main_container = gui.VBox(width=300, height=200, style={'margin': '0px auto'})
        return main_container

    def on_close(self):
        if False:
            for i in range(10):
                print('nop')
        ' Overloading App.on_close event allows to perform some \n             activities before app termination. '
        super(MyApp, self).on_close()

    def onload(self, emitter):
        if False:
            i = 10
            return i + 15
        ' WebPage Event that occurs on webpage loaded '
        super(MyApp, self).onload(emitter)

    def onerror(self, message, source, lineno, colno, error):
        if False:
            while True:
                i = 10
        ' WebPage Event that occurs on webpage errors '
        super(MyApp, self).onerror(message, source, lineno, colno, error)

    def ononline(self, emitter):
        if False:
            for i in range(10):
                print('nop')
        ' WebPage Event that occurs on webpage goes online after a disconnection '
        super(MyApp, self).ononline(emitter)

    def onpagehide(self, emitter):
        if False:
            print('Hello World!')
        ' WebPage Event that occurs on webpage when the user navigates away '
        super(MyApp, self).onpagehide(emitter)

    def onpageshow(self, emitter, width, height):
        if False:
            return 10
        ' WebPage Event that occurs on webpage gets shown '
        super(MyApp, self).onpageshow(emitter, width, height)

    def onresize(self, emitter, width, height):
        if False:
            i = 10
            return i + 15
        ' WebPage Event that occurs on webpage gets resized '
        super(MyApp, self).onresize(emitter, width, height)
if __name__ == '__main__':
    start(MyApp, debug=True, address='0.0.0.0', port=0, start_browser=True, username=None, password=None)