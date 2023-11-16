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
' Here is an "Hello World" application showing a simple interaction\n     with the user.\n'
import remi.gui as gui
from remi import start, App

class MyApp(App):

    def __init__(self, *args):
        if False:
            print('Hello World!')
        super(MyApp, self).__init__(*args)

    def main(self):
        if False:
            print('Hello World!')
        wid = gui.VBox(width=300, height=200)
        self.lbl = gui.Label('Hello\n test', width='80%', height='50%', style={'white-space': 'pre'})
        bt = gui.Button('Press me!', width=200, height=30)
        bt.onclick.do(self.on_button_pressed)
        wid.append(self.lbl)
        wid.append(bt)
        return wid

    def on_button_pressed(self, emitter):
        if False:
            return 10
        self.lbl.set_text('Hello World!')
if __name__ == '__main__':
    start(MyApp, debug=True, address='0.0.0.0', port=0)