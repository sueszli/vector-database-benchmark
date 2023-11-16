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
' This example shows the possibility to stop the server by App.close() method.\n    With multiple_clients, the server will stop as soon as all the clients \n    will disconnect.\n'
import remi.gui as gui
from remi import start, App

class MyApp(App):

    def main(self, name='world'):
        if False:
            return 10
        wid = gui.VBox(width=300, height=200, margin='0px auto')
        bt = gui.Button('Close App', width=200, height=30)
        bt.style['margin'] = 'auto 50px'
        bt.style['background-color'] = 'red'
        bt.onclick.do(self.on_button_pressed)
        wid.append(bt)
        return wid

    def on_button_pressed(self, _):
        if False:
            i = 10
            return i + 15
        self.close()

    def on_close(self):
        if False:
            while True:
                i = 10
        ' Overloading App.on_close event allows to perform some \n             activities before app termination.\n        '
        print("I'm going to be closed.")
        super(MyApp, self).on_close()
if __name__ == '__main__':
    start(MyApp)