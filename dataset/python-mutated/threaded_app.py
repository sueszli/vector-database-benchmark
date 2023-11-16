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
import threading

class MyApp(App):

    def __init__(self, *args):
        if False:
            while True:
                i = 10
        super(MyApp, self).__init__(*args)

    def idle(self):
        if False:
            while True:
                i = 10
        self.lbl.set_text('Thread result:' + str(self.my_thread_result))

    def main(self):
        if False:
            for i in range(10):
                print('nop')
        wid = gui.VBox(width=300, height=200, margin='0px auto')
        self.lbl = gui.Label('Thread result:', width='80%', height='50%')
        self.lbl.style['margin'] = 'auto'
        bt = gui.Button('Stop algorithm', width=200, height=30)
        bt.style['margin'] = 'auto 50px'
        bt.style['background-color'] = 'red'
        wid.append(self.lbl)
        wid.append(bt)
        self.thread_alive_flag = True
        self.my_thread_result = 0
        t = threading.Thread(target=self.my_intensive_long_time_algorithm)
        t.start()
        bt.onclick.do(self.on_button_pressed)
        return wid

    def my_intensive_long_time_algorithm(self):
        if False:
            print('Hello World!')
        while self.thread_alive_flag:
            self.my_thread_result = self.my_thread_result + 1

    def on_button_pressed(self, emitter):
        if False:
            while True:
                i = 10
        self.thread_alive_flag = False

    def on_close(self):
        if False:
            print('Hello World!')
        self.thread_alive_flag = False
        super(MyApp, self).on_close()
if __name__ == '__main__':
    start(MyApp, debug=True, address='0.0.0.0', port=0, update_interval=0.1)