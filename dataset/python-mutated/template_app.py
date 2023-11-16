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
            for i in range(10):
                print('nop')
        res_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'res')
        super(MyApp, self).__init__(*args, static_file_path={'myres': res_path})

    def idle(self):
        if False:
            for i in range(10):
                print('nop')
        ' Idle loop, you can place here custom code,\n             avoid to use infinite iterations, it would stop gui update.\n            This is a Thread safe method where you can update the \n             gui with information from external Threads.\n        '
        pass

    def main(self):
        if False:
            print('Hello World!')
        main_container = gui.VBox(width=300, height=200, style={'margin': '0px auto'})
        return main_container
if __name__ == '__main__':
    start(MyApp, address='0.0.0.0', port=0, start_browser=True, username=None, password=None)