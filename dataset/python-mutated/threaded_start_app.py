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
'\n    This example shows how to start the application as a thread, \n     without stopping the main thread.\n    A label is accessed from the main thread. \n    NOTE:\n        It is important to run the server with parameter multiple_instance=False\n'
import remi
import remi.gui as gui
from remi import start, App, Server
import time
global_app_instance = None

class MyApp(App):
    label = None

    def main(self):
        if False:
            for i in range(10):
                print('nop')
        global global_app_instance
        global_app_instance = self
        main_container = gui.VBox(width=300, height=200, style={'margin': '0px auto'})
        self.label = gui.Label('a label')
        main_container.append(self.label)
        return main_container
if __name__ == '__main__':
    server = remi.Server(MyApp, start=False, address='0.0.0.0', port=0, start_browser=True, multiple_instance=False)
    server.start()
    index = 0
    while True:
        if not global_app_instance is None:
            with global_app_instance.update_lock:
                global_app_instance.label.set_text('%s' % index)
                index = index + 1
        time.sleep(1)