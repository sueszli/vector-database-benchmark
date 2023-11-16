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
            return 10
        res_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'res')
        super(MyApp, self).__init__(*args, static_file_path={'my_res_folder': res_path})

    def main(self):
        if False:
            i = 10
            return i + 15
        resource_image = gui.Image('/my_res_folder:mine.png', width='30', height='30')
        local_image = gui.Image(gui.load_resource('./res/mine.png'), width='30', height='30')
        standard_widget = gui.Widget(width='30', height='30', style={'background-repeat': 'no-repeat'})
        standard_widget2 = gui.Widget(width='30', height='30', style={'background-repeat': 'no-repeat'})
        standard_widget.style['background-image'] = gui.to_uri('/my_res_folder:mine.png')
        standard_widget2.style['background-image'] = gui.to_uri(gui.load_resource('./res/mine.png'))
        print(gui.to_uri('/my_res_folder:mine.png'))
        print(gui.to_uri(gui.load_resource('./res/mine.png')))
        main_container = gui.VBox(children=[resource_image, local_image, standard_widget, standard_widget2], width=200, height=300, style={'margin': '0px auto'})
        return main_container
if __name__ == '__main__':
    start(MyApp, address='0.0.0.0', port=0, start_browser=True, username=None, password=None)