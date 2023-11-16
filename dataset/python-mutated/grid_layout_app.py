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
' Here is shown the usage of GridBox layout.\n    The gridbox layouting allows a flexible way to define a layout matrix\n     using GridBox.define_grid, passing as parameter a two dimensional iterable.\n    Each element in the defined grid, is the "key" to address a widget by the \n    GridBox.append method.\n    In this example, the matrix is a list of strings, where each character is used\n     as a "key". A key can occur multiple times in the defined matrix, making the \n     widget to cover a bigger area.\n    The size of each column and row in the grid can be defined by GridBox.style,\n     and the style parameters are \n     {\'grid-template-columns\':\'10% 90%\', \'grid-template-rows\':\'10% 90%\'}.\n'
import remi.gui as gui
from remi import start, App
import os

class MyApp(App):

    def main(self):
        if False:
            print('Hello World!')
        main_container = gui.GridBox(width='100%', height='100%', style={'margin': '0px auto'})
        label = gui.Label('This is a label')
        label.style['background-color'] = 'lightgreen'
        button = gui.Button('Change layout', height='100%')
        button.onclick.do(self.redefine_grid, main_container)
        text = gui.TextInput()
        main_container.set_from_asciiart('\n            |label |button                      |\n            |label |text                        |\n            |label |text                        |\n            |label |text                        |\n            |label |text                        |\n            ', 10, 10)
        main_container.append({'label': label, 'button': button, 'text': text})
        return main_container

    def redefine_grid(self, emitter, container):
        if False:
            for i in range(10):
                print('nop')
        container.define_grid([['text', 'label', 'button'], ['text', '.', '.']])
        container.style.update({'grid-template-columns': '33% 33% 33%', 'grid-template-rows': '50% 50%'})
        container.set_column_gap('0%')
        container.set_row_gap('0%')
        emitter.set_text('Done')
if __name__ == '__main__':
    start(MyApp, debug=True)