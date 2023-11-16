from random import sample, randint
from string import ascii_lowercase
from kivy.app import App
from kivy.lang import Builder
from kivy.uix.boxlayout import BoxLayout
kv = "\n<Row@RecycleKVIDsDataViewBehavior+BoxLayout>:\n    canvas.before:\n        Color:\n            rgba: 0.5, 0.5, 0.5, 1\n        Rectangle:\n            size: self.size\n            pos: self.pos\n    value: ''\n    Label:\n        id: name\n    Label:\n        text: root.value\n\n<Test>:\n    canvas:\n        Color:\n            rgba: 0.3, 0.3, 0.3, 1\n        Rectangle:\n            size: self.size\n            pos: self.pos\n    rv: rv\n    orientation: 'vertical'\n    GridLayout:\n        cols: 3\n        rows: 2\n        size_hint_y: None\n        height: dp(108)\n        padding: dp(8)\n        spacing: dp(16)\n        Button:\n            text: 'Populate list'\n            on_press: root.populate()\n        Button:\n            text: 'Sort list'\n            on_press: root.sort()\n        Button:\n            text: 'Clear list'\n            on_press: root.clear()\n        BoxLayout:\n            spacing: dp(8)\n            Button:\n                text: 'Insert new item'\n                on_press: root.insert(new_item_input.text)\n            TextInput:\n                id: new_item_input\n                size_hint_x: 0.6\n                hint_text: 'value'\n                padding: dp(10), dp(10), 0, 0\n        BoxLayout:\n            spacing: dp(8)\n            Button:\n                text: 'Update first item'\n                on_press: root.update(update_item_input.text)\n            TextInput:\n                id: update_item_input\n                size_hint_x: 0.6\n                hint_text: 'new value'\n                padding: dp(10), dp(10), 0, 0\n        Button:\n            text: 'Remove first item'\n            on_press: root.remove()\n\n    RecycleView:\n        id: rv\n        scroll_type: ['bars', 'content']\n        scroll_wheel_distance: dp(114)\n        bar_width: dp(10)\n        viewclass: 'Row'\n        RecycleBoxLayout:\n            default_size: None, dp(56)\n            default_size_hint: 1, None\n            size_hint_y: None\n            height: self.minimum_height\n            orientation: 'vertical'\n            spacing: dp(2)\n"
Builder.load_string(kv)

class Test(BoxLayout):

    def populate(self):
        if False:
            i = 10
            return i + 15
        self.rv.data = [{'name.text': ''.join(sample(ascii_lowercase, 6)), 'value': str(randint(0, 2000))} for x in range(50)]

    def sort(self):
        if False:
            for i in range(10):
                print('nop')
        self.rv.data = sorted(self.rv.data, key=lambda x: x['name.text'])

    def clear(self):
        if False:
            print('Hello World!')
        self.rv.data = []

    def insert(self, value):
        if False:
            i = 10
            return i + 15
        self.rv.data.insert(0, {'name.text': value or 'default value', 'value': 'unknown'})

    def update(self, value):
        if False:
            while True:
                i = 10
        if self.rv.data:
            self.rv.data[0]['name.text'] = value or 'default new value'
            self.rv.refresh_from_data()

    def remove(self):
        if False:
            return 10
        if self.rv.data:
            self.rv.data.pop(0)

class TestApp(App):

    def build(self):
        if False:
            i = 10
            return i + 15
        return Test()
if __name__ == '__main__':
    TestApp().run()