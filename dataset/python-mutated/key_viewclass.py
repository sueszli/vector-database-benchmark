"""
A form generator, using random data, but can be data driven (json or whatever)

Shows that you can use the key_viewclass attribute of RecycleView to select a
different Widget for each item.
"""
from random import choice, choices
from string import ascii_lowercase
from kivy.app import App
from kivy.lang import Builder
from kivy import properties as P
KV = "\n<RVTextInput,RVCheckBox,RVSpinner>:\n    size_hint_y: None\n    height: self.minimum_height\n    index: None\n    title: ''\n\n\n<RVTextInput@BoxLayout>:\n    value: ''\n    Label:\n        text: root.title\n        size_hint_y: None\n        height: self.texture_size[1]\n    TextInput:\n        text: root.value\n        on_text: app.handle_update(self.text, root.index)\n        size_hint_y: None\n        height: dp(40)\n        multiline: False\n\n\n<RVCheckBox@BoxLayout>:\n    value: False\n    Label:\n        text: root.title\n        size_hint_y: None\n        height: self.texture_size[1]\n    CheckBox:\n        active: root.value\n        on_active: app.handle_update(self.active, root.index)\n        size_hint_y: None\n        height: dp(40)\n\n\n<RVSpinner@BoxLayout>:\n    value: ''\n    values: []\n    Label:\n        text: root.title\n        size_hint_y: None\n        height: self.texture_size[1]\n    Spinner:\n        text: root.value\n        values: root.values\n        size_hint_y: None\n        height: dp(40)\n        on_text: app.handle_update(self.text, root.index)\n\n\nFloatLayout:\n    RecycleView:\n        id: rv\n        data: app.data\n        key_viewclass: 'widget'\n        size_hint_x: 1\n        RecycleBoxLayout:\n            orientation: 'vertical'\n            size_hint_y: None\n            height: self.minimum_height\n            default_size_hint: 1, None\n\n"

class Application(App):
    """A form manager demonstrating the power of RecycleView's key_viewclass
    property.
    """
    data = P.ListProperty()

    def build(self):
        if False:
            for i in range(10):
                print('nop')
        root = Builder.load_string(KV)
        rv = root.ids.rv
        self.data = [self.create_random_input(rv, index) for index in range(20)]
        return root

    def handle_update(self, value, index):
        if False:
            return 10
        if None not in (index, value):
            self.data[index]['value'] = value

    def create_random_input(self, rv, index):
        if False:
            i = 10
            return i + 15
        return choice((self.create_textinput, self.create_checkbox, self.create_spinner))(rv, index)

    def create_spinner(self, rv, index):
        if False:
            for i in range(10):
                print('nop')
        '\n        create a dict of data for a spinner\n        '
        return {'index': index, 'widget': 'RVSpinner', 'value': '', 'values': [letter * 5 for letter in ascii_lowercase[:5]], 'ready': True}

    def create_checkbox(self, rv, index):
        if False:
            return 10
        '\n        create a dict of data for a checkbox\n        '
        return {'index': index, 'widget': 'RVCheckBox', 'value': choice((True, False)), 'title': ''.join(choices(ascii_lowercase, k=10)), 'ready': True}

    def create_textinput(self, rv, index):
        if False:
            return 10
        '\n        create a dict of data for a textinput\n        '
        return {'index': index, 'widget': 'RVTextInput', 'value': ''.join(choices(ascii_lowercase, k=10)), 'title': ''.join(choices(ascii_lowercase, k=10)), 'ready': True}
if __name__ == '__main__':
    Application().run()