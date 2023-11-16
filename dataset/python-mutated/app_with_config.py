from kivy.app import App
from kivy.lang import Builder
from kivy.properties import ConfigParserProperty
KV = "\nFloatLayout:\n    BoxLayout:\n        size_hint: .5, .5\n        pos_hint: {'center': (.5, .5)}\n\n        orientation: 'vertical'\n\n        TextInput:\n            text: app.text\n            on_text: app.text = self.text\n\n        Slider:\n            min: 0\n            max: 100\n            value: app.number\n            on_value: app.number = self.value\n"

class ConfigApp(App):
    number = ConfigParserProperty(0, 'general', 'number', 'app', val_type=float)
    text = ConfigParserProperty('', 'general', 'text', 'app', val_type=str)

    def build_config(self, config):
        if False:
            i = 10
            return i + 15
        config.setdefaults('general', {'number': 0, 'text': 'test'})

    def build(self):
        if False:
            while True:
                i = 10
        return Builder.load_string(KV)
if __name__ == '__main__':
    ConfigApp().run()