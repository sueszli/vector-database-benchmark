from kivy.app import App
data = '\n[\n    {\n        "type": "string",\n        "title": "String",\n        "desc": "-",\n        "section": "test",\n        "key": "string"\n    },\n    {\n        "type": "path",\n        "title": "Path",\n        "desc": "-",\n        "section": "test",\n        "key": "path"\n    }\n]\n'

class UnicodeIssueSetting(App):

    def build_config(self, config):
        if False:
            print('Hello World!')
        config.add_section('test')
        config.setdefault('test', 'string', 'Hello world')
        config.setdefault('test', 'path', '/')

    def build(self):
        if False:
            i = 10
            return i + 15
        from kivy.uix.settings import Settings
        s = Settings()
        s.add_json_panel('Test Panel', self.config, data=data)
        return s
if __name__ == '__main__':
    UnicodeIssueSetting().run()