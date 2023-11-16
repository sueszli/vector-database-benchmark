from kivy.lang import Builder
from kivy.app import App
from kivy.network.urlrequest import UrlRequest
from kivy.properties import NumericProperty, StringProperty, DictProperty
import json
KV = "\n#:import json json\n#:import C kivy.utils.get_color_from_hex\n\nBoxLayout:\n    orientation: 'vertical'\n    Label:\n        text: 'see https://httpbin.org for more information'\n\n    TextInput:\n        id: ti\n        hint_text: 'type url or select from dropdown'\n        size_hint_y: None\n        height: 48\n        multiline: False\n        foreground_color:\n            (\n            C('000000')\n            if (self.text).startswith('http') else\n            C('FF2222')\n            )\n\n    BoxLayout:\n        size_hint_y: None\n        height: 48\n        Spinner:\n            id: spinner\n            text: 'select'\n            values:\n                [\n                'http://httpbin.org/ip',\n                'http://httpbin.org/user-agent',\n                'http://httpbin.org/headers',\n                'http://httpbin.org/delay/3',\n                'http://httpbin.org/image/jpeg',\n                'http://httpbin.org/image/png',\n                'https://httpbin.org/delay/3',\n                'https://httpbin.org/image/jpeg',\n                'https://httpbin.org/image/png',\n                ]\n            on_text: ti.text = self.text\n\n        Button:\n            text: 'GET'\n            on_press: app.fetch_content(ti.text)\n            disabled: not (ti.text).startswith('http')\n            size_hint_x: None\n            width: 50\n\n    Label:\n        text: str(app.status)\n\n    TextInput:\n        readonly: True\n        text: app.result_text\n\n    Image:\n        source: app.result_image\n        nocache: True\n\n    TextInput\n        readonly: True\n        text: json.dumps(app.headers, indent=2)\n"

class UrlExample(App):
    status = NumericProperty()
    result_text = StringProperty()
    result_image = StringProperty()
    headers = DictProperty()

    def build(self):
        if False:
            for i in range(10):
                print('nop')
        return Builder.load_string(KV)

    def fetch_content(self, url):
        if False:
            return 10
        self.cleanup()
        UrlRequest(url, on_success=self.on_success, on_failure=self.on_failure, on_error=self.on_error)

    def cleanup(self):
        if False:
            return 10
        self.result_text = ''
        self.result_image = ''
        self.status = 0
        self.headers = {}

    def on_success(self, req, result):
        if False:
            return 10
        self.cleanup()
        headers = req.resp_headers
        content_type = headers.get('content-type', headers.get('Content-Type'))
        if content_type.startswith('image/'):
            fn = 'tmpfile.{}'.format(content_type.split('/')[1])
            with open(fn, 'wb') as f:
                f.write(result)
            self.result_image = fn
        elif isinstance(result, dict):
            self.result_text = json.dumps(result, indent=2)
        else:
            self.result_text = result
        self.status = req.resp_status
        self.headers = headers

    def on_failure(self, req, result):
        if False:
            i = 10
            return i + 15
        self.cleanup()
        self.result_text = result
        self.status = req.resp_status
        self.headers = req.resp_headers

    def on_error(self, req, result):
        if False:
            while True:
                i = 10
        self.cleanup()
        self.result_text = str(result)
if __name__ == '__main__':
    UrlExample().run()