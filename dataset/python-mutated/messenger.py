from kivy.app import App
from kivy.lang import Builder
from kivy.clock import Clock
from kivy.properties import ListProperty
from kivy.animation import Animation
from kivy.metrics import dp
KV = "\n#:import RGBA kivy.utils.rgba\n\n<ImageButton@ButtonBehavior+Image>:\n    size_hint: None, None\n    size: self.texture_size\n\n    canvas.before:\n        PushMatrix\n        Scale:\n            origin: self.center\n            x: .75 if self.state == 'down' else 1\n            y: .75 if self.state == 'down' else 1\n\n    canvas.after:\n        PopMatrix\n\nBoxLayout:\n    orientation: 'vertical'\n    padding: dp(5), dp(5)\n    RecycleView:\n        id: rv\n        data: app.messages\n        viewclass: 'Message'\n        do_scroll_x: False\n\n        RecycleBoxLayout:\n            id: box\n            orientation: 'vertical'\n            size_hint_y: None\n            size: self.minimum_size\n            default_size_hint: 1, None\n            # magic value for the default height of the message\n            default_size: 0, 38\n            key_size: '_size'\n\n    FloatLayout:\n        size_hint_y: None\n        height: 0\n        Button:\n            size_hint_y: None\n            height: self.texture_size[1]\n            opacity: 0 if not self.height else 1\n            text:\n                (\n                'go to last message'\n                if rv.height < box.height and rv.scroll_y > 0 else\n                ''\n                )\n            pos_hint: {'pos': (0, 0)}\n            on_release: app.scroll_bottom()\n\n    BoxLayout:\n        size_hint: 1, None\n        size: self.minimum_size\n        TextInput:\n            id: ti\n            size_hint: 1, None\n            height: min(max(self.line_height, self.minimum_height), 150)\n            multiline: False\n\n            on_text_validate:\n                app.send_message(self)\n\n        ImageButton:\n            source: 'data/logo/kivy-icon-48.png'\n            on_release:\n                app.send_message(ti)\n\n<Message@FloatLayout>:\n    message_id: -1\n    bg_color: '#223344'\n    side: 'left'\n    text: ''\n    size_hint_y: None\n    _size: 0, 0\n    size: self._size\n    text_size: None, None\n    opacity: min(1, self._size[0])\n\n    Label:\n        text: root.text\n        padding: 10, 10\n        size_hint: None, 1\n        size: self.texture_size\n        text_size: root.text_size\n\n        on_texture_size:\n            app.update_message_size(\n            root.message_id,\n            self.texture_size,\n            root.width,\n            )\n\n        pos_hint:\n            (\n            {'x': 0, 'center_y': .5}\n            if root.side == 'left' else\n            {'right': 1, 'center_y': .5}\n            )\n\n        canvas.before:\n            Color:\n                rgba: RGBA(root.bg_color)\n            RoundedRectangle:\n                size: self.texture_size\n                radius: dp(5), dp(5), dp(5), dp(5)\n                pos: self.pos\n\n        canvas.after:\n            Color:\n            Line:\n                rounded_rectangle: self.pos + self.texture_size + [dp(5)]\n                width: 1.01\n"

class MessengerApp(App):
    messages = ListProperty()

    def build(self):
        if False:
            while True:
                i = 10
        return Builder.load_string(KV)

    def add_message(self, text, side, color):
        if False:
            print('Hello World!')
        self.messages.append({'message_id': len(self.messages), 'text': text, 'side': side, 'bg_color': color, 'text_size': [None, None]})

    def update_message_size(self, message_id, texture_size, max_width):
        if False:
            i = 10
            return i + 15
        if max_width == 0:
            return
        one_line = dp(50)
        if texture_size[0] >= max_width * 2 / 3:
            self.messages[message_id] = {**self.messages[message_id], 'text_size': (max_width * 2 / 3, None)}
        elif texture_size[0] < max_width * 2 / 3 and texture_size[1] > one_line:
            self.messages[message_id] = {**self.messages[message_id], 'text_size': (max_width * 2 / 3, None), '_size': texture_size}
        else:
            self.messages[message_id] = {**self.messages[message_id], '_size': texture_size}

    @staticmethod
    def focus_textinput(textinput):
        if False:
            for i in range(10):
                print('nop')
        textinput.focus = True

    def send_message(self, textinput):
        if False:
            print('Hello World!')
        text = textinput.text
        textinput.text = ''
        self.add_message(text, 'right', '#223344')
        self.focus_textinput(textinput)
        Clock.schedule_once(lambda *args: self.answer(text), 1)
        self.scroll_bottom()

    def answer(self, text, *args):
        if False:
            while True:
                i = 10
        self.add_message('do you really think so?', 'left', '#332211')

    def scroll_bottom(self):
        if False:
            i = 10
            return i + 15
        rv = self.root.ids.rv
        box = self.root.ids.box
        if rv.height < box.height:
            Animation.cancel_all(rv, 'scroll_y')
            Animation(scroll_y=0, t='out_quad', d=0.5).start(rv)
if __name__ == '__main__':
    MessengerApp().run()