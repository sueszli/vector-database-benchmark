"""
Label textsize
============

This example shows how to size a Label to its content (texture_size) and how
setting text_size controls text wrapping.
"""
from kivy.app import App
from kivy.clock import Clock
from kivy.lang import Builder
from kivy.properties import StringProperty, NumericProperty, BooleanProperty
_example_title_text = 'A Tale of Two Cities, by Charles Dickens\n'
_example_text = 'It was the best of times, it was the worst of times,\nit was the age of wisdom, it was the age of foolishness, it was the epoch of\nbelief, it was the epoch of incredulity, it was the season of Light, it was\nthe season of Darkness, it was the spring of hope, it was the winter of\ndespair, we had everything before us, we had nothing before us, we were all\ngoing direct to Heaven, we were all going direct the other way - in short,\nthe period was so far like the present period, that some of its noisiest\nauthorities insisted on its being received, for good or for evil, in the\nsuperlative degree of comparison only.\n'
_kv_code = '\nBoxLayout:\n    orientation: \'vertical\'\n\n    HeadingLabel:\n        text: \'These modify all demonstration Labels\'\n\n    StackLayout:\n        # Button is a subclass of Label and can be sized\n        # to text in the same way\n\n        Button:\n            text: \'Reset\'\n            on_press: app.reset_words()\n\n        ToggleButton:\n            text: \'Shorten\'\n            on_state:\n                app.shorten=self.state==\'down\'\n\n        ToggleButton:\n            text: \'max_lines=3\'\n            on_state:\n                app.max_lines=3 if self.state==\'down\' else 0\n\n        Spinner:\n            text: \'bottom\'\n            values: \'bottom\', \'middle\', \'top\'\n            on_text: app.valign=self.text\n\n        Spinner:\n            text: \'left\'\n            values: \'left\', \'center\', \'right\', \'justify\'\n            on_text: app.halign=self.text\n\n    GridLayout:\n        id: grid_layout\n        cols: 2\n        height: cm(6)\n        size_hint_y: None\n\n        HeadingLabel:\n            text: "Default, no text_size set"\n\n        HeadingLabel:\n            text: \'text_size bound to size\'\n\n        DemoLabel:\n            id: left_content\n            disabled_color: 0, 0, 0, 0\n\n        DemoLabel:\n            id: right_content\n            text_size: self.size\n            padding: dp(6), dp(6)\n\n    ToggleButton:\n        text: \'Disable left\'\n        on_state:\n            left_content.disabled=self.state==\'down\'\n\n    # Need one Widget without size_hint_y: None, so that BoxLayout fills\n    # available space.\n    HeadingLabel:\n        text: \'text_size width set, size bound to texture_size\'\n        text_size: self.size\n        size_hint_y: 1\n\n    DemoLabel:\n        id: bottom_content\n        # This Label wraps and expands its height to fit the text because\n        # only text_size width is set and the Label size binds to texture_size.\n        text_size: self.width, None\n        size: self.texture_size\n        padding: mm(4), mm(4)\n        size_hint_y: None\n\n# The column heading labels have their width set by the parent,\n# but determine their height from the text.\n<HeadingLabel@Label>:\n    bold: True\n    padding: dp(6), dp(4)\n    valign: \'bottom\'\n    height: self.texture_size[1]\n    text_size: self.width, None\n    size_hint_y: None\n\n<ToggleButton,Button>:\n    padding: dp(10), dp(8)\n    size_hint: None, None\n    size: self.texture_size\n\n# This inherits Button and the modifications above, so reset size\n<Spinner>:\n    size: sp(68), self.texture_size[1]\n\n<DemoLabel@Label>:\n    halign: app.halign\n    valign: app.valign\n    shorten: app.shorten\n    max_lines: app.max_lines\n\n    canvas:\n        Color:\n            rgb: 68/255.0, 164/255.0, 201/255.0\n        Line:\n            rectangle: self.x, self.y, self.width, self.height\n\n<StackLayout>:\n    size_hint_y: None\n    spacing: dp(6)\n    padding: dp(6), dp(4)\n    height: self.minimum_height\n'

class LabelTextureSizeExample(App):
    valign = StringProperty('bottom')
    halign = StringProperty('left')
    shorten = BooleanProperty(False)
    max_lines = NumericProperty(0)

    def build(self):
        if False:
            for i in range(10):
                print('nop')
        self._add_word_ev = None
        return Builder.load_string(_kv_code)

    def on_start(self):
        if False:
            print('Hello World!')
        widget_ids = self.root.ids
        self.text_content_widgets = (widget_ids.left_content, widget_ids.right_content, widget_ids.bottom_content)
        self.reset_words()

    def reset_words(self):
        if False:
            return 10
        if self._add_word_ev is not None:
            self._add_word_ev.cancel()
            self._add_word_ev = None
        for content_widget in self.text_content_widgets:
            content_widget.text = _example_title_text
        self.words = (word for word in _example_text.split())
        self.add_word()

    def add_word(self, dt=None):
        if False:
            for i in range(10):
                print('nop')
        try:
            word = next(self.words)
        except StopIteration:
            return
        for content_widget in self.text_content_widgets:
            content_widget.text += word + ' '
        pause_time = 0.03 * len(word)
        if word.endswith(','):
            pause_time += 0.6
        self._add_word_ev = Clock.schedule_once(self.add_word, pause_time)
if __name__ == '__main__':
    LabelTextureSizeExample().run()