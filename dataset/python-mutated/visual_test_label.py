from kivy.app import runTouchApp
from kivy.uix.gridlayout import GridLayout
from kivy.properties import StringProperty
from kivy.lang import Builder
from kivy.utils import get_hex_from_color, get_random_color
import timeit
import re
import random
from functools import partial

def layout_perf(label, repeat):
    if False:
        i = 10
        return i + 15
    if repeat:
        repeat = int(repeat)
    else:
        return 'None'
    return str(timeit.Timer(label._label.render).repeat(1, repeat))

def layout_real_perf(label, repeat):
    if False:
        while True:
            i = 10
    if repeat:
        repeat = int(repeat)
    else:
        return 'None'
    old_text = label._label.texture
    label._label.texture = label._label.texture_1px
    res = str(timeit.Timer(partial(label._label.render, True)).repeat(1, repeat))
    label._label.texture = old_text
    return res
kv = "\n#:import tlp visual_test_label.layout_perf\n#:import tlrp visual_test_label.layout_real_perf\n\n<TSliderButton@ToggleButton>:\n    size_hint: None, None\n    size: 100, 50\n    group: 'slider'\n    on_press: self.parent.slider.name = self.text if self.state =='down' else    'dummy'\n\n<TSpinner@Spinner>:\n    size_hint: None, None\n    size: 100, 50\n    name: ''\n    on_text: setattr(self.parent.label, self.name, self.text)\n\n<TBoolButton@ToggleButton>:\n    size_hint: None, None\n    size: 100, 50\n    on_state: setattr(self.parent.label, self.text, self.state == 'down')\n\n<TLabel@Label>:\n    size_hint: None, None\n    size: 100, 50\n\n\n<LabelTest>:\n    cols: 1\n    spacing: 10\n    padding: 20\n    TabbedPanel:\n        do_default_tab: False\n        tab_width: self.width / 11 * 3\n        TabbedPanelItem:\n            text: 'Label'\n            BoxLayout:\n                ScrollView:\n                    id: scrollview\n                    Label:\n                        size_hint: None, None\n                        size: self.texture_size\n                        id: label\n                        text: record.text\n                        dummy: 0\n                        canvas:\n                            Color:\n                                rgba: 0, 1, 0, 0.5\n                            Rectangle:\n                                pos: self.pos\n                                size: self.width, self.padding_y\n                            Rectangle:\n                                pos: self.x, self.y + self.height -                                self.padding_y\n                                size: self.width, self.padding_y\n                            Color:\n                                rgba: 0, 0, 1, 0.5\n                            Rectangle:\n                                pos: self.pos\n                                size: self.padding_x, self.height\n                            Rectangle:\n                                pos: self.x + self.width - self.padding_x,                                self.y\n                                size: self.padding_x, self.height\n                Splitter:\n                    sizable_from: 'left'\n                    TextInput:\n                        id: record\n                        text: label.text\n                        text: root.text\n        TabbedPanelItem:\n            text: 'Test performance'\n            BoxLayout:\n                orientation: 'vertical'\n                Label:\n                    text: 'Test timeit performance with current label settings'\n                BoxLayout:\n                    size_hint_y: None\n                    height: 40\n                    padding: [20, 0]\n                    Label:\n                        text: 'Repeat count: '\n                    TextInput:\n                        id: repeat\n                        text: '1000'\n                    Button:\n                        text: 'Go (render - layout)'\n                        on_press: results.text = tlp(label, repeat.text)\n                    Button:\n                        text: 'Go (render_real)'\n                        on_press: results.text = tlrp(label, repeat.text)\n                Label:\n                    id: results\n                    text: 'Results:'\n\n    StackLayout:\n        id: slider_ctrl\n        size_hint_y: None\n        height: self.minimum_height\n        slider: slider\n        label: label\n        TLabel:\n            text: 'halign: '\n        TSpinner:\n            name: 'halign'\n            values: ['left', 'center', 'right', 'justify']\n            text: 'left'\n        TLabel:\n            text: 'valign: '\n        TSpinner:\n            name: 'valign'\n            values: ['top', 'middle', 'center', 'bottom']\n            text: 'bottom'\n        TBoolButton:\n            text: 'markup'\n        TBoolButton:\n            text: 'shorten'\n        TextInput:\n            size_hint: None, None\n            size: 100, 50\n            hint_text: 'split_str'\n            on_text_validate: label.split_str = self.text\n            multiline: False\n        TLabel:\n            text: 'shorten_from: '\n        TSpinner:\n            name: 'shorten_from'\n            values: ['left', 'center', 'right']\n            text: 'right'\n        TBoolButton:\n            text: 'strip'\n            state: 'down'\n        ToggleButton:\n            size_hint: None, None\n            size: 100, 50\n            text: 'random size'\n            on_state: label.text = root.sized_text if self.state == 'down'            else root.text\n        TLabel:\n            text: 'Slider control:'\n        TSliderButton:\n            text: 'font_size'\n        TSliderButton:\n            text: 'line_height'\n        TSliderButton:\n            text: 'max_lines'\n        TSliderButton:\n            text: 'padding_x'\n        TSliderButton:\n            text: 'padding_y'\n        TextInput:\n            size_hint: None, None\n            size: 100, 50\n            hint_text: 'text_size[0]'\n            on_text_validate: label.text_size = (int(self.text) if self.text            else None), label.text_size[1]\n            multiline: False\n        TextInput:\n            size_hint: None, None\n            size: 100, 50\n            hint_text: 'text_size[1]'\n            on_text_validate: label.text_size = label.text_size[0],            (int(self.text) if self.text else None)\n            multiline: False\n        TLabel:\n            text: '<-- w/ validate'\n    Label:\n        size_hint_y: None\n        height: 40\n        color: [0, 1, 0, 1]\n        text_size: self.size\n        text: 'scrollview size: {}, label size: {}, text_size: {}, '        'texture_size: {}, padding: {}'.format(scrollview.size, label.size,        label.text_size, label.texture_size, label.padding)\n\n    BoxLayout:\n        size_hint_y: None\n        height: 40\n        Slider:\n            id: slider\n            range: -10, 200\n            value: 15\n            name: 'dummy'\n            on_value: setattr(label, self.name, self.value)\n        Label:\n            size_hint_x: None\n            width: 50\n            text: str(int(slider.value))\n\n"
text = '\nBecause it would spare your Majesty all fear of future annoyance. If the lady loves her husband, she does not love your Majesty. If she does not love your Majesty, there is no reason why she should interfere with your Majesty\'s plan.\n\n"It is true. And yet--Well! I wish she had been of my own station! What a queen she would have made!" He relapsed into a moody silence, which was not broken until we drew up in Serpentine Avenue.\n\nThe door of Briony Lodge was open, and an elderly woman stood upon the steps. She watched us with a sardonic eye as we stepped from the brougham.\n\nMr. Sherlock Holmes, I believe?" said she.\n\nI am Mr. Holmes," answered my companion, looking at her with a questioning and rather startled gaze.\n\nIndeed! My mistress told me that you were likely to call. She left this morning with her husband by the 5:15 train from Charing Cross for the Continent."\n\n"What!" Sherlock Holmes staggered back, white with chagrin and surprise. "Do you mean that she has left England?"\n\nNever to return.\n\n"And the papers?" asked the King hoarsely. "All is lost."\n'
words = re.split('( +|\\n+)', text)

def annotate(pre, post, callable, words):
    if False:
        return 10
    state = False
    i = random.randint(0, 4)
    while i < len(words):
        if ' ' in words[i] or '\n' in words[i]:
            i += 1
            continue
        if not state:
            words[i] = pre.format(callable(), words[i])
        else:
            words[i] = post.format(words[i])
        state = not state
        i += random.randint(1, 7)
annotate('[size={0}]{1}', '{0}[/size]', partial(random.randint, 8, 24), words)
annotate('[b]{1}', '{0}[/b]', str, words)
annotate('[i]{1}', '{0}[/i]', str, words)
annotate('[color={0}]{1}', '{0}[/color]', lambda : get_hex_from_color(get_random_color()), words)
annotated_text = ''.join(words)

class LabelTest(GridLayout):
    text = StringProperty(text)
    sized_text = StringProperty(annotated_text)
if __name__ in ('__main__',):
    Builder.load_string(kv)
    runTouchApp(LabelTest())