"""
Demonstrate shorten / number of line in label
=============================================

--------------- ------- -------------------------------------------------------
Number of lines Shorten Behavior
--------------- ------- -------------------------------------------------------
0 (unlimited)   False   Default behavior
1               False   Display as much as possible, at least one word
N               False   Display as much as possible
0 (unlimited)   True    Default behavior (as kivy <= 1.7 series)
1               True    Display as much as possible, shorten long word.
N               True    Display as much as possible, shorten long word.
--------------- ------- -------------------------------------------------------

"""
from kivy.app import App
from kivy.lang import Builder
kv = "\n<LabeledSlider@Slider>:\n    step: 1\n    Label:\n        text: '{}'.format(int(root.value))\n        size: self.texture_size\n        top: root.center_y - sp(20)\n        center_x: root.value_pos[0]\n\nBoxLayout:\n    orientation: 'vertical'\n    BoxLayout:\n        spacing: '10dp'\n        padding: '4dp'\n        size_hint_y: None\n        height: '48dp'\n        LabeledSlider:\n            id: slider\n            value: 500\n            min: 25\n            max: root.width\n            on_value: self.value = int(self.value)\n        ToggleButton:\n            id: shorten\n            text: 'Shorten'\n        LabeledSlider:\n            id: max_lines\n            value: 0\n            min: 0\n            max: 5\n\n    AnchorLayout:\n        RelativeLayout:\n            size_hint: None, None\n            size: slider.value, 50\n            canvas:\n                Color:\n                    rgb: .4, .4, .4\n                Rectangle:\n                    size: self.size\n            Label:\n                size_hint: 1, 1\n                text_size: self.size\n                shorten: shorten.state == 'down'\n                max_lines: max_lines.value\n                valign: 'middle'\n                halign: 'center'\n                color: (1, 1, 1, 1)\n                font_size: 22\n                text: 'Michaelangelo Smith'\n"

class ShortenText(App):

    def build(self):
        if False:
            while True:
                i = 10
        return Builder.load_string(kv)
ShortenText().run()