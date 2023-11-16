"""
A constantly appending log, using recycleview.
- use variable size widgets using the key_size property to cache texture_size
- keeps current position in scroll when new data is happened, unless the view
  is at the very bottom, in which case it follows the log
- works well with mouse scrolling, but less nicely when using swipes,
  improvements welcome.
"""
from random import sample
from string import printable
from time import asctime
from kivy.app import App
from kivy.uix.recycleview import RecycleView
from kivy.lang import Builder
from kivy.properties import NumericProperty, ListProperty
from kivy.clock import Clock
KV = '\n#:import rgba kivy.utils.rgba\n\n<LogLabel@RelativeLayout>:\n    # using a boxlayout here allows us to have better control of the text\n    # position\n    text: \'\'\n    index: None\n    Label:\n        y: 0\n        x: 5\n        size_hint: None, None\n        size: self.texture_size\n        padding: dp(5), dp(5)\n        color: rgba("#3f3e36")\n        text: root.text\n        on_texture_size: app.update_size(root.index, self.texture_size)\n\n        canvas.before:\n            Color:\n                rgba: rgba("#dbeeff")\n            RoundedRectangle:\n                pos: self.pos\n                size: self.size\n                radius: dp(5), dp(5)\n\nBoxLayout:\n    orientation: \'vertical\'\n    spacing: dp(2)\n\n    # a label to help understand what\'s happening with the scrolling\n    Label:\n        size_hint_y: None\n        height: self.texture_size[1]\n        text:\n            \'\'\'height: {height}\n            scrollable_distance: {scrollable_distance}\n            distance_to_top: {distance_to_top}\n            scroll_y: {scroll_y}\n            \'\'\'.format(\n            height=rv.height,\n            scrollable_distance=rv.scrollable_distance,\n            distance_to_top=rv.distance_to_top,\n            scroll_y=rv.scroll_y,\n            )\n\n        canvas.before:\n            Color:\n                rgba: rgba("#77b4ff")\n            RoundedRectangle:\n                pos: self.pos\n                size: self.size\n                radius: dp(5), dp(5)\n\n    FixedRecycleView:\n        id: rv\n        data: app.data\n        viewclass: \'LogLabel\'\n        scrollable_distance: box.height - self.height\n\n        RecycleBoxLayout:\n            id: box\n            orientation: \'vertical\'\n            size_hint_y: None\n            height: self.minimum_height\n            default_size: 0, 48\n            default_size_hint: 1, None\n            spacing: dp(1)\n            key_size: \'cached_size\'\n'

class FixedRecycleView(RecycleView):
    distance_to_top = NumericProperty()
    scrollable_distance = NumericProperty()

    def on_scrollable_distance(self, *args):
        if False:
            return 10
        'This method maintains the position in scroll, by using the saved\n        distance_to_top property to adjust the scroll_y property. Only if we\n        are currently scrolled back.\n        '
        if self.scroll_y > 0:
            self.scroll_y = (self.scrollable_distance - self.distance_to_top) / self.scrollable_distance

    def on_scroll_y(self, *args):
        if False:
            for i in range(10):
                print('nop')
        'Save the distance_to_top everytime we scroll.\n        '
        self.distance_to_top = (1 - self.scroll_y) * self.scrollable_distance

class Application(App):
    data = ListProperty()

    def build(self):
        if False:
            return 10
        Clock.schedule_interval(self.add_log, 0.1)
        return Builder.load_string(KV)

    def add_log(self, dt):
        if False:
            i = 10
            return i + 15
        "Produce random text to append in the log, with the date, we don't\n        want to forget when we babbled incoherently.\n        "
        self.data.append({'index': len(self.data), 'text': f"[{asctime()}]: {''.join(sample(printable, 50))}", 'cached_size': (0, 0)})

    def update_size(self, index, size):
        if False:
            i = 10
            return i + 15
        "Maintain the size data for a log entry, so recycleview can adjust\n        the size computation.\n        As a log entry needs to be displayed to compute its size, it's by\n        default considered to be (0, 0) which is a good enough approximation\n        for such a small widget, but you might want do give a better default\n        value if that doesn't fit your needs.\n        "
        self.data[index]['cached_size'] = size
if __name__ == '__main__':
    Application().run()