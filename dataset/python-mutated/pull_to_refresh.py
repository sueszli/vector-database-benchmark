"""Detecting and acting upon "Pull down actions" in a RecycleView
- When using overscroll or being at the to, a "pull down to refresh" message
  appears
- if the user pulls down far enough, then a refresh is triggered, which adds
  new elements at the top of the list.

"""
from threading import Thread
from time import sleep
from datetime import datetime
from kivy.app import App
from kivy.lang import Builder
from kivy.properties import ListProperty, BooleanProperty
from kivy.metrics import dp
from kivy.clock import mainthread
KV = "\nFloatLayout:\n    Label:\n        opacity: 1 if app.refreshing or rv.scroll_y > 1 else 0\n        size_hint_y: None\n        pos_hint: {'top': 1}\n        text: 'Refreshing…' if app.refreshing else 'Pull down to refresh'\n\n    RecycleView:\n        id: rv\n        data: app.data\n        viewclass: 'Row'\n        do_scroll_y: True\n        do_scroll_x: False\n        on_scroll_y: app.check_pull_refresh(self, grid)\n\n        RecycleGridLayout:\n            id: grid\n            cols: 1\n            size_hint_y: None\n            height: self.minimum_height\n            default_size: 0, 36\n            default_size_hint: 1, None\n\n\n<Row@Label>:\n    _id: 0\n    text: ''\n    canvas:\n        Line:\n            rectangle: self.pos + self.size\n            width: 0.6\n"

class Application(App):
    data = ListProperty([])
    refreshing = BooleanProperty()

    def build(self):
        if False:
            i = 10
            return i + 15
        self.refresh_data()
        return Builder.load_string(KV)

    def check_pull_refresh(self, view, grid):
        if False:
            for i in range(10):
                print('nop')
        'Check the amount of overscroll to decide if we want to trigger the\n        refresh or not.\n        '
        max_pixel = dp(200)
        to_relative = max_pixel / (grid.height - view.height)
        if view.scroll_y <= 1.0 + to_relative or self.refreshing:
            return
        self.refresh_data()

    def refresh_data(self):
        if False:
            i = 10
            return i + 15
        self.refreshing = True
        Thread(target=self._refresh_data).start()

    def _refresh_data(self):
        if False:
            while True:
                i = 10
        sleep(2)
        update_time = datetime.now().strftime('%H:%M:%S')
        self.prepend_data([{'_id': i, 'text': '[{}] hello {}'.format(update_time, i)} for i in range(len(self.data) + 10, len(self.data), -1)])

    @mainthread
    def prepend_data(self, data):
        if False:
            for i in range(10):
                print('nop')
        self.data = data + self.data
        self.refreshing = False
if __name__ == '__main__':
    Application().run()