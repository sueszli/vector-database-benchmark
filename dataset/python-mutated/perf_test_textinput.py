from kivy.app import App
from kivy.uix.floatlayout import FloatLayout
from kivy.lang import Builder
from kivy.resources import resource_find
from kivy.clock import Clock
import timeit
Builder.load_string("\n<PerfApp>:\n    value: 0\n    but: but.__self__\n    slider: slider\n    text_input: text_input\n    BoxLayout:\n        orientation: 'vertical'\n        TextInput:\n            id: text_input\n        BoxLayout:\n            orientation: 'vertical'\n            size_hint: 1, .2\n            BoxLayout:\n                Button:\n                    id: but\n                    text: 'Start Test'\n                    on_release: root.start_test() if self.text == 'Start Test'                    else ''\n            Slider:\n                id: slider\n                min: 0\n                max: 100\n                value: root.value\n")

class PerfApp(App, FloatLayout):

    def build(self):
        if False:
            for i in range(10):
                print('nop')
        return self

    def __init__(self, **kwargs):
        if False:
            i = 10
            return i + 15
        super(PerfApp, self).__init__(**kwargs)
        self.tests = []
        tests = (self.load_large_text, self.stress_insert, self.stress_del, self.stress_selection)
        for test in tests:
            but = type(self.but)(text=test.__name__)
            self.but.parent.add_widget(but)
            but.test = test
            self.tests.append(but)
        self.test_done = True

    def load_large_text(self, *largs):
        if False:
            i = 10
            return i + 15
        print('loading uix/textinput.py....')
        self.test_done = False
        fd = open(resource_find('uix/textinput.py'), 'r')
        print('putting text in textinput')

        def load_text(*l):
            if False:
                for i in range(10):
                    print('nop')
            self.text_input.text = fd.read()
        t = timeit.Timer(load_text)
        ttk = t.timeit(1)
        fd.close()
        import resource
        print('mem usage after test')
        print(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 'MB')
        print('------------------------------------------')
        print('Loaded', len(self.text_input._lines), 'lines', ttk, 'secs')
        print('------------------------------------------')
        self.test_done = True

    def stress_del(self, *largs):
        if False:
            return 10
        self.test_done = False
        text_input = self.text_input
        self.lt = len_text = len(text_input.text)
        target = len_text - 210 * 9
        self.tot_time = 0
        ev = None

        def dlt(*l):
            if False:
                for i in range(10):
                    print('nop')
            if len(text_input.text) <= target:
                ev.cancel()
                print('Done!')
                m_len = len(text_input._lines)
                print('deleted 210 characters 9 times')
                import resource
                print('mem usage after test')
                print(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 'MB')
                print('total lines in text input:', m_len)
                print('--------------------------------------')
                print('total time elapsed:', self.tot_time)
                print('--------------------------------------')
                self.test_done = True
                return
            text_input.select_text(self.lt - 220, self.lt - 10)
            text_input.delete_selection()
            self.lt -= 210
            text_input.scroll_y -= 100
            self.tot_time += l[0]
            ev()
        ev = Clock.create_trigger(dlt)
        ev()

    def stress_insert(self, *largs):
        if False:
            print('Hello World!')
        self.test_done = False
        text_input = self.text_input
        text_input.select_all()
        text_input.copy(text_input.selection_text)
        text_input.cursor = text_input.get_cursor_from_index(text_input.selection_to)
        len_text = len(text_input._lines)
        self.tot_time = 0
        ev = None

        def pste(*l):
            if False:
                print('Hello World!')
            if len(text_input._lines) >= len_text * 9:
                ev.cancel()
                print('Done!')
                m_len = len(text_input._lines)
                print('pasted', len_text, 'lines', round((m_len - len_text) / len_text), 'times')
                import resource
                print('mem usage after test')
                print(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 'MB')
                print('total lines in text input:', m_len)
                print('--------------------------------------')
                print('total time elapsed:', self.tot_time)
                print('--------------------------------------')
                self.test_done = True
                return
            self.tot_time += l[0]
            text_input.paste()
            ev()
        ev = Clock.create_trigger(pste)
        ev()

    def stress_selection(self, *largs):
        if False:
            while True:
                i = 10
        self.test_done = False
        text_input = self.text_input
        self.tot_time = 0
        old_selection_from = text_input.selection_from - 210
        ev = None

        def pste(*l):
            if False:
                while True:
                    i = 10
            if text_input.selection_from >= old_selection_from:
                ev.cancel()
                print('Done!')
                import resource
                print('mem usage after test')
                print(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 'MB')
                print('--------------------------------------')
                print('total time elapsed:', self.tot_time)
                print('--------------------------------------')
                self.test_done = True
                return
            text_input.select_text(text_input.selection_from - 1, text_input.selection_to)
            ev()
        ev = Clock.create_trigger(pste)
        ev()

    def start_test(self, *largs):
        if False:
            return 10
        self.but.text = 'test started'
        self.slider.max = len(self.tests)
        ev = None

        def test(*l):
            if False:
                while True:
                    i = 10
            if self.test_done:
                try:
                    but = self.tests[int(self.slider.value)]
                    self.slider.value += 1
                    but.state = 'down'
                    print('=====================')
                    print('Test:', but.text)
                    print('=====================')
                    but.test(but)
                except IndexError:
                    for but in self.tests:
                        but.state = 'normal'
                    self.but.text = 'Start Test'
                    self.slider.value = 0
                    print('===================')
                    print('All Tests Completed')
                    print('===================')
                    ev.cancel()
        ev = Clock.schedule_interval(test, 1)
if __name__ in ('__main__',):
    PerfApp().run()