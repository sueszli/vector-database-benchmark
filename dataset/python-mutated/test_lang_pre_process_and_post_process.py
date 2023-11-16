import unittest
import textwrap
from collections import defaultdict

class TrackCallbacks(object):
    kv_pre_events = []
    'Stores values added during the pre event dispatched callbacks.\n    '
    kv_applied_events = []
    'Stores values added during the applied event dispatched callbacks.\n    '
    kv_post_events = []
    'Stores values added during the post event dispatched callbacks.\n    '
    events_in_pre = []
    'List of expected events that should be in kv_pre_events after all the\n    callbacks has been executed.\n    '
    events_in_applied = []
    'List of expected events that should be in kv_applied_events after all\n    the callbacks has been executed.\n    '
    events_in_post = []
    'List of expected events that should be in kv_post_events after all the\n    callbacks has been executed.\n    '
    instantiated_widgets = []
    'Whenever a widget of this class is instantiated, it is added to this\n    list, which is class specific.\n\n    It lets us iterate through all the instance of this class and assert for\n    all of them as needed.\n    '
    root_widget = None
    'The expected root widget in the kv rule as dispatched in on_kv_applied.\n    '
    base_widget = None
    'The expected base widget as dispatched in on_kv_post.\n    '
    actual_root_widget = None
    'The actual root widget in the kv rule as dispatched in on_kv_applied.\n    '
    actual_base_widget = None
    'The actual base widget as dispatched in on_kv_post.\n    '
    name = 'none'
    'Optional name given to the widget to help it identify during a test\n    failure.\n    '
    my_roots_expected_ids = {}
    "Dictionary containing the expected ids as stored in the root\n    widget's `ids`. The root being this widget's root widget from kv.\n    "
    actual_ids = {}
    "Dictionary containing the actual ids as stored in the root\n    widget's `ids`. The root being this widget's root widget from kv.\n\n    The ids is saved here during the `on_kv_post` callback.\n    "
    expected_prop_values = {}
    'A dict of property names and the values they are expected to have\n    during the on_kv_post dispatch.\n    '
    actual_prop_values = {}
    'A dict of property names and the values they actually had\n    during the on_kv_post dispatch.\n    '

    def __init__(self, name='none', **kwargs):
        if False:
            return 10
        self.kv_pre_events = self.kv_pre_events[:]
        self.kv_applied_events = self.kv_applied_events[:]
        self.kv_post_events = self.kv_post_events[:]
        self.events_in_pre = self.events_in_pre[:]
        self.events_in_applied = self.events_in_applied[:]
        self.events_in_post = self.events_in_post[:]
        self.name = name
        super(TrackCallbacks, self).__init__(**kwargs)
        self.instantiated_widgets.append(self)

    def add(self, name, event):
        if False:
            return 10
        'Add name to the list of the names added in the callbacks for this\n        event.\n        '
        events = getattr(self, 'kv_{}_events'.format(event))
        events.append(name)

    @classmethod
    def check(cls, testcase):
        if False:
            return 10
        'Checks that all the widgets of this class pass all the assertions.\n        '
        for widget in cls.instantiated_widgets:
            for event in ('pre', 'applied', 'post'):
                cls.check_event(widget, event, testcase)
            expected = {k: v.__self__ for (k, v) in widget.my_roots_expected_ids.items()}
            actual = {k: v.__self__ for (k, v) in widget.actual_ids.items()}
            testcase.assertEqual(expected, actual)
            testcase.assertIs(widget.root_widget and widget.root_widget.__self__, widget.actual_root_widget and widget.actual_root_widget.__self__, 'expected "{}", got "{}" instead for root_widget'.format(widget.root_widget and widget.root_widget.name, widget.actual_root_widget and widget.actual_root_widget.name))
            testcase.assertIs(widget.base_widget and widget.base_widget.__self__, widget.actual_base_widget and widget.actual_base_widget.__self__, 'expected "{}", got "{}" instead for base_widget'.format(widget.base_widget and widget.base_widget.name, widget.actual_base_widget and widget.actual_base_widget.name))
            testcase.assertEqual(widget.expected_prop_values, widget.actual_prop_values)

    @staticmethod
    def check_event(widget, event_name, testcase):
        if False:
            for i in range(10):
                print('nop')
        'Check that the names are added as expected for this event.\n        '
        events = getattr(widget, 'kv_{}_events'.format(event_name))
        should_be_in = getattr(widget, 'events_in_{}'.format(event_name))
        counter = defaultdict(int)
        for name in events:
            counter[name] += 1
        for (name, value) in counter.items():
            testcase.assertEqual(value, 1, '"{}" was present "{}" times for event "{}" for widget "{} ({})"'.format(name, value, event_name, widget.name, widget))
        testcase.assertEqual(set(should_be_in), set(events), 'Expected and actual event callbacks do not match for event "{}" for widget "{} ({})"'.format(event_name, widget.name, widget))

    @staticmethod
    def get_base_class():
        if False:
            while True:
                i = 10
        'The base class to use for widgets during testing so we can use\n        this class variables to ease testing.\n        '
        from kivy.uix.widget import Widget

        class TestEventsBase(TrackCallbacks, Widget):
            __events__ = ('on_kv_pre', 'on_kv_applied')
            instantiated_widgets = []
            events_in_pre = [1]
            events_in_applied = [1]
            events_in_post = [1]

            def on_kv_pre(self):
                if False:
                    i = 10
                    return i + 15
                self.add(1, 'pre')

            def on_kv_applied(self, root_widget):
                if False:
                    return 10
                self.add(1, 'applied')
                self.actual_root_widget = root_widget

            def on_kv_post(self, base_widget):
                if False:
                    while True:
                        i = 10
                self.add(1, 'post')
                self.actual_base_widget = base_widget
                self.actual_prop_values = {k: getattr(self, k) for k in self.expected_prop_values}
                if self.actual_root_widget is not None:
                    self.actual_ids = dict(self.actual_root_widget.ids)

            def apply_class_lang_rules(self, root=None, **kwargs):
                if False:
                    for i in range(10):
                        print('nop')
                self.dispatch('on_kv_pre')
                super(TestEventsBase, self).apply_class_lang_rules(root=root, **kwargs)
                self.dispatch('on_kv_applied', root)
        return TestEventsBase

    def __repr__(self):
        if False:
            return 10
        module = type(self).__module__
        try:
            qualname = type(self).__qualname__
        except AttributeError:
            qualname = ''
        return '<Name: "{}" {}.{} object at {}>'.format(self.name, module, qualname, hex(id(self)))

class TestKvEvents(unittest.TestCase):

    def test_pure_python_auto_binding(self):
        if False:
            while True:
                i = 10

        class TestEventsPureAuto(TrackCallbacks.get_base_class()):
            instantiated_widgets = []
        widget = TestEventsPureAuto()
        widget.root_widget = None
        widget.base_widget = widget
        TestEventsPureAuto.check(self)

    def test_pure_python_callbacks(self):
        if False:
            while True:
                i = 10

        class TestEventsPure(TrackCallbacks.get_base_class()):
            instantiated_widgets = []
            events_in_pre = [1, 2]
            events_in_applied = [1, 2]
            events_in_post = [1, 2]

            def __init__(self, **kwargs):
                if False:
                    while True:
                        i = 10
                self.fbind('on_kv_pre', lambda _: self.add(2, 'pre'))
                self.fbind('on_kv_applied', lambda _, x: self.add(2, 'applied'))
                self.fbind('on_kv_post', lambda _, x: self.add(2, 'post'))
                super(TestEventsPure, self).__init__(**kwargs)
        widget = TestEventsPure()
        widget.root_widget = None
        widget.base_widget = widget
        widget.fbind('on_kv_pre', lambda _: widget.add(3, 'pre'))
        widget.fbind('on_kv_applied', lambda _, x: widget.add(3, 'applied'))
        widget.fbind('on_kv_post', lambda _, x: widget.add(3, 'post'))
        TestEventsPure.check(self)

    def test_instantiate_from_kv(self):
        if False:
            for i in range(10):
                print('nop')
        from kivy.lang import Builder

        class TestEventsFromKV(TrackCallbacks.get_base_class()):
            instantiated_widgets = []
        widget = Builder.load_string('TestEventsFromKV')
        self.assertIsInstance(widget, TestEventsFromKV)
        widget.root_widget = widget
        widget.base_widget = widget
        widget.check(self)

    def test_instantiate_from_kv_with_event(self):
        if False:
            while True:
                i = 10
        from kivy.lang import Builder

        class TestEventsFromKVEvent(TrackCallbacks.get_base_class()):
            instantiated_widgets = []
        widget = Builder.load_string(textwrap.dedent("\n        TestEventsFromKVEvent:\n            events_in_post: [1, 2]\n            on_kv_pre: self.add(2, 'pre')\n            on_kv_applied: self.add(2, 'applied')\n            on_kv_post: self.add(2, 'post')\n            root_widget: self\n            base_widget: self\n        "))
        self.assertIsInstance(widget, TestEventsFromKVEvent)
        widget.check(self)

    def test_instantiate_from_kv_with_child(self):
        if False:
            return 10
        from kivy.lang import Builder

        class TestEventsFromKVChild(TrackCallbacks.get_base_class()):
            instantiated_widgets = []
        widget = Builder.load_string(textwrap.dedent("\n        TestEventsFromKVChild:\n            events_in_post: [1, 2]\n            on_kv_pre: self.add(2, 'pre')\n            on_kv_applied: self.add(2, 'applied')\n            on_kv_post: self.add(2, 'post')\n            root_widget: self\n            base_widget: self\n            name: 'root'\n            my_roots_expected_ids: {'child_widget': child_widget}\n            TestEventsFromKVChild:\n                events_in_post: [1, 2]\n                on_kv_pre: self.add(2, 'pre')\n                on_kv_applied: self.add(2, 'applied')\n                on_kv_post: self.add(2, 'post')\n                root_widget: root\n                base_widget: root\n                name: 'child'\n                id: child_widget\n                my_roots_expected_ids: {'child_widget': self}\n        "))
        self.assertIsInstance(widget, TestEventsFromKVChild)
        widget.check(self)

    def test_instantiate_from_kv_with_child_inherit(self):
        if False:
            print('Hello World!')
        from kivy.lang import Builder

        class TestEventsFromKVChildInherit(TrackCallbacks.get_base_class()):
            instantiated_widgets = []
        widget = Builder.load_string(textwrap.dedent("\n        <TestEventsFromKVChildInherit2@TestEventsFromKVChildInherit>:\n            on_kv_pre: self.add(3, 'pre')\n            on_kv_applied: self.add(3, 'applied')\n            on_kv_post: self.add(3, 'post')\n\n        <TestEventsFromKVChildInherit3@TestEventsFromKVChildInherit2>:\n            on_kv_pre: self.add(4, 'pre')\n            on_kv_applied: self.add(4, 'applied')\n            on_kv_post: self.add(4, 'post')\n            some_value: 'fruit'\n            TestEventsFromKVChildInherit2:\n                events_in_applied: [1, 2, 3]\n                events_in_post: [1, 2, 3, 4]\n                on_kv_pre: self.add(4, 'pre')\n                on_kv_applied: self.add(4, 'applied')\n                on_kv_post: self.add(4, 'post')\n                root_widget: root\n                base_widget: self.parent.parent\n                name: 'third child'\n                id: third_child\n                my_roots_expected_ids: {'third_child': self}\n\n        <TestEventsFromKVChildInherit>:\n            on_kv_pre: self.add(2, 'pre')\n            on_kv_applied: self.add(2, 'applied')\n            on_kv_post: self.add(2, 'post')\n            another_value: 'apple'\n\n        TestEventsFromKVChildInherit:\n            events_in_applied: [1, 2]\n            events_in_post: [1, 2, 3]\n            on_kv_pre: self.add(3, 'pre')\n            on_kv_applied: self.add(3, 'applied')\n            on_kv_post: self.add(3, 'post')\n            root_widget: self\n            base_widget: self\n            name: 'root'\n            my_roots_expected_ids:                 {'second_child': second_child, 'first_child': first_child}\n            TestEventsFromKVChildInherit:\n                events_in_applied: [1, 2]\n                events_in_post: [1, 2, 3]\n                on_kv_pre: self.add(3, 'pre')\n                on_kv_applied: self.add(3, 'applied')\n                on_kv_post: self.add(3, 'post')\n                root_widget: root\n                base_widget: root\n                name: 'first child'\n                id: first_child\n                my_roots_expected_ids:                     {'second_child': second_child, 'first_child': self}\n            TestEventsFromKVChildInherit3:\n                events_in_applied: [1, 2, 3, 4]\n                events_in_post: [1, 2, 3, 4, 5]\n                on_kv_pre: self.add(5, 'pre')\n                on_kv_applied: self.add(5, 'applied')\n                on_kv_post: self.add(5, 'post')\n                root_widget: root\n                base_widget: root\n                name: 'second child'\n                some_value: first_child.another_value\n                expected_prop_values: {'some_value': 'apple'}\n                id: second_child\n                my_roots_expected_ids:                     {'second_child': self, 'first_child': first_child}\n        "))
        widget.check(self)