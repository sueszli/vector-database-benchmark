import unittest
'ACHTUNG: when testing, never re-use widget names, otherwise the tests\nmay fail as the namespace will remember the names between tests.\n'

class KNSpaceTestCase(unittest.TestCase):

    def test_not_exists(self):
        if False:
            i = 10
            return i + 15
        from kivy.uix.behaviors.knspace import knspace
        self.assertRaises(AttributeError, lambda : knspace.label)

    def test_not_exists_property(self):
        if False:
            return 10
        from kivy.uix.behaviors.knspace import knspace
        self.assertRaises(AttributeError, lambda : knspace.label2)
        knspace.property('label2')
        self.assertIsNone(knspace.label2)

    def test_allow_none(self):
        if False:
            i = 10
            return i + 15
        from kivy.uix.behaviors.knspace import knspace, KNSpaceBehavior
        from kivy.uix.widget import Widget

        class MyWidget(KNSpaceBehavior, Widget):
            pass
        knspace.label3 = 1
        knspace.label3 = None
        w = MyWidget()
        w.knspace = knspace
        w.knspace = None

    def test_name(self):
        if False:
            return 10
        from kivy.uix.behaviors.knspace import knspace, KNSpaceBehavior
        from kivy.uix.widget import Widget

        class MyWidget(KNSpaceBehavior, Widget):
            pass
        w = MyWidget()
        w.knsname = 'label4'
        w.knsname = ''

    def test_proxy_ref(self):
        if False:
            print('Hello World!')
        from kivy.uix.behaviors.knspace import knspace
        from kivy.uix.widget import Widget
        w = Widget()
        knspace.widget1 = w
        self.assertIs(w.proxy_ref, knspace.widget1)
        knspace.widget1 = 55
        self.assertIs(55, knspace.widget1)

    def test_constructor(self):
        if False:
            return 10
        from kivy.uix.behaviors.knspace import knspace, KNSpaceBehavior
        from kivy.uix.widget import Widget

        class MyWidget(KNSpaceBehavior, Widget):
            pass
        w = MyWidget(knsname='construct_name')
        self.assertEqual(knspace.construct_name, w)

    def test_re_assign(self):
        if False:
            return 10
        from kivy.uix.behaviors.knspace import knspace, KNSpaceBehavior
        from kivy.uix.widget import Widget

        class MyWidget(KNSpaceBehavior, Widget):
            pass
        w = MyWidget(knsname='construct_name2')
        self.assertEqual(knspace.construct_name2, w)
        w2 = MyWidget(knsname='construct_name2')
        self.assertEqual(knspace.construct_name2, w2)

    def test_simple(self):
        if False:
            while True:
                i = 10
        from kivy.lang import Builder
        from kivy.uix.behaviors.knspace import knspace
        w = Builder.load_string("\n<NamedLabel@KNSpaceBehavior+Label>\n\nNamedLabel:\n    knsname: 'label5'\n    text: 'Hello'\n")
        self.assertEqual(knspace.label5, w)
        self.assertIs(w, knspace.label5.__self__)
        self.assertIsNot(w, knspace.label5)
        self.assertEqual('Hello', knspace.label5.text)
        w.text = 'Goodbye'
        self.assertEqual('Goodbye', knspace.label5.text)

    def test_simple_multiple_names(self):
        if False:
            print('Hello World!')
        from kivy.lang import Builder
        from kivy.uix.behaviors.knspace import knspace
        w = Builder.load_string("\n<NamedLabel@KNSpaceBehavior+Label>\n\nBoxLayout:\n    NamedLabel:\n        knsname: 'label61'\n        text: 'Hello'\n    NamedLabel:\n        knsname: 'label62'\n        text: 'Goodbye'\n")
        self.assertEqual('Hello', knspace.label61.text)
        self.assertEqual('Goodbye', knspace.label62.text)

    def test_simple_binding(self):
        if False:
            while True:
                i = 10
        from kivy.lang import Builder
        from kivy.uix.behaviors.knspace import knspace
        w = Builder.load_string("\n#:import knspace kivy.uix.behaviors.knspace.knspace\n<NamedLabel@KNSpaceBehavior+Label>\n\nBoxLayout:\n    text: knspace.label7.text if knspace.label7 else ''\n    NamedLabel:\n        knsname: 'label7'\n        text: 'Hello'\n")
        self.assertEqual('Hello', w.text)
        knspace.label7.text = 'Goodbye'
        self.assertEqual('Goodbye', w.text)

    def test_simple_name_change(self):
        if False:
            print('Hello World!')
        from kivy.lang import Builder
        from kivy.uix.behaviors.knspace import knspace
        w = Builder.load_string("\n<NamedLabel@KNSpaceBehavior+Label>\n\nNamedLabel:\n    knsname: 'label8'\n    text: 'Hello'\n")
        self.assertEqual(w, knspace.label8)
        w.knsname = 'named_label8'
        self.assertIsNone(knspace.label8)
        self.assertEqual(w, knspace.named_label8)

    def test_fork_string(self):
        if False:
            return 10
        from kivy.lang import Builder
        from kivy.uix.behaviors.knspace import knspace
        w = Builder.load_string("\n<NamedLabel@KNSpaceBehavior+Label>\n\nBoxLayout:\n    NamedLabel:\n        knspace: 'fork'\n        knsname: 'label9'\n        text: 'Hello'\n    NamedLabel:\n        knspace: 'fork'\n        knsname: 'label9'\n        text: 'Goodbye'\n")
        self.assertEqual(w.children[0].knspace.label9.text, 'Goodbye')
        self.assertEqual(w.children[1].knspace.label9.text, 'Hello')

    def test_fork(self):
        if False:
            return 10
        from kivy.lang import Builder
        from kivy.uix.behaviors.knspace import knspace, KNSpaceBehavior
        from kivy.uix.widget import Widget

        class NamedWidget(KNSpaceBehavior, Widget):
            pass
        nw = NamedWidget()
        w = Widget()
        w2 = Widget()
        before = knspace.fork()
        knspace.widget2 = w
        after = knspace.fork()
        self.assertEqual(knspace.widget2, w)
        self.assertEqual(before.widget2, w)
        self.assertEqual(after.widget2, w)
        child = knspace.fork()
        child2 = knspace.fork()
        grandchild = child.fork()
        child.widget3 = w2
        self.assertEqual(grandchild.widget3, w2)
        self.assertEqual(child.widget3, w2)
        self.assertIsNone(knspace.widget3)
        grandchild.parent = child2
        self.assertIsNone(grandchild.widget3)

    def test_fork_binding(self):
        if False:
            print('Hello World!')
        from kivy.lang import Builder
        from kivy.uix.behaviors.knspace import knspace
        w = Builder.load_string("\n<NamedLabel@KNSpaceBehavior+Label>\n\n<MyComplexWidget@NamedLabel>:\n    knsname: 'root_label'\n    text: 'Hello'\n    NamedLabel:\n        id: child\n        knsname: 'child_label'\n        text: self.knspace.root_label.text if self.knspace.root_label else ''\n\nBoxLayout:\n    MyComplexWidget:\n        knspace: 'fork'\n        id: first\n    MyComplexWidget:\n        knspace: 'fork'\n        id: second\n")
        self.assertEqual(w.ids.first.ids.child.text, 'Hello')
        self.assertEqual(w.ids.second.ids.child.text, 'Hello')
        self.assertEqual(w.ids.first.knspace.child_label.text, 'Hello')
        self.assertEqual(w.ids.second.knspace.child_label.text, 'Hello')
        w.ids.first.text = 'Goodbye'
        self.assertEqual(w.ids.first.ids.child.text, 'Goodbye')
        self.assertEqual(w.ids.second.ids.child.text, 'Hello')
        self.assertEqual(w.ids.first.knspace.child_label.text, 'Goodbye')
        self.assertEqual(w.ids.second.knspace.child_label.text, 'Hello')
        first = w.ids.first.knspace
        w.ids.first.knspace = w.ids.second.knspace
        w.ids.second.knspace = first
        self.assertEqual(w.ids.first.ids.child.text, 'Goodbye')
        self.assertEqual(w.ids.second.ids.child.text, 'Hello')
        self.assertEqual(w.ids.first.knspace.child_label.text, 'Goodbye')
        self.assertEqual(w.ids.second.knspace.child_label.text, 'Hello')
        w.ids.first.text = 'Goodbye2'
        self.assertEqual(w.ids.first.ids.child.text, 'Goodbye2')
        self.assertEqual(w.ids.second.ids.child.text, 'Hello')
        self.assertEqual(w.ids.first.knspace.child_label.text, 'Goodbye2')
        self.assertEqual(w.ids.second.knspace.child_label.text, 'Hello')
        w.ids.first.knspace.root_label.text = 'Goodbye3'
        self.assertEqual(w.ids.first.ids.child.text, 'Goodbye3')
        self.assertEqual(w.ids.second.ids.child.text, 'Hello')
        self.assertEqual(w.ids.first.knspace.child_label.text, 'Goodbye3')
        self.assertEqual(w.ids.second.knspace.child_label.text, 'Hello')