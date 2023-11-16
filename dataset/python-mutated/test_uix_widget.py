from kivy.tests.common import GraphicUnitTest

class UIXWidgetTestCase(GraphicUnitTest):

    def test_default_widgets(self):
        if False:
            return 10
        from kivy.uix.button import Button
        from kivy.uix.slider import Slider
        r = self.render
        r(Button())
        r(Slider())

    def test_button_properties(self):
        if False:
            print('Hello World!')
        from kivy.uix.button import Button
        r = self.render
        r(Button(text='Hello world'))
        r(Button(text='Multiline\ntext\nbutton'))
        r(Button(text='Hello world', font_size=42))
        r(Button(text='This is my first line\nSecond line', halign='center'))

    def test_slider_properties(self):
        if False:
            while True:
                i = 10
        from kivy.uix.slider import Slider
        r = self.render
        r(Slider(value=25))
        r(Slider(value=50))
        r(Slider(value=100))
        r(Slider(min=-100, max=100, value=0))
        r(Slider(orientation='vertical', value=25))
        r(Slider(orientation='vertical', value=50))
        r(Slider(orientation='vertical', value=100))
        r(Slider(orientation='vertical', min=-100, max=100, value=0))

    def test_image_properties(self):
        if False:
            return 10
        from kivy.uix.image import Image
        from os.path import dirname, join
        r = self.render
        filename = join(dirname(__file__), 'test_button.png')
        r(Image(source=filename))

    def test_add_widget_index_0(self):
        if False:
            return 10
        from kivy.uix.widget import Widget
        from kivy.uix.button import Button
        r = self.render
        root = Widget()
        a = Button(text='Hello')
        b = Button(text='World', pos=(50, 10))
        c = Button(text='Kivy', pos=(10, 50))
        root.add_widget(a)
        root.add_widget(b)
        root.add_widget(c, 0)
        r(root)

    def test_add_widget_index_1(self):
        if False:
            i = 10
            return i + 15
        from kivy.uix.widget import Widget
        from kivy.uix.button import Button
        r = self.render
        root = Widget()
        a = Button(text='Hello')
        b = Button(text='World', pos=(50, 10))
        c = Button(text='Kivy', pos=(10, 50))
        root.add_widget(a)
        root.add_widget(b)
        root.add_widget(c, 1)
        r(root)

    def test_add_widget_index_2(self):
        if False:
            for i in range(10):
                print('nop')
        from kivy.uix.widget import Widget
        from kivy.uix.button import Button
        r = self.render
        root = Widget()
        a = Button(text='Hello')
        b = Button(text='World', pos=(50, 10))
        c = Button(text='Kivy', pos=(10, 50))
        root.add_widget(a)
        root.add_widget(b)
        root.add_widget(c, 2)
        r(root)

    def test_widget_root_from_code_with_kv(self):
        if False:
            while True:
                i = 10
        from kivy.lang import Builder
        from kivy.factory import Factory
        from kivy.properties import StringProperty
        from kivy.uix.floatlayout import FloatLayout
        Builder.load_string('\n<UIXWidget>:\n    Label:\n        text: root.title\n\n<BaseWidget>:\n    CallerWidget:\n')

        class CallerWidget(FloatLayout):

            def __init__(self, **kwargs):
                if False:
                    i = 10
                    return i + 15
                super(CallerWidget, self).__init__(**kwargs)
                self.add_widget(UIXWidget(title='Hello World'))

        class NestedWidget(FloatLayout):
            title = StringProperty('aa')

        class UIXWidget(NestedWidget):
            pass

        class BaseWidget(FloatLayout):
            pass
        Factory.register('UIXWidget', cls=UIXWidget)
        Factory.register('CallerWidget', cls=CallerWidget)
        r = self.render
        root = BaseWidget()
        r(root)
    "\n    def test_default_label(self):\n        from kivy.uix.label import Label\n        self.render(Label())\n\n    def test_button_state_down(self):\n        from kivy.uix.button import Button\n        self.render(Button(state='down'))\n\n    def test_label_text(self):\n        from kivy.uix.label import Label\n        self.render(Label(text='Hello world'))\n\n    def test_label_font_size(self):\n        from kivy.uix.label import Label\n        self.render(Label(text='Hello world', font_size=16))\n\n    def test_label_font_size(self):\n        from kivy.uix.label import Label\n        self.render(Label(text='Hello world'))\n    "