import kivy
kivy.require('1.0.8')
from kivy.app import App
from kivy.uix.button import Button
from kivy.uix.scrollview import ScrollView
from kivy.uix.gridlayout import GridLayout

class ScrollViewApp(App):

    def build(self):
        if False:
            print('Hello World!')
        layout = GridLayout(cols=1, padding=10, spacing=10, size_hint=(None, None), width=500)
        layout.bind(minimum_height=layout.setter('height'))
        for i in range(30):
            btn = Button(text=str(i), size=(480, 40), size_hint=(None, None))
            layout.add_widget(btn)
        root = ScrollView(size_hint=(None, None), size=(500, 320), pos_hint={'center_x': 0.5, 'center_y': 0.5}, do_scroll_x=False)
        root.add_widget(layout)
        return root
if __name__ == '__main__':
    ScrollViewApp().run()