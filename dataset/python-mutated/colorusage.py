from kivy.app import App
from kivy.uix.gridlayout import GridLayout
from kivy.lang import Builder
Builder.load_string('\n#:import hex kivy.utils.get_color_from_hex\n\n<Root>:\n    cols: 2\n    canvas:\n        Color:\n            rgba: 1, 1, 1, 1\n        Rectangle:\n            pos: self.pos\n            size: self.size\n\n    Label:\n        canvas.before:\n            Color:\n                rgb: 39/255., 174/255., 96/255.\n            Rectangle:\n                pos: self.pos\n                size: self.size\n        text: "rgb: 39/255., 174/255., 96/255."\n    Label:\n        canvas.before:\n            Color:\n                rgba: 39/255., 174/255., 96/255., 1\n            Rectangle:\n                pos: self.pos\n                size: self.size\n        text: "rgba: 39/255., 174/255., 96/255., 1"\n    Label:\n        canvas.before:\n            Color:\n                hsv: 145/360., 77.6/100, 68.2/100\n            Rectangle:\n                pos: self.pos\n                size: self.size\n        text: "hsv: 145/360., 77.6/100, 68.2/100"\n    Label:\n        canvas.before:\n            Color:\n                rgba: hex(\'#27ae60\')\n            Rectangle:\n                pos: self.pos\n                size: self.size\n        text: "rgba: hex(\'#27ae60\')"\n')

class Root(GridLayout):
    pass

class ColorusageApp(App):

    def build(self):
        if False:
            while True:
                i = 10
        return Root()
if __name__ == '__main__':
    ColorusageApp().run()