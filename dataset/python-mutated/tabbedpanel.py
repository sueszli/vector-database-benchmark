"""
TabbedPanel
============

Test of the widget TabbedPanel.
"""
from kivy.app import App
from kivy.uix.tabbedpanel import TabbedPanel
from kivy.lang import Builder
Builder.load_string('\n\n<Test>:\n    size_hint: .5, .5\n    pos_hint: {\'center_x\': .5, \'center_y\': .5}\n    do_default_tab: False\n\n    TabbedPanelItem:\n        text: \'first tab\'\n        Label:\n            text: \'First tab content area\'\n    TabbedPanelItem:\n        text: \'tab2\'\n        BoxLayout:\n            Label:\n                text: \'Second tab content area\'\n            Button:\n                text: \'Button that does nothing\'\n    TabbedPanelItem:\n        text: \'tab3\'\n        RstDocument:\n            text:\n                \'\\n\'.join(("Hello world", "-----------",\n                "You are in the third tab."))\n\n')

class Test(TabbedPanel):
    pass

class TabbedPanelApp(App):

    def build(self):
        if False:
            while True:
                i = 10
        return Test()
if __name__ == '__main__':
    TabbedPanelApp().run()