"""
TabbedPanel
============

Test of the widget TabbedPanel showing all capabilities.
"""
from kivy.app import App
from kivy.animation import Animation
from kivy.uix.floatlayout import FloatLayout
from kivy.uix.tabbedpanel import TabbedPanel, TabbedPanelHeader
from kivy.factory import Factory

class StandingHeader(TabbedPanelHeader):
    pass

class CloseableHeader(TabbedPanelHeader):
    pass
Factory.register('StandingHeader', cls=StandingHeader)
Factory.register('CloseableHeader', cls=CloseableHeader)
from kivy.lang import Builder
Builder.load_string('\n<TabShowcase>\n    but: _but\n    Button:\n        id: _but\n        text: \'Press to show Tabbed Panel\'\n        on_release: root.show_tab()\n\n<StandingHeader>\n    color: 0,0,0,0\n    disabled_color: self.color\n    Scatter:\n        do_translation: False\n        do_scale: False\n        do_rotation: False\n        auto_bring_to_front: False\n        rotation: 70\n        size_hint: None, None\n        size: lbl.size\n        center_x: root.center_x\n        center_y: root.center_y\n        Label:\n            id: lbl\n            text: root.text\n            size: root.size\n            color: 1, 1, 1, .5 if self.disabled else 1\n            pos: 0,0\n\n<PanelLeft>\n    size_hint: (.45, .45)\n    pos_hint: {\'center_x\': .25, \'y\': .55}\n    # replace the default tab with our custom tab class\n    default_tab_cls: sh.__class__\n    do_default_tab: True\n    default_tab_content: default_content.__self__\n    tab_width: 40\n    tab_height: 70\n    FloatLayout:\n        RstDocument:\n            id: default_content\n            text: \'\\n\'.join(("Standing tabs", "-------------",                "Tabs in \\\'%s\\\' position" %root.tab_pos))\n        Image:\n            id: tab_2_content\n            source: \'data/images/defaulttheme-0.png\'\n        Image:\n            id: tab_3_content\n            source: \'data/images/image-loading.zip\'\n    StandingHeader:\n        id: sh\n        content: tab_2_content.__self__\n        text: \'tab 2\'\n    StandingHeader:\n        content: tab_3_content\n        text: \'tab 3\'\n\n<CloseableHeader>\n    color: 0,0,0,0\n    disabled_color: self.color\n    # variable tab_width\n    text: \'tabx\'\n    size_hint_x: None\n    width: self.texture_size[0] + 40\n    BoxLayout:\n        pos: root.pos\n        size_hint: None, None\n        size: root.size\n        padding: 3\n        Label:\n            id: lbl\n            text: root.text\n        BoxLayout:\n            size_hint: None, 1\n            orientation: \'vertical\'\n            width: 22\n            Image:\n                source: \'tools/theming/defaulttheme/close.png\'\n                on_touch_down:\n                    if self.collide_point(*args[1].pos) :                        root.panel.remove_widget(root)\n\n\n<PanelRight>\n    tab_pos: \'top_right\'\n    size_hint: (.45, .45)\n    pos_hint: {\'center_x\': .75, \'y\': .55}\n    # replace the default tab with our custom tab\n    default_tab: def_tab\n    # allow variable tab width\n    tab_width: None\n    FloatLayout:\n        RstDocument:\n            id: default_content\n            text: \'\\n\'.join(("Closeable tabs", "---------------",                "- The tabs above are also scrollable",                "- Tabs in \\\'%s\\\' position" %root.tab_pos))\n        Image:\n            id: tab_2_content\n            source: \'data/images/defaulttheme-0.png\'\n        BoxLayout:\n            id: tab_3_content\n            BubbleButton:\n                text: \'Press to add new tab\'\n                on_release: root.add_header()\n            BubbleButton:\n                text: \'Press set this tab as default\'\n                on_release: root.default_tab = tab3\n    CloseableHeader:\n        id: def_tab\n        text: \'default tab\'\n        content:default_content.__self__\n        panel: root\n    CloseableHeader:\n        text: \'tab2\'\n        content: tab_2_content.__self__\n        panel: root\n    CloseableHeader:\n        id: tab3\n        text: \'tab3\'\n        content: tab_3_content.__self__\n        panel: root\n    CloseableHeader:\n        panel: root\n    CloseableHeader:\n        panel: root\n    CloseableHeader:\n        panel: root\n    CloseableHeader:\n        panel: root\n    CloseableHeader:\n        panel: root\n    CloseableHeader:\n        panel: root\n    CloseableHeader:\n        panel: root\n\n<PanelbLeft>\n    tab_pos: \'bottom_left\'\n    size_hint: (.45, .45)\n    pos_hint: {\'center_x\': .25, \'y\': .02}\n    do_default_tab: False\n\n    TabbedPanelItem:\n        id: settings\n        text: \'Settings\'\n        RstDocument:\n            text: \'\\n\'.join(("Normal tabs", "-------------",            "Tabs in \\\'%s\\\' position" %root.tab_pos))\n    TabbedPanelItem:\n        text: \'tab2\'\n        BubbleButton:\n            text: \'switch to settings\'\n            on_press: root.switch_to(settings)\n    TabbedPanelItem:\n        text: \'tab3\'\n        Image:\n            source: \'data/images/image-loading.zip\'\n\n<PanelbRight>\n    tab_pos: \'right_top\'\n    size_hint: (.45, .45)\n    pos_hint: {\'center_x\': .75, \'y\': .02}\n    default_tab: def_tab\n    tab_height: img.width\n    FloatLayout:\n        RstDocument:\n            id: default_content\n            text: \'\\n\'.join(("Image tabs","-------------",                "1. Normal image tab","2. Image with Text","3. Rotated Image",                "4. Tabs in \\\'%s\\\' position" %root.tab_pos))\n        Image:\n            id: tab_2_content\n            source: \'data/images/defaulttheme-0.png\'\n        VideoPlayer:\n            id: tab_3_content\n            source: \'cityCC0.mpg\'\n    TabbedPanelHeader:\n        id: def_tab\n        content:default_content.__self__\n        border: 0, 0, 0, 0\n        background_down: \'cityCC0.png\'\n        background_normal:\'sequenced_images/data/images/info.png\'\n    TabbedPanelHeader:\n        id: tph\n        content: tab_2_content.__self__\n        BoxLayout:\n            pos: tph.pos\n            size: tph.size\n            orientation: \'vertical\'\n            Image:\n                source: \'sequenced_images/data/images/info.png\'                    if tph.state == \'normal\' else \'cityCC0.png\'\n            Label:\n                text: \'text & img\'\n    TabbedPanelHeader:\n        id: my_header\n        content: tab_3_content.__self__\n        Scatter:\n            do_translation: False\n            do_scale: False\n            do_rotation: False\n            auto_bring_to_front: False\n            rotation: 90\n            size_hint: None, None\n            size: img.size\n            center: my_header.center\n            Image:\n                id: img\n                source: \'sequenced_images/data/images/info.png\'                    if my_header.state == \'normal\' else \'cityCC0.png\'\n                size: my_header.size\n                fit_mode: "fill"\n')

class Tp(TabbedPanel):

    def switch_to(self, header):
        if False:
            print('Hello World!')
        anim = Animation(opacity=0, d=0.24, t='in_out_quad')

        def start_anim(_anim, child, in_complete, *lt):
            if False:
                for i in range(10):
                    print('nop')
            _anim.start(child)

        def _on_complete(*lt):
            if False:
                print('Hello World!')
            if header.content:
                header.content.opacity = 0
                anim = Animation(opacity=1, d=0.43, t='in_out_quad')
                start_anim(anim, header.content, True)
            super(Tp, self).switch_to(header)
        anim.bind(on_complete=_on_complete)
        if self.current_tab.content:
            start_anim(anim, self.current_tab.content, False)
        else:
            _on_complete()

class PanelLeft(Tp):
    pass

class PanelRight(Tp):

    def add_header(self):
        if False:
            print('Hello World!')
        self.add_widget(CloseableHeader(panel=self))

class PanelbLeft(Tp):
    pass

class PanelbRight(Tp):
    pass

class TabShowcase(FloatLayout):

    def show_tab(self):
        if False:
            while True:
                i = 10
        if not hasattr(self, 'tab'):
            self.tab = tab = PanelLeft()
            self.add_widget(tab)
            self.tab1 = tab = PanelRight()
            self.add_widget(tab)
            self.tab2 = tab = PanelbRight()
            self.add_widget(tab)
            self.tab3 = tab = PanelbLeft()
            self.add_widget(tab)
            self.but.text = 'Tabs in variable positions, press to change to top_left'
        else:
            values = ('left_top', 'left_mid', 'left_bottom', 'top_left', 'top_mid', 'top_right', 'right_top', 'right_mid', 'right_bottom', 'bottom_left', 'bottom_mid', 'bottom_right')
            index = values.index(self.tab.tab_pos)
            self.tab.tab_pos = self.tab1.tab_pos = self.tab2.tab_pos = self.tab3.tab_pos = values[(index + 1) % len(values)]
            self.but.text = "Tabs in '%s' position," % self.tab.tab_pos + '\n press to change to next pos'

class TestTabApp(App):

    def build(self):
        if False:
            for i in range(10):
                print('nop')
        return TabShowcase()
if __name__ == '__main__':
    TestTabApp().run()