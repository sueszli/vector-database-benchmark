__all__ = 'AnimatedButton'
from kivy.factory import Factory
from kivy.uix.label import Label
from kivy.uix.image import Image
from kivy.properties import StringProperty, OptionProperty, ObjectProperty, BooleanProperty

class AnimatedButton(Label):
    state = OptionProperty('normal', options=('normal', 'down'))
    fit_mode = StringProperty('fill')
    border = ObjectProperty(None)
    anim_delay = ObjectProperty(None)
    background_normal = StringProperty('atlas://data/images/defaulttheme/button')
    texture_background = ObjectProperty(None)
    background_down = StringProperty('atlas://data/images/defaulttheme/button_pressed')

    def __init__(self, **kwargs):
        if False:
            return 10
        super(AnimatedButton, self).__init__(**kwargs)
        self.register_event_type('on_press')
        self.register_event_type('on_release')
        self.border = (16, 16, 16, 16)
        self.img = Image(source=self.background_normal, fit_mode=self.fit_mode, mipmap=True)

        def anim_reset(*l):
            if False:
                for i in range(10):
                    print('nop')
            self.img.anim_delay = self.anim_delay
        self.bind(anim_delay=anim_reset)
        self.anim_delay = 0.1
        self.img.bind(texture=self.on_tex_changed)
        self.on_tex_changed()

        def background_changed(*l):
            if False:
                return 10
            self.img.source = self.background_normal
            self.anim_delay = 0.1
        self.bind(background_normal=background_changed)

    def on_tex_changed(self, *largs):
        if False:
            print('Hello World!')
        self.texture_background = self.img.texture

    def _do_press(self):
        if False:
            while True:
                i = 10
        self.state = 'down'

    def _do_release(self):
        if False:
            return 10
        self.state = 'normal'

    def on_touch_down(self, touch):
        if False:
            i = 10
            return i + 15
        if not self.collide_point(touch.x, touch.y):
            return False
        if repr(self) in touch.ud:
            return False
        touch.grab(self)
        touch.ud[repr(self)] = True
        _animdelay = self.img.anim_delay
        self.img.source = self.background_down
        self.img.anim_delay = _animdelay
        self._do_press()
        self.dispatch('on_press')
        return True

    def on_touch_move(self, touch):
        if False:
            for i in range(10):
                print('nop')
        return repr(self) in touch.ud

    def on_touch_up(self, touch):
        if False:
            while True:
                i = 10
        if touch.grab_current is not self:
            return
        assert repr(self) in touch.ud
        touch.ungrab(self)
        _animdelay = self.img._coreimage.anim_delay
        self.img.source = self.background_normal
        self.anim_delay = _animdelay
        self._do_release()
        self.dispatch('on_release')
        return True

    def on_press(self):
        if False:
            print('Hello World!')
        pass

    def on_release(self):
        if False:
            print('Hello World!')
        pass
Factory.register('AnimatedButton', cls=AnimatedButton)