""" modal view unit tests. """
from kivy.tests import async_run, UnitKivyApp
from math import isclose

def modal_app():
    if False:
        while True:
            i = 10
    ' test app factory function. '
    from kivy.app import App
    from kivy.uix.button import Button
    from kivy.uix.modalview import ModalView

    class ModalButton(Button):
        """ button used as root widget to test touch. """
        modal = None

        def on_touch_down(self, touch):
            if False:
                return 10
            ' touch down event handler. '
            assert self.modal._window is None
            assert not self.modal._is_open
            return super(ModalButton, self).on_touch_down(touch)

        def on_touch_move(self, touch):
            if False:
                print('Hello World!')
            ' touch move event handler. '
            assert self.modal._window is None
            assert not self.modal._is_open
            return super(ModalButton, self).on_touch_move(touch)

        def on_touch_up(self, touch):
            if False:
                print('Hello World!')
            ' touch up event handler. '
            assert self.modal._window is None
            assert not self.modal._is_open
            return super(ModalButton, self).on_touch_up(touch)

    class TestApp(UnitKivyApp, App):
        """ test app class. """

        def build(self):
            if False:
                return 10
            ' build root layout. '
            root = ModalButton()
            root.modal = ModalView(size_hint=(0.2, 0.5))
            return root
    return TestApp()

@async_run(app_cls_func=modal_app)
async def test_modal_app(kivy_app):
    await kivy_app.wait_clock_frames(2)
    button = kivy_app.root
    modal = button.modal
    modal._anim_duration = 0
    assert modal._window is None
    assert not modal._is_open
    async for _ in kivy_app.do_touch_down_up(widget=button):
        assert modal._window is None
        assert not modal._is_open
    async for _ in kivy_app.do_touch_drag(widget=button, dx=button.width / 4):
        assert modal._window is None
        assert not modal._is_open
    modal.open()
    await kivy_app.wait_clock_frames(2)
    assert modal._window is not None
    assert modal._is_open
    assert isclose(modal.center_x, button.center_x, abs_tol=0.1)
    assert isclose(modal.center_y, button.center_y, abs_tol=0.1)
    async for _ in kivy_app.do_touch_down_up(widget=button):
        pass
    assert modal._window is not None
    assert modal._is_open
    async for _ in kivy_app.do_touch_drag(widget=button, dx=button.width / 4):
        pass
    assert modal._window is not None
    assert modal._is_open
    async for _ in kivy_app.do_touch_drag(pos=(button.center_x + button.width / 4, button.center_y), target_widget=button):
        pass
    assert modal._window is None
    assert not modal._is_open
    modal.open()
    await kivy_app.wait_clock_frames(2)
    assert modal._window is not None
    assert modal._is_open
    async for _ in kivy_app.do_touch_down_up(pos=(button.center_x + button.width / 4, button.center_y)):
        pass
    assert modal._window is None
    assert not modal._is_open