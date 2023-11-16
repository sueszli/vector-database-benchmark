"""
This example shows how to retrieve event information from a callback.
You should see information displayed for any event you triggered.
"""
from vispy import gloo, app

class Canvas(app.Canvas):

    def __init__(self, *args, **kwargs):
        if False:
            i = 10
            return i + 15
        app.Canvas.__init__(self, *args, **kwargs)
        self.title = 'App demo'

    def on_close(self, event):
        if False:
            return 10
        print('closing!')

    def on_resize(self, event):
        if False:
            for i in range(10):
                print('nop')
        print('Resize %r' % (event.size,))

    def on_key_press(self, event):
        if False:
            print('Hello World!')
        modifiers = [key.name for key in event.modifiers]
        print('Key pressed - text: %r, key: %s, modifiers: %r' % (event.text, event.key.name, modifiers))

    def on_key_release(self, event):
        if False:
            for i in range(10):
                print('nop')
        modifiers = [key.name for key in event.modifiers]
        print('Key released - text: %r, key: %s, modifiers: %r' % (event.text, event.key.name, modifiers))

    def on_mouse_press(self, event):
        if False:
            return 10
        self.print_mouse_event(event, 'Mouse press')

    def on_mouse_release(self, event):
        if False:
            print('Hello World!')
        self.print_mouse_event(event, 'Mouse release')

    def on_mouse_move(self, event):
        if False:
            return 10
        self.print_mouse_event(event, 'Mouse move')

    def on_mouse_wheel(self, event):
        if False:
            i = 10
            return i + 15
        self.print_mouse_event(event, 'Mouse wheel')

    def print_mouse_event(self, event, what):
        if False:
            print('Hello World!')
        modifiers = ', '.join([key.name for key in event.modifiers])
        print('%s - pos: %r, button: %s, modifiers: %s, delta: %r' % (what, event.pos, event.button, modifiers, event.delta))

    def on_draw(self, event):
        if False:
            return 10
        gloo.clear(color=True, depth=True)
if __name__ == '__main__':
    canvas = Canvas(keys='interactive')
    canvas.show()
    app.run()