from vispy import app, gloo, visuals
import numpy as np

class Canvas(app.Canvas):
    """
    This is an interactive example of how to display text on app.Canvas.
    The text rendering will respond to standard C escape sequences.

    - Scroll for changing text font size
    - Press left and right arrows for changing text anchor positions
    - Press up and down arrows for changing text content

    Standard C escape sequences:
        -   \x07  ->  alert (beep, bell)
        -   \x08  ->  Backspace
        -   \x0c  ->  Formfeed
        -   
  ->  Newline
        -   \r  ->  Carriage Return
        -   	  ->  Horizontal Tab
        -   \x0b  ->  Vertical Tab
        -   \\  ->  Backslash
        -   '  ->  Single quotation mark
        -   "  ->  Double quotation mark
    """

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        app.Canvas.__init__(self, title='Glyphs', keys='interactive')
        self.font_size = 9.0
        l_pos = np.array([[-1.0, 0.0], [1.0, 0.0], [0.0, 0.0], [0.0, 1.0], [0.0, -1.0]])
        self.cross_eye_line = visuals.LineVisual(pos=l_pos, color=(1.0, 0.0, 0.0, 1), method='gl')
        big_test_string = 'This is the big test string!\n'
        big_test_string += 'It includes all of the escape sequences known\n'
        big_test_string += 'to man:\n\n'
        big_test_string += '\t-\t\\n\n'
        big_test_string += '\t-\t\\v\n'
        big_test_string += '\t-\t\\t\n'
        big_test_string += '\t-\tetc..\x0b'
        big_test_string += 'So \x08if \x0cthis \rlooks correct, somebody did a \n'
        big_test_string += 'decent job and deserves a beer '
        big_test_string += 'and a digital salute\x07! ;)'
        big_test_string += '\x0bThe end!'
        self.string_alternatives = ['', 'Hello (scroll/arrows to change text properties)|\x07|how are u', 'Hello (scroll/arrows to change text properties)|\x08|how are u', 'Hello (scroll/arrows to change text properties)|\x0c|how are u', 'Hello (scroll/arrows to change text properties)|\n|how are u', 'Hello (scroll/arrows to change text properties)|\r|how are u', 'Hello (scroll/arrows to change text properties)|\t|how are u', 'Hello (scroll/arrows to change text properties)|\x0b|how are u', 'Hello (scroll/arrows to change text properties)|\\|how are u', "Hello (scroll/arrows to change text properties)|'|how are u", 'Hello (scroll/arrows to change text properties)|"|how are u', 'Hello (scroll/arrows to change text properties)|?|how are u', big_test_string]
        self.str_ind = 0
        self.anchor_variants = [['top', 'left'], ['center', 'left'], ['bottom', 'left'], ['top', 'center'], ['center', 'center'], ['bottom', 'center'], ['top', 'right'], ['center', 'right'], ['bottom', 'right']]
        self.anchor_ind = 0
        self.text = visuals.TextVisual('', bold=True, pos=(0.0, 0.0))
        self.update_text()

    def on_draw(self, event):
        if False:
            for i in range(10):
                print('nop')
        gloo.clear(color='white')
        gloo.set_viewport(0, 0, *self.physical_size)
        self.cross_eye_line.draw()
        self.text.draw()

    def on_mouse_wheel(self, event):
        if False:
            i = 10
            return i + 15
        'Use the mouse wheel to zoom.'
        self.font_size *= 1.25 if event.delta[1] > 0 else 0.8
        self.font_size = max(min(self.font_size, 160.0), 6.0)
        self.update_text()

    def on_resize(self, event):
        if False:
            while True:
                i = 10
        vp = (0, 0, self.physical_size[0], self.physical_size[1])
        self.context.set_viewport(*vp)
        self.text.transforms.configure(canvas=self, viewport=vp)
        self.update_text()

    def on_key_release(self, event):
        if False:
            print('Hello World!')
        if 'Down' in str(event.key):
            if self.str_ind == 0:
                self.str_ind = len(self.string_alternatives) - 1
            else:
                self.str_ind -= 1
        if 'Up' in str(event.key):
            if self.str_ind == len(self.string_alternatives) - 1:
                self.str_ind = 0
            else:
                self.str_ind += 1
        if 'Left' in str(event.key):
            if self.anchor_ind == 0:
                self.anchor_ind = len(self.anchor_variants) - 1
            else:
                self.anchor_ind -= 1
        if 'Right' in str(event.key):
            if self.anchor_ind == len(self.anchor_variants) - 1:
                self.anchor_ind = 0
            else:
                self.anchor_ind += 1
        if event.key == 'b':
            self.text.bold = not self.text.bold
        if event.key == 'i':
            self.text.italic = not self.text.italic
        self.update_text()

    def update_text(self):
        if False:
            i = 10
            return i + 15
        pre_text = '%s pt, ' % round(self.font_size, 1)
        post_text = '\n(anchor_x = ' + self.anchor_variants[self.anchor_ind][1]
        post_text += ', anchor_y = '
        post_text += self.anchor_variants[self.anchor_ind][0] + ')'
        new_txt = pre_text + self.string_alternatives[self.str_ind] + post_text
        self.text.text = new_txt
        anchor_x = self.anchor_variants[self.anchor_ind][1]
        anchor_y = self.anchor_variants[self.anchor_ind][0]
        self.text.anchors = (anchor_x, anchor_y)
        self.text.font_size = self.font_size
        self.text.pos = (self.size[0] // 2, self.size[1] // 2)
        self.update()
if __name__ == '__main__':
    c = Canvas()
    c.show()
    c.app.run()