""" pygame.examples.playmus

A simple music player.

   Use pygame.mixer.music to play an audio file.

A window is created to handle keyboard events for playback commands.


Keyboard Controls
-----------------

space - play/pause toggle
r     - rewind
f     - fade out
q     - stop

"""
import sys
import pygame as pg
import pygame.freetype

class Window:
    """The application's Pygame window

    A Window instance manages the creation of and drawing to a
    window. It is a singleton class. Only one instance can exist.

    """
    instance = None

    def __new__(cls, *args, **kwds):
        if False:
            i = 10
            return i + 15
        'Return an open Pygame window'
        if Window.instance is not None:
            return Window.instance
        self = object.__new__(cls)
        pg.display.init()
        self.screen = pg.display.set_mode((600, 400))
        Window.instance = self
        return self

    def __init__(self, title):
        if False:
            print('Hello World!')
        pg.display.set_caption(title)
        self.text_color = (254, 231, 21, 255)
        self.background_color = (16, 24, 32, 255)
        self.screen.fill(self.background_color)
        pg.display.flip()
        pygame.freetype.init()
        self.font = pygame.freetype.Font(None, 20)
        self.font.origin = True
        self.ascender = int(self.font.get_sized_ascender() * 1.5)
        self.descender = int(self.font.get_sized_descender() * 1.5)
        self.line_height = self.ascender - self.descender
        self.write_lines("\nPress 'q' or 'ESCAPE' or close this window to quit\nPress 'SPACE' to play / pause\nPress 'r' to rewind to the beginning (restart)\nPress 'f' to fade music out over 5 seconds\n\nWindow will quit automatically when music ends\n", 0)

    def __enter__(self):
        if False:
            i = 10
            return i + 15
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if False:
            for i in range(10):
                print('nop')
        self.close()
        return False

    def close(self):
        if False:
            while True:
                i = 10
        pg.display.quit()
        Window.instance = None

    def write_lines(self, text, line=0):
        if False:
            return 10
        (w, h) = self.screen.get_size()
        line_height = self.line_height
        nlines = h // line_height
        if line < 0:
            line = nlines + line
        for (i, text_line) in enumerate(text.split('\n'), line):
            y = i * line_height + self.ascender
            self.screen.fill(self.background_color, (0, i * line_height, w, line_height))
            self.font.render_to(self.screen, (15, y), text_line, self.text_color)
        pg.display.flip()

def show_usage_message():
    if False:
        while True:
            i = 10
    print('Usage: python playmus.py <file>')
    print('       python -m pygame.examples.playmus <file>')

def main(file_path):
    if False:
        return 10
    'Play an audio file with pg.mixer.music'
    with Window(file_path) as win:
        win.write_lines('Loading ...', -1)
        pg.mixer.init(frequency=44100)
        try:
            paused = False
            pg.mixer.music.load(file_path)
            pg.time.set_timer(pg.USEREVENT, 500)
            pg.mixer.music.play()
            win.write_lines('Playing ...\n', -1)
            while pg.mixer.music.get_busy() or paused:
                e = pg.event.wait()
                if e.type == pg.KEYDOWN:
                    key = e.key
                    if key == pg.K_SPACE:
                        if paused:
                            pg.mixer.music.unpause()
                            paused = False
                            win.write_lines('Playing ...\n', -1)
                        else:
                            pg.mixer.music.pause()
                            paused = True
                            win.write_lines('Paused ...\n', -1)
                    elif key == pg.K_r:
                        if file_path[-3:].lower() in ('ogg', 'mp3', 'mod'):
                            status = 'Rewound.'
                            pg.mixer.music.rewind()
                        else:
                            status = 'Restarted.'
                            pg.mixer.music.play()
                        if paused:
                            pg.mixer.music.pause()
                            win.write_lines(status, -1)
                    elif key == pg.K_f:
                        win.write_lines('Fading out ...\n', -1)
                        pg.mixer.music.fadeout(5000)
                    elif key in [pg.K_q, pg.K_ESCAPE]:
                        paused = False
                        pg.mixer.music.stop()
                elif e.type == pg.QUIT:
                    paused = False
                    pg.mixer.music.stop()
            pg.time.set_timer(pg.USEREVENT, 0)
        finally:
            pg.mixer.quit()
    pg.quit()
if __name__ == '__main__':
    if len(sys.argv) != 2:
        show_usage_message()
    else:
        main(sys.argv[1])