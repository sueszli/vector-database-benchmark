"""
SDL2 text provider
==================

Based on SDL2 + SDL2_ttf
"""
__all__ = ('LabelSDL2',)
from kivy.compat import PY2
from kivy.core.text import LabelBase
try:
    from kivy.core.text._text_sdl2 import _SurfaceContainer, _get_extents, _get_fontdescent, _get_fontascent
except ImportError:
    from kivy.core import handle_win_lib_import_error
    handle_win_lib_import_error('text', 'sdl2', 'kivy.core.text._text_sdl2')
    raise

class LabelSDL2(LabelBase):

    def _get_font_id(self):
        if False:
            i = 10
            return i + 15
        return '|'.join([str(self.options[x]) for x in ('font_size', 'font_name_r', 'bold', 'italic', 'underline', 'strikethrough')])

    def get_extents(self, text):
        if False:
            i = 10
            return i + 15
        try:
            if PY2:
                text = text.encode('UTF-8')
        except:
            pass
        return _get_extents(self, text)

    def get_descent(self):
        if False:
            while True:
                i = 10
        return _get_fontdescent(self)

    def get_ascent(self):
        if False:
            return 10
        return _get_fontascent(self)

    def _render_begin(self):
        if False:
            return 10
        self._surface = _SurfaceContainer(self._size[0], self._size[1])

    def _render_text(self, text, x, y):
        if False:
            print('Hello World!')
        self._surface.render(self, text, x, y)

    def _render_end(self):
        if False:
            print('Hello World!')
        return self._surface.get_data()