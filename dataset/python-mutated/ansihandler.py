"""Terminal emulation tools"""
import os

class ANSIEscapeCodeHandler(object):
    """ANSI Escape sequences handler"""
    if os.name == 'nt':
        ANSI_COLORS = (('#000000', '#808080'), ('#800000', '#ff0000'), ('#008000', '#00ff00'), ('#808000', '#ffff00'), ('#000080', '#0000ff'), ('#800080', '#ff00ff'), ('#008080', '#00ffff'), ('#c0c0c0', '#ffffff'))
    elif os.name == 'mac':
        ANSI_COLORS = (('#000000', '#818383'), ('#C23621', '#FC391F'), ('#25BC24', '#25BC24'), ('#ADAD27', '#EAEC23'), ('#492EE1', '#5833FF'), ('#D338D3', '#F935F8'), ('#33BBC8', '#14F0F0'), ('#CBCCCD', '#E9EBEB'))
    else:
        ANSI_COLORS = (('#000000', '#7F7F7F'), ('#CD0000', '#ff0000'), ('#00CD00', '#00ff00'), ('#CDCD00', '#ffff00'), ('#0000EE', '#5C5CFF'), ('#CD00CD', '#ff00ff'), ('#00CDCD', '#00ffff'), ('#E5E5E5', '#ffffff'))

    def __init__(self):
        if False:
            i = 10
            return i + 15
        self.intensity = 0
        self.italic = None
        self.bold = None
        self.underline = None
        self.foreground_color = None
        self.background_color = None
        self.default_foreground_color = 30
        self.default_background_color = 47

    def set_code(self, code):
        if False:
            while True:
                i = 10
        assert isinstance(code, int)
        if code == 0:
            self.reset()
        elif code == 1:
            self.intensity = 1
        elif code == 3:
            self.italic = True
        elif code == 4:
            self.underline = True
        elif code == 22:
            self.intensity = 0
            self.bold = False
        elif code == 23:
            self.italic = False
        elif code == 24:
            self.underline = False
        elif code >= 30 and code <= 37:
            self.foreground_color = code
        elif code == 39:
            self.foreground_color = self.default_foreground_color
        elif code >= 40 and code <= 47:
            self.background_color = code
        elif code == 49:
            self.background_color = self.default_background_color
        self.set_style()

    def set_style(self):
        if False:
            while True:
                i = 10
        "\n        Set font style with the following attributes:\n        'foreground_color', 'background_color', 'italic',\n        'bold' and 'underline'\n        "
        raise NotImplementedError

    def reset(self):
        if False:
            for i in range(10):
                print('nop')
        self.current_format = None
        self.intensity = 0
        self.italic = False
        self.bold = False
        self.underline = False
        self.foreground_color = None
        self.background_color = None