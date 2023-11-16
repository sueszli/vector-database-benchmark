import array
from . import GimpGradientFile, GimpPaletteFile, ImageColor, PaletteFile

class ImagePalette:
    """
    Color palette for palette mapped images

    :param mode: The mode to use for the palette. See:
        :ref:`concept-modes`. Defaults to "RGB"
    :param palette: An optional palette. If given, it must be a bytearray,
        an array or a list of ints between 0-255. The list must consist of
        all channels for one color followed by the next color (e.g. RGBRGBRGB).
        Defaults to an empty palette.
    """

    def __init__(self, mode='RGB', palette=None):
        if False:
            i = 10
            return i + 15
        self.mode = mode
        self.rawmode = None
        self.palette = palette or bytearray()
        self.dirty = None

    @property
    def palette(self):
        if False:
            return 10
        return self._palette

    @palette.setter
    def palette(self, palette):
        if False:
            return 10
        self._colors = None
        self._palette = palette

    @property
    def colors(self):
        if False:
            i = 10
            return i + 15
        if self._colors is None:
            mode_len = len(self.mode)
            self._colors = {}
            for i in range(0, len(self.palette), mode_len):
                color = tuple(self.palette[i:i + mode_len])
                if color in self._colors:
                    continue
                self._colors[color] = i // mode_len
        return self._colors

    @colors.setter
    def colors(self, colors):
        if False:
            print('Hello World!')
        self._colors = colors

    def copy(self):
        if False:
            i = 10
            return i + 15
        new = ImagePalette()
        new.mode = self.mode
        new.rawmode = self.rawmode
        if self.palette is not None:
            new.palette = self.palette[:]
        new.dirty = self.dirty
        return new

    def getdata(self):
        if False:
            print('Hello World!')
        '\n        Get palette contents in format suitable for the low-level\n        ``im.putpalette`` primitive.\n\n        .. warning:: This method is experimental.\n        '
        if self.rawmode:
            return (self.rawmode, self.palette)
        return (self.mode, self.tobytes())

    def tobytes(self):
        if False:
            print('Hello World!')
        'Convert palette to bytes.\n\n        .. warning:: This method is experimental.\n        '
        if self.rawmode:
            msg = 'palette contains raw palette data'
            raise ValueError(msg)
        if isinstance(self.palette, bytes):
            return self.palette
        arr = array.array('B', self.palette)
        return arr.tobytes()
    tostring = tobytes

    def getcolor(self, color, image=None):
        if False:
            print('Hello World!')
        'Given an rgb tuple, allocate palette entry.\n\n        .. warning:: This method is experimental.\n        '
        if self.rawmode:
            msg = 'palette contains raw palette data'
            raise ValueError(msg)
        if isinstance(color, tuple):
            if self.mode == 'RGB':
                if len(color) == 4:
                    if color[3] != 255:
                        msg = 'cannot add non-opaque RGBA color to RGB palette'
                        raise ValueError(msg)
                    color = color[:3]
            elif self.mode == 'RGBA':
                if len(color) == 3:
                    color += (255,)
            try:
                return self.colors[color]
            except KeyError as e:
                if not isinstance(self.palette, bytearray):
                    self._palette = bytearray(self.palette)
                index = len(self.palette) // 3
                special_colors = ()
                if image:
                    special_colors = (image.info.get('background'), image.info.get('transparency'))
                while index in special_colors:
                    index += 1
                if index >= 256:
                    if image:
                        for (i, count) in reversed(list(enumerate(image.histogram()))):
                            if count == 0 and i not in special_colors:
                                index = i
                                break
                    if index >= 256:
                        msg = 'cannot allocate more than 256 colors'
                        raise ValueError(msg) from e
                self.colors[color] = index
                if index * 3 < len(self.palette):
                    self._palette = self.palette[:index * 3] + bytes(color) + self.palette[index * 3 + 3:]
                else:
                    self._palette += bytes(color)
                self.dirty = 1
                return index
        else:
            msg = f'unknown color specifier: {repr(color)}'
            raise ValueError(msg)

    def save(self, fp):
        if False:
            for i in range(10):
                print('nop')
        'Save palette to text file.\n\n        .. warning:: This method is experimental.\n        '
        if self.rawmode:
            msg = 'palette contains raw palette data'
            raise ValueError(msg)
        if isinstance(fp, str):
            fp = open(fp, 'w')
        fp.write('# Palette\n')
        fp.write(f'# Mode: {self.mode}\n')
        for i in range(256):
            fp.write(f'{i}')
            for j in range(i * len(self.mode), (i + 1) * len(self.mode)):
                try:
                    fp.write(f' {self.palette[j]}')
                except IndexError:
                    fp.write(' 0')
            fp.write('\n')
        fp.close()

def raw(rawmode, data):
    if False:
        return 10
    palette = ImagePalette()
    palette.rawmode = rawmode
    palette.palette = data
    palette.dirty = 1
    return palette

def make_linear_lut(black, white):
    if False:
        return 10
    lut = []
    if black == 0:
        for i in range(256):
            lut.append(white * i // 255)
    else:
        msg = 'unavailable when black is non-zero'
        raise NotImplementedError(msg)
    return lut

def make_gamma_lut(exp):
    if False:
        while True:
            i = 10
    lut = []
    for i in range(256):
        lut.append(int((i / 255.0) ** exp * 255.0 + 0.5))
    return lut

def negative(mode='RGB'):
    if False:
        print('Hello World!')
    palette = list(range(256 * len(mode)))
    palette.reverse()
    return ImagePalette(mode, [i // len(mode) for i in palette])

def random(mode='RGB'):
    if False:
        i = 10
        return i + 15
    from random import randint
    palette = []
    for i in range(256 * len(mode)):
        palette.append(randint(0, 255))
    return ImagePalette(mode, palette)

def sepia(white='#fff0c0'):
    if False:
        while True:
            i = 10
    bands = [make_linear_lut(0, band) for band in ImageColor.getrgb(white)]
    return ImagePalette('RGB', [bands[i % 3][i // 3] for i in range(256 * 3)])

def wedge(mode='RGB'):
    if False:
        return 10
    palette = list(range(256 * len(mode)))
    return ImagePalette(mode, [i // len(mode) for i in palette])

def load(filename):
    if False:
        return 10
    with open(filename, 'rb') as fp:
        for paletteHandler in [GimpPaletteFile.GimpPaletteFile, GimpGradientFile.GimpGradientFile, PaletteFile.PaletteFile]:
            try:
                fp.seek(0)
                lut = paletteHandler(fp).getpalette()
                if lut:
                    break
            except (SyntaxError, ValueError):
                pass
        else:
            msg = 'cannot load palette'
            raise OSError(msg)
    return lut