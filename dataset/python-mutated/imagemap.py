from __future__ import division, absolute_import, with_statement, print_function, unicode_literals
from renpy.compat import PY2, basestring, bchr, bord, chr, open, pystr, range, round, str, tobytes, unicode
import hashlib
import pygame_sdl2 as pygame
import renpy
from renpy.display.render import render
cached = set()

class ImageMapCrop(renpy.display.displayable.Displayable):
    """
    This handles the cropping of uncached imagemap components.
    """

    def __init__(self, child, rect):
        if False:
            i = 10
            return i + 15
        super(ImageMapCrop, self).__init__()
        self.child = child
        self.rect = rect

    def visit(self):
        if False:
            for i in range(10):
                print('nop')
        return [self.child]

    def render(self, width, height, st, at):
        if False:
            print('Hello World!')
        cr = render(self.child, width, height, st, at)
        return cr.subsurface(self.rect)

class ImageCacheCrop(renpy.display.displayable.Displayable):
    """
    This handles the cropping of an imagemap component.
    """

    def __init__(self, cache, index):
        if False:
            while True:
                i = 10
        super(ImageCacheCrop, self).__init__()
        self.cache = cache
        self.index = index

    def visit(self):
        if False:
            return 10
        return self.cache.visit(self.index)

    def render(self, width, height, st, at):
        if False:
            for i in range(10):
                print('nop')
        return self.cache.render(self.index, width, height, st, at)

class ImageMapCache(renpy.object.Object):

    def __init__(self, enable):
        if False:
            print('Hello World!')
        self.md5 = hashlib.md5()
        self.imagerect = []
        self.hotspots = {}
        self.areas = []
        self.cache = None
        self.cache_rect = None
        self.cache_width = None
        self.cache_height = None
        enable = False
        self.enable = enable

    def visit(self, index):
        if False:
            i = 10
            return i + 15
        if self.cache is not None:
            return [self.cache]
        else:
            return [self.imagerect[index][0]]

    def crop(self, d, rect):
        if False:
            for i in range(10):
                print('nop')
        if not isinstance(d, renpy.display.im.ImageBase) or not renpy.config.imagemap_cache or (not self.enable):
            return ImageMapCrop(d, rect)
        key = (d, rect)
        rv = self.hotspots.get(key, None)
        if rv is not None:
            return rv
        self.md5.update(repr(d.identity).encode('utf-8'))
        self.md5.update(repr(rect).encode('utf-8'))
        index = len(self.imagerect)
        rv = ImageCacheCrop(self, index)
        self.imagerect.append(key)
        self.hotspots[key] = rv
        self.areas.append((rect[2] + 2, rect[3] + 2, index))
        return rv

    def layout(self):
        if False:
            while True:
                i = 10
        self.areas.sort()
        self.areas.reverse()
        self.cache_rect = [None] * len(self.areas)
        width = self.areas[0][0]
        x = 0
        y = 0
        line_height = 0
        for (w, h, i) in self.areas:
            if x + w > width:
                y += line_height
                line_height = 0
                x = 0
            self.cache_rect[i] = (x + 1, y + 1, w - 2, h - 2)
            x += w
            if line_height < h:
                line_height = h
        self.cache_width = width
        self.cache_height = y + line_height

    def write_cache(self, filename):
        if False:
            i = 10
            return i + 15
        if filename in cached:
            return
        cached.add(filename)
        if renpy.loader.loadable(filename):
            return
        fn = renpy.loader.get_path(filename)
        cache = pygame.Surface((self.cache_width, self.cache_height), pygame.SRCALPHA, 32)
        for (i, (d, rect)) in enumerate(self.imagerect):
            (x, y, _w, _h) = self.cache_rect[i]
            surf = renpy.display.im.cache.get(d).subsurface(rect)
            cache.blit(surf, (x, y))
        pygame.image.save(cache, renpy.exports.fsencode(fn))

    def image_file_hash(self):
        if False:
            print('Hello World!')
        '\n        Returns a hash of the contents of the image files. (As an integer.)\n        '
        rv = 0
        for i in self.imagerect:
            rv += i[0].get_hash()
        return rv & 2147483647

    def finish(self):
        if False:
            print('Hello World!')
        if not self.areas:
            return
        filename = 'im-%s-%x.png' % (self.md5.hexdigest(), self.image_file_hash())
        if renpy.game.preferences.language:
            filename = renpy.game.preferences.language + '-' + filename
        filename = 'cache/' + filename
        self.md5 = None
        self.layout()
        if renpy.config.developer:
            try:
                self.write_cache(filename)
            except Exception:
                pass
        if renpy.loader.loadable(filename):
            self.cache = renpy.display.im.Image(filename)

    def render(self, index, width, height, st, at):
        if False:
            for i in range(10):
                print('nop')
        if self.cache is None:
            (d, rect) = self.imagerect[index]
            return render(d, width, height, st, at).subsurface(rect)
        return render(self.cache, width, height, st, at).subsurface(self.cache_rect[index])