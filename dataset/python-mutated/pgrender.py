from __future__ import division, absolute_import, with_statement, print_function, unicode_literals
from renpy.compat import PY2, basestring, bchr, bord, chr, open, pystr, range, round, str, tobytes, unicode
import sys
import threading
import pygame_sdl2 as pygame
import renpy
sample_alpha = None
sample_noalpha = None

def set_rgba_masks():
    if False:
        for i in range(10):
            print('nop')
    '\n    This rebuilds the sample surfaces, to ones that use the given\n    masks.\n    '
    global sample_alpha
    global sample_noalpha
    s = pygame.Surface((10, 10), 0, 32)
    sample_alpha = s.convert_alpha()
    masks = list(sample_alpha.get_masks())
    masks.sort(key=abs)
    if sys.byteorder == 'big':
        masks = (masks[3], masks[2], masks[1], masks[0])
    else:
        masks = (masks[0], masks[1], masks[2], masks[3])
    sample_alpha = pygame.Surface((10, 10), 0, 32, masks)
    sample_noalpha = pygame.Surface((10, 10), 0, 32, masks[:3] + (0,))
    renpy.audio.audio.sample_surfaces(sample_noalpha, sample_alpha)

class Surface(pygame.Surface):
    """
    This allows us to wrap around pygame's surface, to change
    its mode, as necessary.
    """

    def convert_alpha(self, surface=None):
        if False:
            for i in range(10):
                print('nop')
        return copy_surface_unscaled(self, True)

    def convert(self, surface=None):
        if False:
            return 10
        return copy_surface(self, False)

    def copy(self):
        if False:
            print('Hello World!')
        return copy_surface(self, self)

    def subsurface(self, rect):
        if False:
            print('Hello World!')
        rv = pygame.Surface.subsurface(self, rect)
        return rv

def surface(rect, alpha):
    if False:
        i = 10
        return i + 15
    '\n    Constructs a new surface. The allocated surface is actually a subsurface\n    of a surface that has a 2 pixel border in all directions.\n\n    `alpha` - True if the new surface should have an alpha channel.\n    '
    (width, height) = rect
    if isinstance(alpha, pygame.Surface):
        alpha = alpha.get_masks()[3]
    if alpha:
        sample = sample_alpha
    else:
        sample = sample_noalpha
    if sample is None:
        sample = pygame.Surface((4, 4), pygame.SRCALPHA, 32)
    surf = Surface((width + 4, height + 4), 0, sample)
    return surf.subsurface((2, 2, width, height))
surface_unscaled = surface

def copy_surface(surf, alpha=True):
    if False:
        while True:
            i = 10
    '\n    Creates a copy of the surface.\n    '
    rv = surface_unscaled(surf.get_size(), alpha)
    renpy.display.accelerator.nogil_copy(surf, rv)
    return rv
copy_surface_unscaled = copy_surface
safe_formats = {'png', 'jpg', 'jpeg', 'webp'}
image_load_lock = threading.RLock()
formats = {'png': pygame.image.INIT_PNG, 'jpg': pygame.image.INIT_JPG, 'jpeg': pygame.image.INIT_JPG, 'webp': pygame.image.INIT_WEBP, 'avif': pygame.image.INIT_AVIF, 'tga': 0, 'bmp': 0, 'ico': 0, 'svg': 0}

def load_image(f, filename, size=None):
    if False:
        while True:
            i = 10
    '\n    `f`\n        A file-like object that can be used to load the image.\n    `filename`\n        The name of the file that is being loaded. Used for hinting what\n        kind of image it is.\n    `size`\n        If given, the image is scaled to this size. This only works for\n        SVG images.\n    '
    (_basename, _dot, ext) = filename.rpartition('.')
    try:
        if ext.lower() in safe_formats:
            surf = pygame.image.load(f, renpy.exports.fsencode(filename), size=size)
        else:
            with image_load_lock:
                surf = pygame.image.load(f, renpy.exports.fsencode(filename), size=size)
    except Exception as e:
        extra = ''
        if ext.lower() not in formats:
            extra = " ({} files are not supported by Ren'Py)".format(ext)
        elif formats[ext] and (not pygame.image.has_init(formats[ext])):
            extra = ' (your SDL2_image library does not support {} files)'.format(ext)
        raise Exception('Could not load image {!r}{}: {!r}'.format(filename, extra, e))
    rv = copy_surface_unscaled(surf)
    return rv
load_image_unscaled = load_image

def flip(surf, horizontal, vertical):
    if False:
        return 10
    surf = pygame.transform.flip(surf, horizontal, vertical)
    return copy_surface_unscaled(surf)
flip_unscaled = flip

def rotozoom(surf, angle, zoom):
    if False:
        print('Hello World!')
    surf = pygame.transform.rotozoom(surf, angle, zoom)
    return copy_surface_unscaled(surf)
rotozoom_unscaled = rotozoom

def transform_scale(surf, size):
    if False:
        for i in range(10):
            print('nop')
    surf = pygame.transform.scale(surf, size)
    return copy_surface_unscaled(surf, surf)
transform_scale_unscaled = transform_scale

def transform_rotate(surf, angle):
    if False:
        i = 10
        return i + 15
    surf = pygame.transform.rotate(surf, angle)
    return copy_surface(surf)
transform_rotate_unscaled = transform_rotate