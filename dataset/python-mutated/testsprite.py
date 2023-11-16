""" pg.examples.testsprite

Like the testsprite.c that comes with libsdl, this pygame version shows
lots of sprites moving around.

It is an abomination of ugly code, and mostly used for testing.


See pg.examples.aliens for some prettyier code.
"""
import sys
import os
from random import randint
from time import time
from typing import List
import pygame as pg
main_dir = os.path.split(os.path.abspath(__file__))[0]
data_dir = os.path.join(main_dir, 'data')
update_rects = True
if '-update_rects' in sys.argv:
    update_rects = True
if '-noupdate_rects' in sys.argv:
    update_rects = False
use_static = False
if '-static' in sys.argv:
    use_static = True
use_layered_dirty = False
if '-layered_dirty' in sys.argv:
    update_rects = True
    use_layered_dirty = True
flags = 0
if '-flip' in sys.argv:
    flags ^= pg.DOUBLEBUF
if '-fullscreen' in sys.argv:
    flags ^= pg.FULLSCREEN
if '-sw' in sys.argv:
    flags ^= pg.SWSURFACE
use_rle = True
if '-hw' in sys.argv:
    flags ^= pg.HWSURFACE
    use_rle = False
if '-scaled' in sys.argv:
    flags ^= pg.SCALED
screen_dims = [640, 480]
if '-height' in sys.argv:
    i = sys.argv.index('-height')
    screen_dims[1] = int(sys.argv[i + 1])
if '-width' in sys.argv:
    i = sys.argv.index('-width')
    screen_dims[0] = int(sys.argv[i + 1])
use_alpha = '-alpha' in sys.argv
print(screen_dims)

class Thingy(pg.sprite.DirtySprite):
    images: List[pg.Surface] = []

    def __init__(self):
        if False:
            i = 10
            return i + 15
        pg.sprite.DirtySprite.__init__(self)
        self.image = Thingy.images[0]
        self.rect = self.image.get_rect()
        self.rect.x = randint(0, screen_dims[0])
        self.rect.y = randint(0, screen_dims[1])
        self.vel = [randint(-1, 1), randint(-1, 1)]
        self.dirty = 2

    def update(self):
        if False:
            return 10
        for i in [0, 1]:
            nv = self.rect[i] + self.vel[i]
            if nv >= screen_dims[i] or nv < 0:
                self.vel[i] = -self.vel[i]
                nv = self.rect[i] + self.vel[i]
            self.rect[i] = nv

class Static(pg.sprite.DirtySprite):
    images: List[pg.Surface] = []

    def __init__(self):
        if False:
            return 10
        pg.sprite.DirtySprite.__init__(self)
        self.image = Static.images[0]
        self.rect = self.image.get_rect()
        self.rect.x = randint(0, 3 * screen_dims[0] // 4)
        self.rect.y = randint(0, 3 * screen_dims[1] // 4)

def main(update_rects=True, use_static=False, use_layered_dirty=False, screen_dims=(640, 480), use_alpha=False, flags=0):
    if False:
        print('Hello World!')
    'Show lots of sprites moving around\n\n    Optional keyword arguments:\n    update_rects - use the RenderUpdate sprite group class (default True)\n    use_static - include non-moving images (default False)\n    use_layered_dirty - Use the FastRenderGroup sprite group (default False)\n    screen_dims - Pygame window dimensions (default [640, 480])\n    use_alpha - use alpha blending (default False)\n    flags - additional display mode flags (default no additional flags)\n\n    '
    if use_layered_dirty:
        update_rects = True
    pg.init()
    pg.display.init()
    screen = pg.display.set_mode(screen_dims, flags, vsync='-vsync' in sys.argv)
    pg.joystick.init()
    num_joysticks = pg.joystick.get_count()
    if num_joysticks > 0:
        stick = pg.joystick.Joystick(0)
        stick.init()
    screen.fill([0, 0, 0])
    pg.display.flip()
    sprite_surface = pg.image.load(os.path.join(data_dir, 'asprite.bmp'))
    sprite_surface2 = pg.image.load(os.path.join(data_dir, 'static.png'))
    if use_rle:
        sprite_surface.set_colorkey([255, 255, 255], pg.SRCCOLORKEY | pg.RLEACCEL)
        sprite_surface2.set_colorkey([255, 255, 255], pg.SRCCOLORKEY | pg.RLEACCEL)
    else:
        sprite_surface.set_colorkey([255, 255, 255], pg.SRCCOLORKEY)
        sprite_surface2.set_colorkey([255, 255, 255], pg.SRCCOLORKEY)
    if use_alpha:
        sprite_surface = sprite_surface.convert_alpha()
        sprite_surface2 = sprite_surface2.convert_alpha()
    else:
        sprite_surface = sprite_surface.convert()
        sprite_surface2 = sprite_surface2.convert()
    Thingy.images = [sprite_surface]
    if use_static:
        Static.images = [sprite_surface2]
    if len(sys.argv) > 1:
        try:
            numsprites = int(sys.argv[-1])
        except Exception:
            numsprites = 100
    else:
        numsprites = 100
    sprites = None
    if use_layered_dirty:
        sprites = pg.sprite.LayeredDirty()
    elif update_rects:
        sprites = pg.sprite.RenderUpdates()
    else:
        sprites = pg.sprite.Group()
    for i in range(0, numsprites):
        if use_static and i % 2 == 0:
            sprites.add(Static())
        sprites.add(Thingy())
    frames = 0
    start = time()
    background = pg.Surface(screen.get_size())
    background = background.convert()
    background.fill([0, 0, 0])
    going = True
    while going:
        if not update_rects:
            screen.fill([0, 0, 0])
        if update_rects:
            sprites.clear(screen, background)
        sprites.update()
        rects = sprites.draw(screen)
        if update_rects:
            pg.display.update(rects)
        else:
            pg.display.flip()
        for event in pg.event.get():
            if event.type in [pg.QUIT, pg.KEYDOWN, pg.QUIT, pg.JOYBUTTONDOWN]:
                going = False
        frames += 1
    end = time()
    print(f'FPS: {frames / (end - start):f}')
    pg.quit()
if __name__ == '__main__':
    main(update_rects, use_static, use_layered_dirty, screen_dims, use_alpha, flags)