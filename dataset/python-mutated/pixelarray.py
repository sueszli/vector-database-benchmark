""" pygame.examples.pixelarray

PixelArray does array processing of pixels.
Sort of like another array processor called 'numpy' - But for pixels.

    Flip it,
            stripe it,
                      rotate it.

Controls
--------

To see different effects - press a key or click a mouse.
"""
import os
import pygame as pg
main_dir = os.path.split(os.path.abspath(__file__))[0]
data_dir = os.path.join(main_dir, 'data')

def show(image):
    if False:
        return 10
    screen = pg.display.get_surface()
    screen.fill((255, 255, 255))
    screen.blit(image, (0, 0))
    pg.display.flip()
    while True:
        event = pg.event.wait()
        if event.type == pg.QUIT:
            pg.quit()
            raise SystemExit
        if event.type in [pg.MOUSEBUTTONDOWN, pg.KEYDOWN]:
            break

def main():
    if False:
        for i in range(10):
            print('nop')
    pg.init()
    pg.display.set_mode((255, 255))
    surface = pg.Surface((255, 255))
    pg.display.flip()
    ar = pg.PixelArray(surface)
    for y in range(255):
        (r, g, b) = (y, y, y)
        ar[:, y] = (r, g, b)
    del ar
    show(surface)
    ar = pg.PixelArray(surface)
    ar[:] = ar[:, ::-1]
    del ar
    show(surface)
    ar = pg.PixelArray(surface)
    ar[::2] = (0, 0, 255)
    del ar
    show(surface)
    ar = pg.PixelArray(surface)
    ar[:, ::2] = (0, 255, 0)
    del ar
    show(surface)
    surface = pg.image.load(os.path.join(data_dir, 'arraydemo.bmp'))
    ar = pg.PixelArray(surface)
    ar[:] = ar[:, ::-1]
    del ar
    show(surface)
    ar = pg.PixelArray(surface)
    ar[:] = ar[::-1, :]
    del ar
    show(surface)
    ar = pg.PixelArray(surface)
    ar[::2] = (255, 255, 255)
    del ar
    show(surface)
    ar = pg.PixelArray(surface)
    ar[:] = ar[::-1, ::-1]
    del ar
    show(surface)
    (w, h) = surface.get_size()
    surface2 = pg.Surface((h, w), surface.get_flags(), surface)
    ar = pg.PixelArray(surface)
    ar2 = pg.PixelArray(surface2)
    ar2[...] = ar.transpose()[::-1, :]
    del ar, ar2
    show(surface2)
    surface = pg.image.load(os.path.join(data_dir, 'arraydemo.bmp'))
    ar = pg.PixelArray(surface)
    sf2 = ar[::2, ::2].make_surface()
    del ar
    show(sf2)
    ar = pg.PixelArray(surface)
    ar.replace((60, 60, 255), (0, 255, 0), 0.06)
    del ar
    show(surface)
    surface = pg.image.load(os.path.join(data_dir, 'arraydemo.bmp'))
    ar = pg.PixelArray(surface)
    ar2 = ar.extract((0, 0, 0), 0.07)
    sf2 = ar2.surface
    del ar, ar2
    show(sf2)
    surface = pg.image.load(os.path.join(data_dir, 'alien1.gif'))
    surface2 = pg.image.load(os.path.join(data_dir, 'alien2.gif'))
    ar1 = pg.PixelArray(surface)
    ar2 = pg.PixelArray(surface2)
    ar3 = ar1.compare(ar2, 0.07)
    sf3 = ar3.surface
    del ar1, ar2, ar3
    show(sf3)
if __name__ == '__main__':
    main()
    pg.quit()