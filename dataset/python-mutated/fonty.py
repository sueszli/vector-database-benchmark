""" pygame.examples.fonty

Here we load a .TTF True Type font file, and display it in
a basic pygame window.

Demonstrating several Font object attributes.

- basic window, event, and font management.
"""
import pygame as pg

def main():
    if False:
        print('Hello World!')
    pg.init()
    resolution = (400, 200)
    screen = pg.display.set_mode(resolution)
    fg = (250, 240, 230)
    bg = (5, 5, 5)
    wincolor = (40, 40, 90)
    screen.fill(wincolor)
    font = pg.font.Font(None, 80)
    text = 'Fonty'
    size = font.size(text)
    ren = font.render(text, 0, fg, bg)
    screen.blit(ren, (10, 10))
    font.set_underline(1)
    ren = font.render(text, 0, fg)
    screen.blit(ren, (10, 40 + size[1]))
    font.set_underline(0)
    a_sys_font = pg.font.SysFont('Arial', 60)
    a_sys_font.set_bold(1)
    ren = a_sys_font.render(text, 1, fg, bg)
    screen.blit(ren, (30 + size[0], 10))
    a_sys_font.set_bold(0)
    a_sys_font.set_italic(1)
    ren = a_sys_font.render(text, 1, fg)
    screen.blit(ren, (30 + size[0], 40 + size[1]))
    a_sys_font.set_italic(0)
    print(f"Font metrics for 'Fonty':  {a_sys_font.metrics(text)}")
    ch = 'だ'
    msg = f"Font metrics for '{ch}':  {a_sys_font.metrics(ch)}"
    print(msg)
    pg.display.flip()
    while True:
        if pg.event.wait().type in (pg.QUIT, pg.KEYDOWN, pg.MOUSEBUTTONDOWN):
            break
    pg.quit()
if __name__ == '__main__':
    main()