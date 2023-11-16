from __future__ import division, absolute_import, with_statement, print_function, unicode_literals
from renpy.compat import PY2, basestring, bchr, bord, chr, open, pystr, range, round, str, tobytes, unicode
from typing import Optional
import pygame_sdl2 as pygame
mouse_pos = None
mouse_buttons = [0, 0, 0]

def get_mouse_pos(x, y):
    if False:
        for i in range(10):
            print('nop')
    '\n    Called to get the overridden mouse position.\n    '
    if mouse_pos is None:
        return (x, y)
    return mouse_pos

def post(event_type, **kwargs):
    if False:
        i = 10
        return i + 15
    pygame.event.post(pygame.event.Event(event_type, test=True, **kwargs))

def move_mouse(x, y):
    if False:
        i = 10
        return i + 15
    '\n    Moves the mouse to x, y.\n    '
    global mouse_pos
    pos = (x, y)
    if mouse_pos != pos:
        if mouse_pos:
            rel = (pos[0] - mouse_pos[0], pos[1] - mouse_pos[1])
        else:
            rel = (0, 0)
        post(pygame.MOUSEMOTION, pos=pos, rel=rel, buttons=tuple(mouse_buttons))
    mouse_pos = pos

def press_mouse(button):
    if False:
        for i in range(10):
            print('nop')
    '\n    Presses mouse button `button`.\n    '
    post(pygame.MOUSEBUTTONDOWN, pos=mouse_pos, button=button)
    if button < 3:
        mouse_buttons[button - 1] = 1

def release_mouse(button):
    if False:
        i = 10
        return i + 15
    '\n    Releases mouse button `button`.\n    '
    post(pygame.MOUSEBUTTONUP, pos=mouse_pos, button=button)
    if button < 3:
        mouse_buttons[button - 1] = 0

def click_mouse(button, x, y):
    if False:
        i = 10
        return i + 15
    '\n    Clicks the mouse at x, y\n    '
    move_mouse(x, y)
    press_mouse(button)
    release_mouse(button)

def reset():
    if False:
        while True:
            i = 10
    '\n    Resets mouse handling once the test has ended.\n    '
    global mouse_pos
    mouse_pos = None