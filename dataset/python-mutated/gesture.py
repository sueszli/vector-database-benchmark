from __future__ import division, absolute_import, with_statement, print_function, unicode_literals
from renpy.compat import PY2, basestring, bchr, bord, chr, open, pystr, range, round, str, tobytes, unicode
import math
import pygame_sdl2 as pygame
import renpy
DIRECTIONS = ['n', 'ne', 'e', 'se', 's', 'sw', 'w', 'nw']

def dispatch_gesture(gesture):
    if False:
        for i in range(10):
            print('nop')
    '\n    This is called with a gesture to dispatch it as an event.\n    '
    event = renpy.config.gestures.get(gesture, None)
    if event is not None:
        renpy.exports.queue_event(event)
        raise renpy.display.core.IgnoreEvent()

class GestureRecognizer(object):

    def __init__(self):
        if False:
            return 10
        super(GestureRecognizer, self).__init__()
        self.x = None
        self.y = None

    def start(self, x, y):
        if False:
            for i in range(10):
                print('nop')
        self.x = x
        self.y = y
        self.min_component = renpy.config.screen_width * renpy.config.gesture_component_size
        self.min_stroke = renpy.config.screen_width * renpy.config.gesture_stroke_size
        self.current_stroke = None
        self.stroke_length = 0
        self.strokes = []

    def take_point(self, x, y):
        if False:
            i = 10
            return i + 15
        if self.x is None:
            return
        dx = x - self.x
        dy = y - self.y
        length = math.hypot(dx, dy)
        if length < self.min_component:
            return
        self.x = x
        self.y = y
        angle = math.atan2(dx, -dy) * 180 / math.pi + 22.5
        if angle < 0:
            angle += 360
        stroke = DIRECTIONS[int(angle / 45)]
        if stroke == self.current_stroke:
            self.stroke_length += length
        else:
            self.current_stroke = stroke
            self.stroke_length = length
        if self.stroke_length > self.min_stroke:
            if not self.strokes or self.strokes[-1] != stroke:
                self.strokes.append(stroke)

    def finish(self):
        if False:
            i = 10
            return i + 15
        rv = None
        if self.x is None:
            return
        if self.strokes:
            func = renpy.config.dispatch_gesture
            if func is None:
                func = dispatch_gesture
            rv = func('_'.join(self.strokes))
        self.x = None
        self.y = None
        return rv

    def cancel(self):
        if False:
            i = 10
            return i + 15
        self.x = None
        self.y = None

    def event(self, ev, x, y):
        if False:
            i = 10
            return i + 15
        if ev.type == pygame.MOUSEBUTTONDOWN:
            self.start(x, y)
        elif ev.type == pygame.MOUSEMOTION:
            if ev.buttons[0]:
                self.take_point(x, y)
        elif ev.type == pygame.MOUSEBUTTONUP:
            self.take_point(x, y)
            if ev.button == 1:
                return self.finish()
recognizer = GestureRecognizer()