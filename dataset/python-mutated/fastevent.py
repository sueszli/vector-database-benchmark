"""
A compatibility shim for pygame.fastevent based on pygame.event.
This module was deprecated in pygame 2.2, and is scheduled for removal in a
future pygame version. If you are using pygame.fastevent, please migrate to
using regular pygame.event module
"""
import pygame.event
import pygame.display
from pygame import error, register_quit
from pygame.event import Event
_ft_init = False

def _ft_init_check():
    if False:
        while True:
            i = 10
    '\n    Raises error if module is not init\n    '
    if not _ft_init:
        raise error('fastevent system not initialized')

def _quit_hook():
    if False:
        print('Hello World!')
    '\n    Hook that gets run to quit module\n    '
    global _ft_init
    _ft_init = False

def init():
    if False:
        i = 10
        return i + 15
    'init() -> None\n    initialize pygame.fastevent\n    '
    global _ft_init
    if not pygame.display.get_init():
        raise error('video system not initialized')
    register_quit(_quit_hook)
    _ft_init = True

def get_init():
    if False:
        while True:
            i = 10
    'get_init() -> bool\n    returns True if the fastevent module is currently initialized\n    '
    return _ft_init

def pump():
    if False:
        while True:
            i = 10
    'pump() -> None\n    internally process pygame event handlers\n    '
    _ft_init_check()
    pygame.event.pump()

def wait():
    if False:
        return 10
    'wait() -> Event\n    wait for an event\n    '
    _ft_init_check()
    return pygame.event.wait()

def poll():
    if False:
        for i in range(10):
            print('nop')
    'poll() -> Event\n    get an available event\n    '
    _ft_init_check()
    return pygame.event.poll()

def get():
    if False:
        for i in range(10):
            print('nop')
    'get() -> list of Events\n    get all events from the queue\n    '
    _ft_init_check()
    return pygame.event.get()

def post(event: Event):
    if False:
        print('Hello World!')
    'post(Event) -> None\n    place an event on the queue\n    '
    _ft_init_check()
    pygame.event.post(event)