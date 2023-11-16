"""
Prints the scan code of all currently pressed keys.
Updates on every keyboard event.
"""
import sys
sys.path.append('..')
import keyboard

def print_pressed_keys(e):
    if False:
        i = 10
        return i + 15
    line = ', '.join((str(code) for code in keyboard._pressed_events))
    print('\r' + line + ' ' * 40, end='')
keyboard.hook(print_pressed_keys)
keyboard.wait()