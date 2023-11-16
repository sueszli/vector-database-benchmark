from __future__ import annotations
import json
from typing import Iterable

class FontEncoder(json.JSONEncoder):

    def default(self, obj):
        if False:
            for i in range(10):
                print('nop')
        if isinstance(obj, Font):
            return {'__gradio_font__': True, 'name': obj.name, 'class': 'google' if isinstance(obj, GoogleFont) else 'font'}
        return json.JSONEncoder.default(self, obj)

def as_font(dct):
    if False:
        for i in range(10):
            print('nop')
    if '__gradio_font__' in dct:
        name = dct['name']
        return GoogleFont(name) if dct['class'] == 'google' else Font(name)
    return dct

class Font:

    def __init__(self, name: str):
        if False:
            i = 10
            return i + 15
        self.name = name

    def __str__(self) -> str:
        if False:
            print('Hello World!')
        return self.name if self.name in ['sans-serif', 'serif', 'monospace', 'cursive', 'fantasy'] else f"'{self.name}'"

    def stylesheet(self) -> str:
        if False:
            print('Hello World!')
        return None

    def __eq__(self, other: Font) -> bool:
        if False:
            i = 10
            return i + 15
        return self.name == other.name and self.stylesheet() == other.stylesheet()

    def __repr__(self) -> str:
        if False:
            while True:
                i = 10
        klass = type(self)
        class_repr = klass.__module__ + '.' + klass.__qualname__
        attrs = ', '.join([k + '=' + repr(v) for (k, v) in self.__dict__.items()])
        return f'<{class_repr} ({attrs})>'

class GoogleFont(Font):

    def __init__(self, name: str, weights: Iterable[int]=(400, 600)):
        if False:
            i = 10
            return i + 15
        self.name = name
        self.weights = weights

    def stylesheet(self) -> str:
        if False:
            while True:
                i = 10
        return f"https://fonts.googleapis.com/css2?family={self.name.replace(' ', '+')}:wght@{';'.join((str(weight) for weight in self.weights))}&display=swap"