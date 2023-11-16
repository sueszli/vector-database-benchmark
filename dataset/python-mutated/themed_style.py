"""
Author: RedFantom
License: GNU GPLv3
Copyright (c) 2017-2018 RedFantom
"""
from ._widget import ThemedWidget
import tkinter as tk
from tkinter import ttk

class ThemedStyle(ttk.Style, ThemedWidget):
    """
    Style that supports setting the theme for a Tk instance. Can be
    used as a drop-in replacement for normal ttk.Style instances.
    Supports the themes provided by this package.
    """

    def __init__(self, *args, **kwargs):
        if False:
            while True:
                i = 10
        '\n        :param theme: Theme to set up initialization completion. If the\n                      theme is not available, fails silently.\n        '
        theme = kwargs.pop('theme', None)
        gif_override = kwargs.pop('gif_override', False)
        ttk.Style.__init__(self, *args, **kwargs)
        ThemedWidget.__init__(self, self.tk, gif_override)
        if theme is not None and theme in self.get_themes():
            self.set_theme(theme)

    def theme_use(self, theme_name=None):
        if False:
            while True:
                i = 10
        '\n        Set a new theme to use or return current theme name\n\n        :param theme_name: name of theme to use\n        :returns: active theme name\n        '
        if theme_name is not None:
            self.set_theme(theme_name)
        return ttk.Style.theme_use(self)

    def theme_names(self):
        if False:
            while True:
                i = 10
        '\n        Alias of get_themes() to allow for a drop-in replacement of the\n        normal ttk.Style instance.\n\n        :returns: Result of get_themes()\n        '
        return self.get_themes()