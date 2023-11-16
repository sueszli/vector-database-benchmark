from __future__ import annotations
import json
import logging
import re
from pathlib import Path
from ulauncher.config import PATHS
from ulauncher.utils.json_conf import JsonConf
logger = logging.getLogger()
DEFAULT_THEME = 'light'
CSS_RESET = '\n* {\n  color: inherit;\n  font-size: inherit;\n  font-family: inherit;\n  font-style: inherit;\n  font-variant: inherit;\n  font-weight: inherit;\n  text-shadow: inherit;\n  -icon-shadow: inherit;\n  background-color: initial;\n  box-shadow: initial;\n  margin: initial;\n  padding: initial;\n  border-color: initial;\n  border-style: initial;\n  border-width: initial;\n  border-radius: initial;\n  outline-color: initial;\n  outline-style: initial;\n  outline-width: initial;\n  outline-offset: initial;\n  background-clip: initial;\n  background-origin: initial;\n  background-size: initial;\n  background-position: initial;\n  background-repeat: initial;\n  background-image: initial;\n  transition-property: initial;\n  transition-duration: initial;\n  transition-timing-function: initial;\n  transition-delay: initial;\n}\n'

def get_themes():
    if False:
        print('Hello World!')
    '\n    Gets a dict with the theme name as the key and theme as the value\n    '
    themes = {}
    css_theme_paths = [*Path(PATHS.SYSTEM_THEMES).glob('*.css'), *Path(PATHS.USER_THEMES).glob('*.css')]
    all_themes = [Theme(name=p.stem, base_path=str(p.parent)) for p in css_theme_paths]
    for manifest_path in Path(PATHS.USER_THEMES).glob('**/manifest.json'):
        data = json.loads(manifest_path.read_text())
        if data.get('extend_theme', '') is None:
            del data['extend_theme']
        all_themes.append(LegacyTheme(data, base_path=str(manifest_path.parent)))
    for theme in all_themes:
        try:
            theme.validate()
            if themes.get(theme.name):
                logger.warning("Duplicate theme name '%s'", theme.name)
            else:
                themes[theme.name] = theme
        except Exception as e:
            logger.warning("Ignoring invalid or broken theme '%s' in '%s' (%s): %s", theme.name, theme.base_path, type(e).__name__, e)
    return themes

class Theme(JsonConf):
    name = ''
    base_path = ''

    def get_css_path(self):
        if False:
            while True:
                i = 10
        return Path(self.base_path, f'{self.name}.css')

    def get_css(self):
        if False:
            return 10
        css = self.get_css_path().read_text()
        return CSS_RESET + re.sub('(?<=url\\([\\"\\\'])(\\./)?(?!\\/)', f'{self.base_path}/', css)

    def validate(self):
        if False:
            print('Hello World!')
        try:
            assert self.get_css_path().is_file(), f'{self.get_css_path()} is not a file'
        except AssertionError as e:
            raise ThemeError(e) from e

    @classmethod
    def load(cls, theme_name: str):
        if False:
            while True:
                i = 10
        themes = get_themes()
        if theme_name in themes:
            return themes[theme_name]
        logger.warning("Couldn't load theme: '%s'", theme_name)
        if theme_name != DEFAULT_THEME and DEFAULT_THEME in themes:
            return themes[DEFAULT_THEME]
        return next(iter(themes))

class LegacyTheme(Theme):
    css_file = ''
    extend_theme = ''
    matched_text_hl_colors: dict[str, str] = {}

    def get_css_path(self):
        if False:
            for i in range(10):
                print('nop')
        return Path(self.base_path, self.get('css_file_gtk_3.20+', self.css_file))

    def get_css(self):
        if False:
            return 10
        css = self.get_css_path().read_text()
        css = CSS_RESET + re.sub('(?<=url\\([\\"\\\'])(\\./)?(?!\\/)', f'{self.base_path}/', css)
        highlight_color = self.matched_text_hl_colors.get('when_not_selected')
        selected_highlight_color = self.matched_text_hl_colors.get('when_selected')
        if self.extend_theme:
            parent_theme = LegacyTheme.load(self.extend_theme)
            if parent_theme.get_css_path().is_file():
                css = f'{parent_theme.get_css()}\n\n{css}'
            else:
                logger.error('Cannot extend theme "%s". It does not exist', self.extend_theme)
        if highlight_color:
            css += f'.item-highlight {{ color: {highlight_color} }}'
        if selected_highlight_color:
            css += f'.selected.item-box .item-highlight {{ color: {selected_highlight_color} }}'
        return css

    def validate(self):
        if False:
            return 10
        try:
            for prop in ['name', 'css_file']:
                assert self.get(prop), f'"{prop}" is empty'
            assert self.get_css_path().is_file(), f'{self.get_css_path()} is not a file'
        except AssertionError as e:
            raise ThemeError(e) from e

class ThemeError(Exception):
    pass