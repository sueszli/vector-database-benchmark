from __future__ import annotations
import importlib
import sys
from pathlib import Path
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from typing import Any
    from typing_extensions import Literal
    from libqtile.config import Group, Key, Mouse, Rule, Screen
    from libqtile.layout.base import Layout

class ConfigError(Exception):
    pass
config_pyi_header = '\nfrom typing import Any\nfrom typing_extensions import Literal\nfrom libqtile.config import Group, Key, Mouse, Rule, Screen\nfrom libqtile.layout.base import Layout\n\n'

class Config:
    keys: list[Key]
    mouse: list[Mouse]
    groups: list[Group]
    dgroups_key_binder: Any
    dgroups_app_rules: list[Rule]
    follow_mouse_focus: bool
    focus_on_window_activation: Literal['focus', 'smart', 'urgent', 'never']
    cursor_warp: bool
    layouts: list[Layout]
    floating_layout: Layout
    screens: list[Screen]
    auto_fullscreen: bool
    widget_defaults: dict[str, Any]
    extension_defaults: dict[str, Any]
    bring_front_click: bool | Literal['floating_only']
    floats_kept_above: bool
    reconfigure_screens: bool
    wmname: str
    auto_minimize: bool
    wl_input_rules: dict[str, Any] | None

    def __init__(self, file_path=None, **settings):
        if False:
            return 10
        'Create a Config() object from settings\n\n        Only attributes found in Config.__annotations__ will be added to object.\n        config attribute precedence is 1.) **settings 2.) self 3.) default_config\n        '
        self.file_path = file_path
        self.update(**settings)

    def update(self, *, fake_screens=None, **settings):
        if False:
            return 10
        from libqtile.resources import default_config
        if fake_screens:
            self.fake_screens = fake_screens
        default = vars(default_config)
        for key in self.__annotations__.keys():
            try:
                value = settings[key]
            except KeyError:
                value = getattr(self, key, default[key])
            setattr(self, key, value)

    def _reload_config_submodules(self, path: Path) -> None:
        if False:
            while True:
                i = 10
        'Reloads python files from same folder as config file.'
        folder = path.parent
        for module in sys.modules.copy().values():
            if hasattr(module, '__file__') and module.__file__ is not None:
                subpath = Path(module.__file__)
                if subpath == path:
                    continue
                if folder in subpath.parents:
                    importlib.reload(module)

    def load(self):
        if False:
            while True:
                i = 10
        if not self.file_path:
            return
        path = Path(self.file_path)
        name = path.stem
        sys.path.insert(0, path.parent.as_posix())
        if name in sys.modules:
            self._reload_config_submodules(path)
            config = importlib.reload(sys.modules[name])
        else:
            config = importlib.import_module(name)
        self.update(**vars(config))

    def validate(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        '\n        Validate the configuration against the X11 core, if it makes sense.\n        '
        try:
            from libqtile.backend.x11 import core
        except ImportError:
            return
        valid_keys = core.get_keys()
        valid_mods = core.get_modifiers()
        for k in self.keys:
            if k.key.lower() not in valid_keys:
                raise ConfigError('No such key: %s' % k.key)
            for m in k.modifiers:
                if m.lower() not in valid_mods:
                    raise ConfigError('No such modifier: %s' % m)
        for ms in self.mouse:
            for m in ms.modifiers:
                if m.lower() not in valid_mods:
                    raise ConfigError('No such modifier: %s' % m)