"""A switch component for toggling color_mode.

To style components based on color mode, use style props with `color_mode_cond`:

```
rx.text(
    "Hover over me",
    _hover={
        "background": rx.color_mode_cond(
            light="var(--chakra-colors-gray-200)",
            dark="var(--chakra-colors-gray-700)",
        ),
    },
)
```
"""
from __future__ import annotations
from typing import Any
from reflex.components.component import Component
from reflex.components.layout.cond import Cond, cond
from reflex.components.media.icon import Icon
from reflex.style import color_mode, toggle_color_mode
from reflex.vars import BaseVar
from .button import Button
from .switch import Switch
DEFAULT_COLOR_MODE: str = 'light'
DEFAULT_LIGHT_ICON: Icon = Icon.create(tag='sun')
DEFAULT_DARK_ICON: Icon = Icon.create(tag='moon')

def color_mode_cond(light: Any, dark: Any=None) -> BaseVar | Component:
    if False:
        i = 10
        return i + 15
    'Create a component or Prop based on color_mode.\n\n    Args:\n        light: The component or prop to render if color_mode is default\n        dark: The component or prop to render if color_mode is non-default\n\n    Returns:\n        The conditional component or prop.\n    '
    return cond(color_mode == DEFAULT_COLOR_MODE, light, dark)

class ColorModeIcon(Cond):
    """Displays the current color mode as an icon."""

    @classmethod
    def create(cls, light_component: Component | None=None, dark_component: Component | None=None):
        if False:
            i = 10
            return i + 15
        'Create an icon component based on color_mode.\n\n        Args:\n            light_component: the component to display when color mode is default\n            dark_component: the component to display when color mode is dark (non-default)\n\n        Returns:\n            The conditionally rendered component\n        '
        return color_mode_cond(light=light_component or DEFAULT_LIGHT_ICON, dark=dark_component or DEFAULT_DARK_ICON)

class ColorModeSwitch(Switch):
    """Switch for toggling chakra light / dark mode via toggle_color_mode."""

    @classmethod
    def create(cls, *children, **props):
        if False:
            for i in range(10):
                print('nop')
        'Create a switch component bound to color_mode.\n\n        Args:\n            *children: The children of the component.\n            **props: The props to pass to the component.\n\n        Returns:\n            The switch component.\n        '
        return Switch.create(*children, is_checked=color_mode != DEFAULT_COLOR_MODE, on_change=toggle_color_mode, **props)

class ColorModeButton(Button):
    """Button for toggling chakra light / dark mode via toggle_color_mode."""

    @classmethod
    def create(cls, *children, **props):
        if False:
            return 10
        'Create a button component that calls toggle_color_mode on click.\n\n        Args:\n            *children: The children of the component.\n            **props: The props to pass to the component.\n\n        Returns:\n            The switch component.\n        '
        return Button.create(*children, on_click=toggle_color_mode, **props)