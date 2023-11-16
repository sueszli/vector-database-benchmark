from __future__ import annotations
from collections.abc import Iterable
from .base import DiscordMessageComponent

class DiscordActionRowError(Exception):

    def __init__(self) -> None:
        if False:
            print('Hello World!')
        super().__init__("A DiscordActionRow cannot be contained within another DiscordActionRow's components")

class DiscordActionRow(DiscordMessageComponent):

    def __init__(self, components: Iterable[DiscordMessageComponent]):
        if False:
            i = 10
            return i + 15
        for component in components:
            if isinstance(component, DiscordActionRow):
                raise DiscordActionRowError()
        self.components = components
        super().__init__(type=1)

    def build(self) -> dict[str, object]:
        if False:
            for i in range(10):
                print('nop')
        return {'type': self.type, 'components': [c.build() for c in self.components]}