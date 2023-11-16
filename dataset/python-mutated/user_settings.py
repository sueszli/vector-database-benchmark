from __future__ import annotations
import os
from ..util.decorators import singleton
tutorial_mode = bool(os.environ.get('SYFT_TUTORIAL_MODE', True))

@singleton
class UserSettings:

    def __init__(self) -> None:
        if False:
            return 10
        pass

    @property
    def helper(self) -> bool:
        if False:
            for i in range(10):
                print('nop')
        return tutorial_mode

    @helper.setter
    def helper(self, value: bool) -> None:
        if False:
            while True:
                i = 10
        global tutorial_mode
        tutorial_mode = value
settings = UserSettings()