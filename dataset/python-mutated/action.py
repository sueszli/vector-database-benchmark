"""

"""
from __future__ import annotations
from typing import TYPE_CHECKING, Callable, Sequence, Union
from .enums import ActionResult
from .ui import failed, passed, skipped
if TYPE_CHECKING:
    from typing_extensions import TypeAlias
UIResultFuncType: TypeAlias = Callable[[str, Union[Sequence[str], None]], str]

class ActionReturn:
    """"""
    kind: ActionResult
    ui: UIResultFuncType

    def __init__(self, message: str, details: Sequence[str] | None=None) -> None:
        if False:
            print('Hello World!')
        self.message = message
        self.details = details

    def __str__(self) -> str:
        if False:
            return 10
        return self.__class__.ui(self.message, self.details)

    def __repr__(self) -> str:
        if False:
            while True:
                i = 10
        return f'{self.__class__.__name__}({self.message!r}, details=...)'

class FAILED(ActionReturn):
    """"""
    kind = ActionResult.FAIL
    ui = failed

class PASSED(ActionReturn):
    """"""
    kind = ActionResult.PASS
    ui = passed

class SKIPPED(ActionReturn):
    """"""
    kind = ActionResult.SKIP
    ui = skipped