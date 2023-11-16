from typing import Sequence, TYPE_CHECKING
from .body import Body, BodyItem, BodyItemParent
from .modelobject import DataDict
if TYPE_CHECKING:
    from .visitor import SuiteVisitor

@Body.register
class Keyword(BodyItem):
    """Base model for a single keyword.

    Extended by :class:`robot.running.model.Keyword` and
    :class:`robot.result.model.Keyword`.
    """
    repr_args = ('name', 'args', 'assign')
    __slots__ = ['name', 'args', 'assign', 'type']

    def __init__(self, name: 'str|None'='', args: Sequence[str]=(), assign: Sequence[str]=(), type: str=BodyItem.KEYWORD, parent: BodyItemParent=None):
        if False:
            return 10
        self.name = name
        self.args = tuple(args)
        self.assign = tuple(assign)
        self.type = type
        self.parent = parent

    @property
    def id(self) -> 'str|None':
        if False:
            while True:
                i = 10
        if not self:
            return None
        return super().id

    def visit(self, visitor: 'SuiteVisitor'):
        if False:
            return 10
        ':mod:`Visitor interface <robot.model.visitor>` entry-point.'
        if self:
            visitor.visit_keyword(self)

    def __bool__(self) -> bool:
        if False:
            i = 10
            return i + 15
        return self.name is not None

    def __str__(self) -> str:
        if False:
            i = 10
            return i + 15
        parts = list(self.assign) + [self.name] + list(self.args)
        return '    '.join((str(p) for p in parts))

    def to_dict(self) -> DataDict:
        if False:
            print('Hello World!')
        data: DataDict = {'name': self.name}
        if self.args:
            data['args'] = self.args
        if self.assign:
            data['assign'] = self.assign
        return data