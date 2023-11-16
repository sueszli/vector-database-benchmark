from UM.Scene.SceneNodeDecorator import SceneNodeDecorator
from typing import List, Optional

class GCodeListDecorator(SceneNodeDecorator):

    def __init__(self) -> None:
        if False:
            i = 10
            return i + 15
        super().__init__()
        self._gcode_list = []
        self._filename = None

    def getGcodeFileName(self) -> Optional[str]:
        if False:
            return 10
        return self._filename

    def setGcodeFileName(self, filename: str) -> None:
        if False:
            while True:
                i = 10
        self._filename = filename

    def getGCodeList(self) -> List[str]:
        if False:
            print('Hello World!')
        return self._gcode_list

    def setGCodeList(self, gcode_list: List[str]) -> None:
        if False:
            print('Hello World!')
        self._gcode_list = gcode_list

    def __deepcopy__(self, memo) -> 'GCodeListDecorator':
        if False:
            return 10
        copied_decorator = GCodeListDecorator()
        copied_decorator.setGCodeList(self.getGCodeList())
        return copied_decorator