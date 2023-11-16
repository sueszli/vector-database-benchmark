from __future__ import annotations

class NBOutput:

    def __init__(self, raw_output: str) -> None:
        if False:
            print('Hello World!')
        self.raw_output = raw_output

    def _repr_html_(self) -> str:
        if False:
            return 10
        return self.raw_output

    def to_html(self) -> NBOutput:
        if False:
            print('Hello World!')
        self.raw_output = self.raw_output.replace('\n', '<br />')
        return self