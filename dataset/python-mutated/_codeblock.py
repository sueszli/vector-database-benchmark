from typing import Any, List
_CodeType = Any

class CodeBlock:
    """Code fragment for the readable format.
    """

    def __init__(self, head: str, codes: _CodeType) -> None:
        if False:
            for i in range(10):
                print('nop')
        self._head = '' if head == '' else head + ' '
        self._codes = codes

    def _to_str_list(self, indent_width: int=0) -> List[str]:
        if False:
            return 10
        codes: List[str] = []
        codes.append(' ' * indent_width + self._head + '{')
        for code in self._codes:
            next_indent_width = indent_width + 2
            if isinstance(code, str):
                codes.append(' ' * next_indent_width + code)
            elif isinstance(code, CodeBlock):
                codes += code._to_str_list(indent_width=next_indent_width)
            else:
                assert False
        codes.append(' ' * indent_width + '}')
        return codes

    def __str__(self) -> str:
        if False:
            print('Hello World!')
        'Emit CUDA program like the following format.\n\n        <<head>> {\n          <<begin codes>>\n          ...;\n          <<end codes>>\n        }\n        '
        return '\n'.join(self._to_str_list())