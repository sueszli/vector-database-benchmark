from __future__ import annotations
from typing import List, Union
from typing_extensions import assert_type

class Foo:

    def asd(self) -> int:
        if False:
            while True:
                i = 10
        return 1

class Bar:

    def asd(self) -> int:
        if False:
            return 10
        return 2
combined = [Foo()] + [Bar()]
assert_type(combined, List[Union[Foo, Bar]])
for item in combined:
    assert_type(item.asd(), int)