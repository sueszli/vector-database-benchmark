from functools import cached_property

class Foo:

    @cached_property
    def prop(self) -> int:
        if False:
            i = 10
            return i + 15
        return 1

    @cached_property
    def prop_with_type_comment(self):
        if False:
            i = 10
            return i + 15
        return 1