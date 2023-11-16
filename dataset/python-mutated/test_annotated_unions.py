from libcst.codemod import CodemodTest
from strawberry.codemods.annotated_unions import ConvertUnionToAnnotatedUnion

class TestConvertConstantCommand(CodemodTest):
    TRANSFORM = ConvertUnionToAnnotatedUnion

    def test_update_union(self) -> None:
        if False:
            while True:
                i = 10
        before = '\n            AUnion = strawberry.union(name="ABC", types=(Foo, Bar))\n        '
        after = '\n            from typing import Annotated, Union\n\n            AUnion = Annotated[Union[Foo, Bar], strawberry.union(name="ABC")]\n        '
        self.assertCodemod(before, after, use_pipe_syntax=False, use_typing_extensions=False)

    def test_update_union_typing_extensions(self) -> None:
        if False:
            while True:
                i = 10
        before = '\n            AUnion = strawberry.union(name="ABC", types=(Foo, Bar))\n        '
        after = '\n            from typing import Annotated, Union\n\n            AUnion = Annotated[Union[Foo, Bar], strawberry.union(name="ABC")]\n        '
        self.assertCodemod(before, after, use_pipe_syntax=False)

    def test_update_union_using_import(self) -> None:
        if False:
            print('Hello World!')
        before = '\n            from strawberry import union\n\n            AUnion = union(name="ABC", types=(Foo, Bar))\n        '
        after = '\n            from typing import Annotated, Union\n\n            AUnion = Annotated[Union[Foo, Bar], strawberry.union(name="ABC")]\n        '
        self.assertCodemod(before, after, use_pipe_syntax=False)

    def test_noop_other_union(self) -> None:
        if False:
            i = 10
            return i + 15
        before = '\n            from potato import union\n\n            union("A", "B")\n        '
        after = '\n            from potato import union\n\n            union("A", "B")\n        '
        self.assertCodemod(before, after, use_pipe_syntax=False)

    def test_update_union_positional_name(self) -> None:
        if False:
            i = 10
            return i + 15
        before = '\n            AUnion = strawberry.union("ABC", types=(Foo, Bar))\n        '
        after = '\n            from typing import Annotated, Union\n\n            AUnion = Annotated[Union[Foo, Bar], strawberry.union(name="ABC")]\n        '
        self.assertCodemod(before, after, use_pipe_syntax=False)

    def test_update_swapped_kwargs(self) -> None:
        if False:
            print('Hello World!')
        before = '\n            AUnion = strawberry.union(types=(Foo, Bar), name="ABC")\n        '
        after = '\n            from typing import Annotated, Union\n\n            AUnion = Annotated[Union[Foo, Bar], strawberry.union(name="ABC")]\n        '
        self.assertCodemod(before, after, use_pipe_syntax=False)

    def test_update_union_list(self) -> None:
        if False:
            i = 10
            return i + 15
        before = '\n            AUnion = strawberry.union(name="ABC", types=[Foo, Bar])\n        '
        after = '\n            from typing import Annotated, Union\n\n            AUnion = Annotated[Union[Foo, Bar], strawberry.union(name="ABC")]\n        '
        self.assertCodemod(before, after, use_pipe_syntax=False)

    def test_update_positional_arguments(self) -> None:
        if False:
            return 10
        before = '\n            AUnion = strawberry.union("ABC", (Foo, Bar))\n        '
        after = '\n            from typing import Annotated, Union\n\n            AUnion = Annotated[Union[Foo, Bar], strawberry.union(name="ABC")]\n        '
        self.assertCodemod(before, after, use_pipe_syntax=False)

    def test_supports_directives_and_description(self) -> None:
        if False:
            i = 10
            return i + 15
        before = '\n            AUnion = strawberry.union(\n                "ABC",\n                (Foo, Bar),\n                description="cool union",\n                directives=[object()],\n            )\n        '
        after = '\n            from typing import Annotated, Union\n\n            AUnion = Annotated[Union[Foo, Bar], strawberry.union(name="ABC", description="cool union", directives=[object()])]\n        '
        self.assertCodemod(before, after, use_pipe_syntax=False)

    def test_noop_with_annotated_unions(self) -> None:
        if False:
            return 10
        before = '\n            AUnion = Annotated[Union[Foo, Bar], strawberry.union(name="ABC")]\n        '
        after = '\n            AUnion = Annotated[Union[Foo, Bar], strawberry.union(name="ABC")]\n        '
        self.assertCodemod(before, after, use_pipe_syntax=False)