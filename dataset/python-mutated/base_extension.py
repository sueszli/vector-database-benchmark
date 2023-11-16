from __future__ import annotations
from enum import Enum
from typing import TYPE_CHECKING, Any, Callable, Dict, Set
from strawberry.utils.await_maybe import AsyncIteratorOrIterator, AwaitableOrValue
if TYPE_CHECKING:
    from graphql import GraphQLResolveInfo
    from strawberry.types import ExecutionContext

class LifecycleStep(Enum):
    OPERATION = 'operation'
    VALIDATION = 'validation'
    PARSE = 'parse'
    RESOLVE = 'resolve'

class SchemaExtension:
    execution_context: ExecutionContext

    def __init__(self, *, execution_context: ExecutionContext):
        if False:
            i = 10
            return i + 15
        self.execution_context = execution_context

    def on_operation(self) -> AsyncIteratorOrIterator[None]:
        if False:
            while True:
                i = 10
        'Called before and after a GraphQL operation (query / mutation) starts'
        yield None

    def on_validate(self) -> AsyncIteratorOrIterator[None]:
        if False:
            print('Hello World!')
        'Called before and after the validation step'
        yield None

    def on_parse(self) -> AsyncIteratorOrIterator[None]:
        if False:
            i = 10
            return i + 15
        'Called before and after the parsing step'
        yield None

    def on_execute(self) -> AsyncIteratorOrIterator[None]:
        if False:
            print('Hello World!')
        'Called before and after the execution step'
        yield None

    def resolve(self, _next: Callable, root: Any, info: GraphQLResolveInfo, *args: str, **kwargs: Any) -> AwaitableOrValue[object]:
        if False:
            print('Hello World!')
        return _next(root, info, *args, **kwargs)

    def get_results(self) -> AwaitableOrValue[Dict[str, Any]]:
        if False:
            for i in range(10):
                print('nop')
        return {}
Hook = Callable[[SchemaExtension], AsyncIteratorOrIterator[None]]
HOOK_METHODS: Set[str] = {SchemaExtension.on_operation.__name__, SchemaExtension.on_validate.__name__, SchemaExtension.on_parse.__name__, SchemaExtension.on_execute.__name__}