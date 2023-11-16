from __future__ import annotations
import itertools
from functools import cached_property
from typing import TYPE_CHECKING, Any, Awaitable, Callable, Union
if TYPE_CHECKING:
    from strawberry.field import StrawberryField
    from strawberry.types import Info
SyncExtensionResolver = Callable[..., Any]
AsyncExtensionResolver = Callable[..., Awaitable[Any]]

class FieldExtension:

    def apply(self, field: StrawberryField) -> None:
        if False:
            return 10
        pass

    def resolve(self, next_: SyncExtensionResolver, source: Any, info: Info, **kwargs: Any) -> Any:
        if False:
            i = 10
            return i + 15
        raise NotImplementedError('Sync Resolve is not supported for this Field Extension')

    async def resolve_async(self, next_: AsyncExtensionResolver, source: Any, info: Info, **kwargs: Any) -> Any:
        raise NotImplementedError('Async Resolve is not supported for this Field Extension')

    @cached_property
    def supports_sync(self) -> bool:
        if False:
            i = 10
            return i + 15
        return type(self).resolve is not FieldExtension.resolve

    @cached_property
    def supports_async(self) -> bool:
        if False:
            i = 10
            return i + 15
        return type(self).resolve_async is not FieldExtension.resolve_async

class SyncToAsyncExtension(FieldExtension):
    """Helper class for mixing async extensions with sync resolvers.
    Applied automatically"""

    async def resolve_async(self, next_: AsyncExtensionResolver, source: Any, info: Info, **kwargs: Any) -> Any:
        return next_(source, info, **kwargs)

def _get_sync_resolvers(extensions: list[FieldExtension]) -> list[SyncExtensionResolver]:
    if False:
        return 10
    return [extension.resolve for extension in extensions]

def _get_async_resolvers(extensions: list[FieldExtension]) -> list[AsyncExtensionResolver]:
    if False:
        for i in range(10):
            print('nop')
    return [extension.resolve_async for extension in extensions]

def build_field_extension_resolvers(field: StrawberryField) -> list[Union[SyncExtensionResolver, AsyncExtensionResolver]]:
    if False:
        print('Hello World!')
    '\n    Verifies that all of the field extensions for a given field support\n    sync or async depending on the field resolver.\n    Inserts a SyncToAsyncExtension to be able to\n    use Async extensions on sync resolvers\n    Throws a TypeError otherwise.\n\n    Returns True if resolving should be async, False on sync resolving\n    based on the resolver and extensions\n    '
    if not field.extensions:
        return []
    non_async_extensions = [extension for extension in field.extensions if not extension.supports_async]
    non_async_extension_names = ','.join([extension.__class__.__name__ for extension in non_async_extensions])
    if field.is_async:
        if len(non_async_extensions) > 0:
            raise TypeError(f'Cannot add sync-only extension(s) {non_async_extension_names} to the async resolver of Field {field.name}. Please add a resolve_async method to the extension(s).')
        return _get_async_resolvers(field.extensions)
    else:
        non_sync_extensions = [extension for extension in field.extensions if not extension.supports_sync]
        if len(non_sync_extensions) == 0:
            return _get_sync_resolvers(field.extensions)
        found_sync_extensions = 0
        found_sync_only_extensions = 0
        for extension in field.extensions:
            if extension in non_sync_extensions:
                break
            if extension in non_async_extensions:
                found_sync_only_extensions += 1
            found_sync_extensions += 1
        if len(non_async_extensions) == found_sync_only_extensions:
            return list(itertools.chain(_get_sync_resolvers(field.extensions[:found_sync_extensions]), [SyncToAsyncExtension().resolve_async], _get_async_resolvers(field.extensions[found_sync_extensions:])))
        async_extension_names = ','.join([extension.__class__.__name__ for extension in non_sync_extensions])
        raise TypeError(f'Cannot mix async-only extension(s) {async_extension_names} with sync-only extension(s) {non_async_extension_names} on Field {field.name}. If possible try to change the execution order so that all sync-only extensions are executed first.')