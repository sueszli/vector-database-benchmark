from typing import AsyncGenerator, Generator
from unittest.mock import MagicMock
import pytest
from litestar._kwargs.cleanup import DependencyCleanupGroup
from litestar.utils.compat import async_next

@pytest.fixture
def cleanup_mock() -> MagicMock:
    if False:
        for i in range(10):
            print('nop')
    return MagicMock()

@pytest.fixture
def async_cleanup_mock() -> MagicMock:
    if False:
        i = 10
        return i + 15
    return MagicMock()

@pytest.fixture
def generator(cleanup_mock: MagicMock) -> Generator[str, None, None]:
    if False:
        print('Hello World!')

    def func() -> Generator[str, None, None]:
        if False:
            return 10
        yield 'hello'
        cleanup_mock()
    return func()

@pytest.fixture
def async_generator(async_cleanup_mock: MagicMock) -> AsyncGenerator[str, None]:
    if False:
        while True:
            i = 10

    async def func() -> AsyncGenerator[str, None]:
        yield 'world'
        async_cleanup_mock()
    return func()

def test_add(generator: Generator[str, None, None]) -> None:
    if False:
        for i in range(10):
            print('nop')
    group = DependencyCleanupGroup()
    group.add(generator)
    assert group._generators == [generator]

async def test_cleanup(generator: Generator[str, None, None], cleanup_mock: MagicMock) -> None:
    next(generator)
    group = DependencyCleanupGroup([generator])
    await group.cleanup()
    cleanup_mock.assert_called_once()
    assert group._closed

async def test_cleanup_multiple(generator: Generator[str, None, None], async_generator: AsyncGenerator[str, None], cleanup_mock: MagicMock, async_cleanup_mock: MagicMock) -> None:
    next(generator)
    await async_next(async_generator)
    group = DependencyCleanupGroup([generator, async_generator])
    await group.cleanup()
    cleanup_mock.assert_called_once()
    async_cleanup_mock.assert_called_once()
    assert group._closed

async def test_cleanup_on_closed_raises(generator: Generator[str, None, None]) -> None:
    next(generator)
    group = DependencyCleanupGroup([generator])
    await group.cleanup()
    with pytest.raises(RuntimeError):
        await group.cleanup()

async def test_add_on_closed_raises(generator: Generator[str, None, None], async_generator: AsyncGenerator[str, None]) -> None:
    next(generator)
    group = DependencyCleanupGroup([generator])
    await group.cleanup()
    with pytest.raises(RuntimeError):
        group.add(async_generator)