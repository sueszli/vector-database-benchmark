from __future__ import annotations
from collections import OrderedDict, deque
from math import inf
from typing import TYPE_CHECKING, Generic, Tuple
import attr
from outcome import Error, Value
import trio
from ._abc import ReceiveChannel, ReceiveType, SendChannel, SendType, T
from ._core import Abort, RaiseCancelT, Task, enable_ki_protection
from ._util import NoPublicConstructor, final, generic_function
if TYPE_CHECKING:
    from types import TracebackType
    from typing_extensions import Self

def _open_memory_channel(max_buffer_size: int | float) -> tuple[MemorySendChannel[T], MemoryReceiveChannel[T]]:
    if False:
        i = 10
        return i + 15
    "Open a channel for passing objects between tasks within a process.\n\n    Memory channels are lightweight, cheap to allocate, and entirely\n    in-memory. They don't involve any operating-system resources, or any kind\n    of serialization. They just pass Python objects directly between tasks\n    (with a possible stop in an internal buffer along the way).\n\n    Channel objects can be closed by calling `~trio.abc.AsyncResource.aclose`\n    or using ``async with``. They are *not* automatically closed when garbage\n    collected. Closing memory channels isn't mandatory, but it is generally a\n    good idea, because it helps avoid situations where tasks get stuck waiting\n    on a channel when there's no-one on the other side. See\n    :ref:`channel-shutdown` for details.\n\n    Memory channel operations are all atomic with respect to\n    cancellation, either `~trio.abc.ReceiveChannel.receive` will\n    successfully return an object, or it will raise :exc:`Cancelled`\n    while leaving the channel unchanged.\n\n    Args:\n      max_buffer_size (int or math.inf): The maximum number of items that can\n        be buffered in the channel before :meth:`~trio.abc.SendChannel.send`\n        blocks. Choosing a sensible value here is important to ensure that\n        backpressure is communicated promptly and avoid unnecessary latency;\n        see :ref:`channel-buffering` for more details. If in doubt, use 0.\n\n    Returns:\n      A pair ``(send_channel, receive_channel)``. If you have\n      trouble remembering which order these go in, remember: data\n      flows from left → right.\n\n    In addition to the standard channel methods, all memory channel objects\n    provide a ``statistics()`` method, which returns an object with the\n    following fields:\n\n    * ``current_buffer_used``: The number of items currently stored in the\n      channel buffer.\n    * ``max_buffer_size``: The maximum number of items allowed in the buffer,\n      as passed to :func:`open_memory_channel`.\n    * ``open_send_channels``: The number of open\n      :class:`MemorySendChannel` endpoints pointing to this channel.\n      Initially 1, but can be increased by\n      :meth:`MemorySendChannel.clone`.\n    * ``open_receive_channels``: Likewise, but for open\n      :class:`MemoryReceiveChannel` endpoints.\n    * ``tasks_waiting_send``: The number of tasks blocked in ``send`` on this\n      channel (summing over all clones).\n    * ``tasks_waiting_receive``: The number of tasks blocked in ``receive`` on\n      this channel (summing over all clones).\n\n    "
    if max_buffer_size != inf and (not isinstance(max_buffer_size, int)):
        raise TypeError('max_buffer_size must be an integer or math.inf')
    if max_buffer_size < 0:
        raise ValueError('max_buffer_size must be >= 0')
    state: MemoryChannelState[T] = MemoryChannelState(max_buffer_size)
    return (MemorySendChannel[T]._create(state), MemoryReceiveChannel[T]._create(state))
if TYPE_CHECKING:

    class open_memory_channel(Tuple['MemorySendChannel[T]', 'MemoryReceiveChannel[T]']):

        def __new__(cls, max_buffer_size: int | float) -> tuple[MemorySendChannel[T], MemoryReceiveChannel[T]]:
            if False:
                for i in range(10):
                    print('nop')
            return _open_memory_channel(max_buffer_size)

        def __init__(self, max_buffer_size: int | float):
            if False:
                print('Hello World!')
            ...
else:
    open_memory_channel = generic_function(_open_memory_channel)

@attr.s(frozen=True, slots=True)
class MemoryChannelStats:
    current_buffer_used: int = attr.ib()
    max_buffer_size: int | float = attr.ib()
    open_send_channels: int = attr.ib()
    open_receive_channels: int = attr.ib()
    tasks_waiting_send: int = attr.ib()
    tasks_waiting_receive: int = attr.ib()

@attr.s(slots=True)
class MemoryChannelState(Generic[T]):
    max_buffer_size: int | float = attr.ib()
    data: deque[T] = attr.ib(factory=deque)
    open_send_channels: int = attr.ib(default=0)
    open_receive_channels: int = attr.ib(default=0)
    send_tasks: OrderedDict[Task, T] = attr.ib(factory=OrderedDict)
    receive_tasks: OrderedDict[Task, None] = attr.ib(factory=OrderedDict)

    def statistics(self) -> MemoryChannelStats:
        if False:
            return 10
        return MemoryChannelStats(current_buffer_used=len(self.data), max_buffer_size=self.max_buffer_size, open_send_channels=self.open_send_channels, open_receive_channels=self.open_receive_channels, tasks_waiting_send=len(self.send_tasks), tasks_waiting_receive=len(self.receive_tasks))

@final
@attr.s(eq=False, repr=False)
class MemorySendChannel(SendChannel[SendType], metaclass=NoPublicConstructor):
    _state: MemoryChannelState[SendType] = attr.ib()
    _closed: bool = attr.ib(default=False)
    _tasks: set[Task] = attr.ib(factory=set)

    def __attrs_post_init__(self) -> None:
        if False:
            i = 10
            return i + 15
        self._state.open_send_channels += 1

    def __repr__(self) -> str:
        if False:
            for i in range(10):
                print('nop')
        return f'<send channel at {id(self):#x}, using buffer at {id(self._state):#x}>'

    def statistics(self) -> MemoryChannelStats:
        if False:
            return 10
        return self._state.statistics()

    @enable_ki_protection
    def send_nowait(self, value: SendType) -> None:
        if False:
            while True:
                i = 10
        "Like `~trio.abc.SendChannel.send`, but if the channel's buffer is\n        full, raises `WouldBlock` instead of blocking.\n\n        "
        if self._closed:
            raise trio.ClosedResourceError
        if self._state.open_receive_channels == 0:
            raise trio.BrokenResourceError
        if self._state.receive_tasks:
            assert not self._state.data
            (task, _) = self._state.receive_tasks.popitem(last=False)
            task.custom_sleep_data._tasks.remove(task)
            trio.lowlevel.reschedule(task, Value(value))
        elif len(self._state.data) < self._state.max_buffer_size:
            self._state.data.append(value)
        else:
            raise trio.WouldBlock

    @enable_ki_protection
    async def send(self, value: SendType) -> None:
        """See `SendChannel.send <trio.abc.SendChannel.send>`.

        Memory channels allow multiple tasks to call `send` at the same time.

        """
        await trio.lowlevel.checkpoint_if_cancelled()
        try:
            self.send_nowait(value)
        except trio.WouldBlock:
            pass
        else:
            await trio.lowlevel.cancel_shielded_checkpoint()
            return
        task = trio.lowlevel.current_task()
        self._tasks.add(task)
        self._state.send_tasks[task] = value
        task.custom_sleep_data = self

        def abort_fn(_: RaiseCancelT) -> Abort:
            if False:
                i = 10
                return i + 15
            self._tasks.remove(task)
            del self._state.send_tasks[task]
            return trio.lowlevel.Abort.SUCCEEDED
        await trio.lowlevel.wait_task_rescheduled(abort_fn)

    @enable_ki_protection
    def clone(self) -> MemorySendChannel[SendType]:
        if False:
            i = 10
            return i + 15
        "Clone this send channel object.\n\n        This returns a new `MemorySendChannel` object, which acts as a\n        duplicate of the original: sending on the new object does exactly the\n        same thing as sending on the old object. (If you're familiar with\n        `os.dup`, then this is a similar idea.)\n\n        However, closing one of the objects does not close the other, and\n        receivers don't get `EndOfChannel` until *all* clones have been\n        closed.\n\n        This is useful for communication patterns that involve multiple\n        producers all sending objects to the same destination. If you give\n        each producer its own clone of the `MemorySendChannel`, and then make\n        sure to close each `MemorySendChannel` when it's finished, receivers\n        will automatically get notified when all producers are finished. See\n        :ref:`channel-mpmc` for examples.\n\n        Raises:\n          trio.ClosedResourceError: if you already closed this\n              `MemorySendChannel` object.\n\n        "
        if self._closed:
            raise trio.ClosedResourceError
        return MemorySendChannel._create(self._state)

    def __enter__(self) -> Self:
        if False:
            print('Hello World!')
        return self

    def __exit__(self, exc_type: type[BaseException] | None, exc_value: BaseException | None, traceback: TracebackType | None) -> None:
        if False:
            i = 10
            return i + 15
        self.close()

    @enable_ki_protection
    def close(self) -> None:
        if False:
            i = 10
            return i + 15
        'Close this send channel object synchronously.\n\n        All channel objects have an asynchronous `~.AsyncResource.aclose` method.\n        Memory channels can also be closed synchronously. This has the same\n        effect on the channel and other tasks using it, but `close` is not a\n        trio checkpoint. This simplifies cleaning up in cancelled tasks.\n\n        Using ``with send_channel:`` will close the channel object on leaving\n        the with block.\n\n        '
        if self._closed:
            return
        self._closed = True
        for task in self._tasks:
            trio.lowlevel.reschedule(task, Error(trio.ClosedResourceError()))
            del self._state.send_tasks[task]
        self._tasks.clear()
        self._state.open_send_channels -= 1
        if self._state.open_send_channels == 0:
            assert not self._state.send_tasks
            for task in self._state.receive_tasks:
                task.custom_sleep_data._tasks.remove(task)
                trio.lowlevel.reschedule(task, Error(trio.EndOfChannel()))
            self._state.receive_tasks.clear()

    @enable_ki_protection
    async def aclose(self) -> None:
        self.close()
        await trio.lowlevel.checkpoint()

@final
@attr.s(eq=False, repr=False)
class MemoryReceiveChannel(ReceiveChannel[ReceiveType], metaclass=NoPublicConstructor):
    _state: MemoryChannelState[ReceiveType] = attr.ib()
    _closed: bool = attr.ib(default=False)
    _tasks: set[trio._core._run.Task] = attr.ib(factory=set)

    def __attrs_post_init__(self) -> None:
        if False:
            i = 10
            return i + 15
        self._state.open_receive_channels += 1

    def statistics(self) -> MemoryChannelStats:
        if False:
            i = 10
            return i + 15
        return self._state.statistics()

    def __repr__(self) -> str:
        if False:
            print('Hello World!')
        return '<receive channel at {:#x}, using buffer at {:#x}>'.format(id(self), id(self._state))

    @enable_ki_protection
    def receive_nowait(self) -> ReceiveType:
        if False:
            print('Hello World!')
        "Like `~trio.abc.ReceiveChannel.receive`, but if there's nothing\n        ready to receive, raises `WouldBlock` instead of blocking.\n\n        "
        if self._closed:
            raise trio.ClosedResourceError
        if self._state.send_tasks:
            (task, value) = self._state.send_tasks.popitem(last=False)
            task.custom_sleep_data._tasks.remove(task)
            trio.lowlevel.reschedule(task)
            self._state.data.append(value)
        if self._state.data:
            return self._state.data.popleft()
        if not self._state.open_send_channels:
            raise trio.EndOfChannel
        raise trio.WouldBlock

    @enable_ki_protection
    async def receive(self) -> ReceiveType:
        """See `ReceiveChannel.receive <trio.abc.ReceiveChannel.receive>`.

        Memory channels allow multiple tasks to call `receive` at the same
        time. The first task will get the first item sent, the second task
        will get the second item sent, and so on.

        """
        await trio.lowlevel.checkpoint_if_cancelled()
        try:
            value = self.receive_nowait()
        except trio.WouldBlock:
            pass
        else:
            await trio.lowlevel.cancel_shielded_checkpoint()
            return value
        task = trio.lowlevel.current_task()
        self._tasks.add(task)
        self._state.receive_tasks[task] = None
        task.custom_sleep_data = self

        def abort_fn(_: RaiseCancelT) -> Abort:
            if False:
                i = 10
                return i + 15
            self._tasks.remove(task)
            del self._state.receive_tasks[task]
            return trio.lowlevel.Abort.SUCCEEDED
        return await trio.lowlevel.wait_task_rescheduled(abort_fn)

    @enable_ki_protection
    def clone(self) -> MemoryReceiveChannel[ReceiveType]:
        if False:
            i = 10
            return i + 15
        "Clone this receive channel object.\n\n        This returns a new `MemoryReceiveChannel` object, which acts as a\n        duplicate of the original: receiving on the new object does exactly\n        the same thing as receiving on the old object.\n\n        However, closing one of the objects does not close the other, and the\n        underlying channel is not closed until all clones are closed. (If\n        you're familiar with `os.dup`, then this is a similar idea.)\n\n        This is useful for communication patterns that involve multiple\n        consumers all receiving objects from the same underlying channel. See\n        :ref:`channel-mpmc` for examples.\n\n        .. warning:: The clones all share the same underlying channel.\n           Whenever a clone :meth:`receive`\\s a value, it is removed from the\n           channel and the other clones do *not* receive that value. If you\n           want to send multiple copies of the same stream of values to\n           multiple destinations, like :func:`itertools.tee`, then you need to\n           find some other solution; this method does *not* do that.\n\n        Raises:\n          trio.ClosedResourceError: if you already closed this\n              `MemoryReceiveChannel` object.\n\n        "
        if self._closed:
            raise trio.ClosedResourceError
        return MemoryReceiveChannel._create(self._state)

    def __enter__(self) -> Self:
        if False:
            i = 10
            return i + 15
        return self

    def __exit__(self, exc_type: type[BaseException] | None, exc_value: BaseException | None, traceback: TracebackType | None) -> None:
        if False:
            while True:
                i = 10
        self.close()

    @enable_ki_protection
    def close(self) -> None:
        if False:
            print('Hello World!')
        'Close this receive channel object synchronously.\n\n        All channel objects have an asynchronous `~.AsyncResource.aclose` method.\n        Memory channels can also be closed synchronously. This has the same\n        effect on the channel and other tasks using it, but `close` is not a\n        trio checkpoint. This simplifies cleaning up in cancelled tasks.\n\n        Using ``with receive_channel:`` will close the channel object on\n        leaving the with block.\n\n        '
        if self._closed:
            return
        self._closed = True
        for task in self._tasks:
            trio.lowlevel.reschedule(task, Error(trio.ClosedResourceError()))
            del self._state.receive_tasks[task]
        self._tasks.clear()
        self._state.open_receive_channels -= 1
        if self._state.open_receive_channels == 0:
            assert not self._state.receive_tasks
            for task in self._state.send_tasks:
                task.custom_sleep_data._tasks.remove(task)
                trio.lowlevel.reschedule(task, Error(trio.BrokenResourceError()))
            self._state.send_tasks.clear()
            self._state.data.clear()

    @enable_ki_protection
    async def aclose(self) -> None:
        self.close()
        await trio.lowlevel.checkpoint()