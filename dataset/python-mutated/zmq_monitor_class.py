"""Module holding utility and convenience functions for zmq event monitoring.

example

import sys
import zmq

# import zmq.asyncio
# from zmq_monitor_class import event_monitor_async

from zmq_monitor_class import event_monitor

context = zmq.Context()
socket = context.socket(zmq.SUB)
socket.setsockopt(zmq.SUBSCRIBE, b"")

event_monitor(socket)
# event_monitor_async(socket, zmq.asyncio.asyncio.get_event_loop() )

socket.connect ("tcp://127.0.0.1:7103")

while True:
    m=socket.recv()
    # print(socket.recv())

"""
import asyncio
import threading
import time
from typing import Any, Dict
import zmq
from zmq.utils.monitor import recv_monitor_message

def event_monitor_thread_async(monitor: zmq.asyncio.Socket, loop: asyncio.BaseEventLoop) -> None:
    if False:
        i = 10
        return i + 15
    "A thread that prints events\n\n    This is a convenience method. It could serve as an example for your code of a monitor,\n    For example if you don't need the prints, then copy paste this part of code to your code and modify it to your needs.\n\n    parameters:\n\n    monitor: a zmq monitor socket, from calling:  my_zmq_socket.get_monitor_socket()\n    loop: an asyncio event loop, from calling zmq.asyncio.asyncio.get_event_loop() , whens starting a thread it does not contains an event loop\n    "
    print('libzmq-%s' % zmq.zmq_version())
    if zmq.zmq_version_info() < (4, 0):
        raise RuntimeError('monitoring in libzmq version < 4.0 is not supported')
    EVENT_MAP = {}
    print('Event names:')
    for name in dir(zmq):
        if name.startswith('EVENT_'):
            value = getattr(zmq, name)
            print('%21s : %4i' % (name, value))
            EVENT_MAP[value] = name
    print('\n')
    asyncio.set_event_loop(loop)

    async def run_loop() -> None:
        while True:
            try:
                while monitor.poll():
                    evt: Dict[str, Any] = {}
                    mon_evt = await recv_monitor_message(monitor)
                    evt.update(mon_evt)
                    evt['description'] = EVENT_MAP[evt['event']]
                    print(f'Event: {evt}')
                    if evt['event'] == zmq.EVENT_MONITOR_STOPPED:
                        break
            except RuntimeError as e:
                print(e)
                time.sleep(1)
        monitor.close()
        print()
        print('event monitor thread done!')
    asyncio.ensure_future(run_loop())

def event_monitor_thread(monitor: zmq.Socket) -> None:
    if False:
        i = 10
        return i + 15
    "A thread that prints events\n\n    This is a convenience method. It could serve as an example for your code of a monitor,\n    For example if you don't need the prints, then copy paste this part of code to your code and modify it to your needs.\n\n    parameters:\n\n    monitor: a zmq monitor socket, from calling:  my_zmq_socket.get_monitor_socket()\n    "
    print('libzmq-%s' % zmq.zmq_version())
    if zmq.zmq_version_info() < (4, 0):
        raise RuntimeError('monitoring in libzmq version < 4.0 is not supported')
    EVENT_MAP = {}
    print('Event names:')
    for name in dir(zmq):
        if name.startswith('EVENT_'):
            value = getattr(zmq, name)
            print('%21s : %4i' % (name, value))
            EVENT_MAP[value] = name
    print()
    print()
    while True:
        try:
            while monitor.poll():
                evt: Dict[str, Any] = {}
                mon_evt = recv_monitor_message(monitor)
                evt.update(mon_evt)
                evt['description'] = EVENT_MAP[evt['event']]
                print(f'Event: {evt}')
                if evt['event'] == zmq.EVENT_MONITOR_STOPPED:
                    break
        except RuntimeError as e:
            print(e)
            time.sleep(1)
    monitor.close()
    print()
    print('event monitor thread done!')

def event_monitor(socket: zmq.Socket) -> None:
    if False:
        while True:
            i = 10
    'Add printing event monitor to a zmq socket, it creates a thread by calling event_monitor_thread\n\n    usage:\n\n    event_monitor_async(socket)\n\n    parameters:\n\n    monitor: a zmq monitor socket, from calling:  my_zmq_socket.get_monitor_socket()\n\n    '
    monitor = socket.get_monitor_socket()
    t = threading.Thread(target=event_monitor_thread, args=(monitor,))
    t.start()

def event_monitor_async(socket: zmq.Socket, loop: asyncio.BaseEventLoop) -> None:
    if False:
        for i in range(10):
            print('nop')
    'Add printing event monitor to a zmq socket, it creates a thread by calling event_monitor_thread_async\n\n    See notes in description of : event_monitor_thread\n\n    usage:\n        loop = zmq.asyncio.asyncio.get_event_loop()\n        event_monitor_async(socket, zmq.asyncio.asyncio.get_event_loop() )\n\n    parameters:\n\n    monitor: a zmq monitor socket, from calling:  my_zmq_socket.get_monitor_socket()\n    '
    monitor = socket.get_monitor_socket()
    t = threading.Thread(target=event_monitor_thread_async, args=(monitor, loop))
    t.start()
__all__ = ['event_monitor', 'event_monitor_async']