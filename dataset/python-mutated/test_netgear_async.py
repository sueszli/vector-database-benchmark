"""
===============================================
vidgear library source-code is deployed under the Apache 2.0 License:

Copyright (c) 2019 Abhishek Thakur(@abhiTronix) <abhi.una12@gmail.com>

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
===============================================
"""
import os
import cv2
import sys
import queue
import platform
import numpy as np
import pytest
import asyncio
import functools
import logging as log
import tempfile
from vidgear.gears.asyncio import NetGear_Async
from vidgear.gears.helper import logger_handler
logger = log.getLogger('Test_NetGear_Async')
logger.propagate = False
logger.addHandler(logger_handler())
logger.setLevel(log.DEBUG)

@pytest.fixture(scope='module')
def event_loop():
    if False:
        while True:
            i = 10
    'Create an instance of the default event loop for each test case.'
    loop = asyncio.SelectorEventLoop()
    yield loop
    loop.close()

def return_testvideo_path():
    if False:
        i = 10
        return i + 15
    '\n    returns Test Video path\n    '
    path = '{}/Downloads/Test_videos/BigBuckBunny_4sec.mp4'.format(tempfile.gettempdir())
    return os.path.abspath(path)

async def custom_frame_generator():
    stream = cv2.VideoCapture(return_testvideo_path())
    while True:
        (grabbed, frame) = stream.read()
        if not grabbed:
            break
        yield frame
        await asyncio.sleep(0)
    stream.release()

class Custom_Generator:
    """
    Custom Generator using OpenCV, for testing bidirectional mode.
    """

    def __init__(self, server=None, data=''):
        if False:
            print('Hello World!')
        assert not server is None, 'Invalid Value'
        self.server = server
        self.data = data

    async def custom_dataframe_generator(self):
        stream = cv2.VideoCapture(return_testvideo_path())
        while True:
            (grabbed, frame) = stream.read()
            if not grabbed:
                break
            recv_data = await self.server.transceive_data()
            if not recv_data is None:
                if isinstance(recv_data, np.ndarray):
                    assert not (recv_data is None or np.shape(recv_data) == ()), 'Failed Test'
                else:
                    logger.debug(recv_data)
            yield (self.data, frame)
            await asyncio.sleep(0)
        stream.release()

async def client_iterator(client, data=False):
    async for frame in client.recv_generator():
        assert not (frame is None or np.shape(frame) == ()), 'Failed Test'
        if data:
            await client.transceive_data(data='invalid')
        await asyncio.sleep(0)

async def client_dataframe_iterator(client, data=''):
    async for (recvd_data, frame) in client.recv_generator():
        if not recvd_data is None:
            logger.debug(recvd_data)
        assert not (frame is None or np.shape(frame) == ()), 'Failed Test'
        await client.transceive_data(data=data)
        await asyncio.sleep(0)

@pytest.mark.asyncio
@pytest.mark.parametrize('pattern', [1, 2, 3, 4])
async def test_netgear_async_playback(pattern):
    try:
        client = NetGear_Async(logging=True, pattern=pattern, receive_mode=True, timeout=7.0).launch()
        options_gear = {'THREAD_TIMEOUT': 60}
        server = NetGear_Async(source=return_testvideo_path(), pattern=pattern, timeout=7.0 if pattern == 4 else 0, logging=True, **options_gear).launch()
        input_coroutines = [server.task, client_iterator(client, data=True if pattern == 4 else False)]
        res = await asyncio.gather(*input_coroutines, return_exceptions=True)
    except Exception as e:
        if isinstance(e, queue.Empty):
            pytest.fail(str(e))
    finally:
        server.close(skip_loop=True)
        client.close(skip_loop=True)
test_data_class = [(None, False), (custom_frame_generator(), True), ([], False)]

@pytest.mark.asyncio
@pytest.mark.parametrize('generator, result', test_data_class)
async def test_netgear_async_custom_server_generator(generator, result):
    try:
        server = NetGear_Async(protocol='udp', timeout=5.0, logging=True)
        server.config['generator'] = generator
        server.launch()
        client = NetGear_Async(logging=True, receive_mode=True, timeout=5.0).launch()
        input_coroutines = [server.task, client_iterator(client)]
        res = await asyncio.gather(*input_coroutines, return_exceptions=True)
    except Exception as e:
        if result:
            pytest.fail(str(e))
        else:
            pytest.xfail(str(e))
    finally:
        server.close(skip_loop=True)
        client.close(skip_loop=True)
test_data_class = [(custom_frame_generator(), 'Hi', {'bidirectional_mode': True}, {'bidirectional_mode': True}, False), ([], 444404444, {'bidirectional_mode': True}, {'bidirectional_mode': False}, False), ([], [1, 'string', ['list']], {'bidirectional_mode': True}, {'bidirectional_mode': True}, True), ([], (np.random.random(size=(480, 640, 3)) * 255).astype(np.uint8), {'bidirectional_mode': True}, {'bidirectional_mode': True}, True)]

@pytest.mark.asyncio
@pytest.mark.parametrize('generator, data, options_server, options_client, result', test_data_class)
async def test_netgear_async_bidirectionalmode(generator, data, options_server, options_client, result):
    try:
        server = NetGear_Async(logging=True, timeout=5.0, **options_server)
        if not generator:
            cg = Custom_Generator(server, data=data)
            generator = cg.custom_dataframe_generator()
        server.config['generator'] = generator
        server.launch()
        client = NetGear_Async(logging=True, receive_mode=True, timeout=5.0, **options_client).launch()
        input_coroutines = [server.task, client_dataframe_iterator(client, data=data)]
        res = await asyncio.gather(*input_coroutines, return_exceptions=True)
    except Exception as e:
        if result:
            pytest.fail(str(e))
        else:
            pytest.xfail(str(e))
    finally:
        server.close(skip_loop=True)
        client.close(skip_loop=True)

@pytest.mark.asyncio
@pytest.mark.parametrize('address, port', [('172.31.11.15.77', '5555'), ('172.31.11.33.44', '5555'), (None, '5555')])
async def test_netgear_async_addresses(address, port):
    server = None
    try:
        client = NetGear_Async(address=address, port=port, logging=True, timeout=5.0, receive_mode=True).launch()
        options_gear = {'THREAD_TIMEOUT': 60}
        if address is None:
            server = NetGear_Async(source=return_testvideo_path(), address=address, port=port, timeout=5.0, logging=True, **options_gear).launch()
            input_coroutines = [server.task, client_iterator(client)]
            await asyncio.gather(*input_coroutines, return_exceptions=True)
        elif address == '172.31.11.33.44':
            options_gear['bidirectional_mode'] = True
            server = NetGear_Async(source=return_testvideo_path(), address=address, port=port, logging=True, timeout=5.0, **options_gear).launch()
            await asyncio.ensure_future(server.task)
        else:
            await asyncio.ensure_future(client_iterator(client))
    except Exception as e:
        if address in ['172.31.11.15.77', '172.31.11.33.44'] or isinstance(e, queue.Empty):
            logger.exception(str(e))
        else:
            pytest.fail(str(e))
    finally:
        if (address is None or address == '172.31.11.33.44') and (not server is None):
            server.close(skip_loop=True)
        client.close(skip_loop=True)

@pytest.mark.asyncio
async def test_netgear_async_recv_generator():
    server = None
    try:
        server = NetGear_Async(source=return_testvideo_path(), timeout=5.0, logging=True)
        async for frame in server.recv_generator():
            logger.warning('Failed')
    except Exception as e:
        if isinstance(e, (ValueError, asyncio.TimeoutError)):
            pytest.xfail(str(e))
        else:
            pytest.fail(str(e))
    finally:
        if not server is None:
            server.close(skip_loop=True)

@pytest.mark.asyncio
@pytest.mark.parametrize('pattern, options', [(0, {'bidirectional_mode': True}), (0, {'bidirectional_mode': False}), (1, {'bidirectional_mode': 'invalid'}), (2, {'bidirectional_mode': True})])
async def test_netgear_async_options(pattern, options):
    client = None
    try:
        client = NetGear_Async(source=None if options['bidirectional_mode'] != True else return_testvideo_path(), receive_mode=True, timeout=5.0, pattern=pattern, logging=True, **options)
        async for frame in client.recv_generator():
            if not options['bidirectional_mode']:
                target_data = 'Client here.'
                await client.transceive_data(data=target_data)
            logger.warning('Failed')
    except Exception as e:
        if isinstance(e, (ValueError, asyncio.TimeoutError)):
            pytest.xfail(str(e))
        else:
            pytest.fail(str(e))
    finally:
        if not client is None:
            client.close(skip_loop=True)