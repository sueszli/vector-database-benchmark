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
import platform
import queue
import cv2
import numpy as np
import pytest
import random
import logging as log
import tempfile
from zmq.error import ZMQError
from vidgear.gears import NetGear, VideoGear
from vidgear.gears.helper import logger_handler
logger = log.getLogger('Test_netgear')
logger.propagate = False
logger.addHandler(logger_handler())
logger.setLevel(log.DEBUG)
_windows = True if os.name == 'nt' else False

def return_testvideo_path():
    if False:
        while True:
            i = 10
    '\n    returns Test Video path\n    '
    path = '{}/Downloads/Test_videos/BigBuckBunny_4sec.mp4'.format(tempfile.gettempdir())
    return os.path.abspath(path)

@pytest.mark.parametrize('address, port', [('172.31.11.15.77', '5555'), (None, '5555')])
def test_playback(address, port):
    if False:
        while True:
            i = 10
    '\n    Tests NetGear Bare-minimum network playback capabilities\n    '
    stream = None
    server = None
    client = None
    try:
        stream = cv2.VideoCapture(return_testvideo_path())
        client = NetGear(address=address, port=port, receive_mode=True)
        server = NetGear(address=address, port=port)
        while True:
            (grabbed, frame_server) = stream.read()
            if not grabbed:
                break
            server.send(frame_server)
            frame_client = client.recv()
    except Exception as e:
        if isinstance(e, (ZMQError, ValueError, RuntimeError)) or address == '172.31.11.15.77':
            logger.exception(str(e))
        else:
            pytest.fail(str(e))
    finally:
        if not stream is None:
            stream.release()
        if not server is None:
            server.close()
        if not client is None:
            client.close()

@pytest.mark.parametrize('receive_mode', [True, False])
def test_primary_mode(receive_mode):
    if False:
        return 10
    '\n    Tests NetGear Bare-minimum network playback capabilities\n    '
    stream = None
    conn = None
    try:
        options_gear = {'THREAD_TIMEOUT': 60}
        stream = VideoGear(source=return_testvideo_path(), **options_gear).start()
        frame = stream.read()
        conn = NetGear(receive_mode=receive_mode)
        if receive_mode:
            conn.send(frame)
        else:
            frame_client = conn.recv()
    except Exception as e:
        if isinstance(e, ValueError):
            pytest.xfail('Test Passed!')
        elif isinstance(e, queue.Empty):
            logger.exception(str(e))
        else:
            pytest.fail(str(e))
    finally:
        if not stream is None:
            stream.stop()
        if not conn is None:
            conn.close()

@pytest.mark.parametrize('pattern', [2, 3])
def test_patterns(pattern):
    if False:
        i = 10
        return i + 15
    '\n    Testing NetGear different messaging patterns\n    '
    options = {'flag': 0, 'copy': False, 'track': False, 'jpeg_compression': False, 'subscriber_timeout': 5}
    frame_server = None
    stream = None
    server = None
    client = None
    try:
        stream = cv2.VideoCapture(return_testvideo_path())
        client = NetGear(pattern=pattern, receive_mode=True, logging=True, **options)
        server = NetGear(pattern=pattern, logging=True, **options)
        i = 0
        random_cutoff = random.randint(10, 100)
        while i < random_cutoff:
            (grabbed, frame_server) = stream.read()
            i += 1
        assert not frame_server is None
        server.send(frame_server)
        frame_client = client.recv(return_data=[1, 2, 3] if pattern == 2 else None)
        assert np.array_equal(frame_server, frame_client)
    except Exception as e:
        if isinstance(e, (ZMQError, ValueError, RuntimeError)):
            logger.exception(str(e))
        else:
            pytest.fail(str(e))
    finally:
        if not stream is None:
            stream.release()
        if not server is None:
            server.close(kill=True)
        if not client is None:
            client.close(kill=True)

@pytest.mark.parametrize('options_server', [{'jpeg_compression': 'invalid', 'jpeg_compression_quality': 5}, {'jpeg_compression': ' gray  ', 'jpeg_compression_quality': 50, 'jpeg_compression_fastdct': True, 'jpeg_compression_fastupsample': True}, {'jpeg_compression': True, 'jpeg_compression_quality': 55.55, 'jpeg_compression_fastdct': True, 'jpeg_compression_fastupsample': True}])
def test_compression(options_server):
    if False:
        print('Hello World!')
    "\n    Testing NetGear's real-time frame compression capabilities\n    "
    stream = None
    server = None
    client = None
    try:
        options_gear = {'THREAD_TIMEOUT': 60}
        colorspace = 'COLOR_BGR2GRAY' if isinstance(options_server['jpeg_compression'], str) and options_server['jpeg_compression'].strip().upper() == 'GRAY' else None
        stream = VideoGear(source=return_testvideo_path(), colorspace=colorspace, **options_gear).start()
        client = NetGear(pattern=0, receive_mode=True, logging=True)
        server = NetGear(pattern=0, logging=True, **options_server)
        while True:
            frame_server = stream.read()
            if frame_server is None:
                break
            server.send(frame_server)
            frame_client = client.recv()
            if isinstance(options_server['jpeg_compression'], str) and options_server['jpeg_compression'].strip().upper() == 'GRAY':
                assert frame_server.ndim == frame_client.ndim, 'Grayscale frame Test Failed!'
    except Exception as e:
        if isinstance(e, (ZMQError, ValueError, RuntimeError, queue.Empty)):
            logger.exception(str(e))
        else:
            pytest.fail(str(e))
    finally:
        if not stream is None:
            stream.stop()
        if not server is None:
            server.close(kill=True)
        if not client is None:
            client.close(kill=True)
test_data_class = [(0, 1, tempfile.gettempdir(), True), (0, 1, ['invalid'], True), (1, 2, os.path.abspath(os.sep) if platform.system() == 'Linux' else 'unknown://invalid.com/', False)]

@pytest.mark.parametrize('pattern, security_mech, custom_cert_location, overwrite_cert', test_data_class)
def test_secure_mode(pattern, security_mech, custom_cert_location, overwrite_cert):
    if False:
        return 10
    "\n    Testing NetGear's Secure Mode\n    "
    options = {'secure_mode': security_mech, 'custom_cert_location': custom_cert_location, 'overwrite_cert': overwrite_cert}
    frame_server = None
    stream = None
    server = None
    client = None
    try:
        stream = cv2.VideoCapture(return_testvideo_path())
        server = NetGear(pattern=pattern, logging=True, **options)
        client = NetGear(pattern=pattern, receive_mode=True, logging=True, **options)
        i = 0
        while i < random.randint(10, 100):
            (grabbed, frame_server) = stream.read()
            i += 1
        assert not frame_server is None
        server.send(frame_server)
        frame_client = client.recv()
        assert np.array_equal(frame_server, frame_client)
    except Exception as e:
        if isinstance(e, (ZMQError, ValueError, RuntimeError, AssertionError)):
            pytest.xfail(str(e))
        else:
            pytest.fail(str(e))
    finally:
        if not stream is None:
            stream.release()
        if not server is None:
            server.close(kill=True)
        if not client is None:
            client.close(kill=True)

@pytest.mark.parametrize('pattern, target_data, options', [(0, [1, 'string', ['list']], {'bidirectional_mode': True, 'jpeg_compression': ['invalid']}), (1, {1: 'apple', 2: 'cat'}, {'bidirectional_mode': True, 'jpeg_compression': False, 'jpeg_compression_quality': 55, 'jpeg_compression_fastdct': False, 'jpeg_compression_fastupsample': False}), (1, (np.random.random(size=(480, 640, 3)) * 255).astype(np.uint8), {'bidirectional_mode': True, 'jpeg_compression': 'GRAY'}), (2, (np.random.random(size=(480, 640, 3)) * 255).astype(np.uint8), {'bidirectional_mode': True, 'jpeg_compression': True})])
def test_bidirectional_mode(pattern, target_data, options):
    if False:
        while True:
            i = 10
    "\n    Testing NetGear's Bidirectional Mode with different data-types\n    "
    stream = None
    server = None
    client = None
    try:
        logger.debug('Given Input Data: {}'.format(target_data if not isinstance(target_data, np.ndarray) else 'IMAGE'))
        options_gear = {'THREAD_TIMEOUT': 60}
        colorspace = 'COLOR_BGR2GRAY' if isinstance(options['jpeg_compression'], str) and options['jpeg_compression'].strip().upper() == 'GRAY' else None
        if colorspace == 'COLOR_BGR2GRAY' and isinstance(target_data, np.ndarray):
            target_data = cv2.cvtColor(target_data, cv2.COLOR_BGR2GRAY)
        stream = VideoGear(source=return_testvideo_path(), colorspace=colorspace, **options_gear).start()
        client = NetGear(pattern=pattern, receive_mode=True, logging=True, **options)
        server = NetGear(pattern=pattern, logging=True, **options)
        if isinstance(target_data, np.ndarray):
            server.send(target_data, message=target_data)
            (server_data, frame_client) = client.recv(return_data=target_data)
            client_data = server.send(target_data)
            assert not client_data is None, 'Test Failed!'
        else:
            frame_server = stream.read()
            assert not frame_server is None
            server.send(frame_server, message=target_data)
            (server_data, frame_client) = client.recv(return_data=target_data)
            client_data = server.send(frame_server, message=target_data)
            if not options['jpeg_compression'] in [True, 'GRAY', ['invalid']]:
                assert np.array_equal(frame_server, frame_client)
            logger.debug('Data received at Server-end: {}'.format(server_data))
            logger.debug('Data received at Client-end: {}'.format(client_data))
            assert client_data == server_data
    except Exception as e:
        if isinstance(e, (ZMQError, ValueError, RuntimeError, queue.Empty)):
            pytest.xfail(str(e))
        else:
            pytest.fail(str(e))
    finally:
        if not stream is None:
            stream.stop()
        if not server is None:
            server.close(kill=True)
        if not client is None:
            client.close(kill=True)

@pytest.mark.parametrize('pattern, options', [(1, {'jpeg_compression': False, 'multiserver_mode': True, 'multiclient_mode': True}), (0, {'jpeg_compression': False, 'multiserver_mode': True, 'multiclient_mode': True}), (1, {'jpeg_compression': False, 'multiserver_mode': True, 'bidirectional_mode': True}), (2, {'multiserver_mode': True, 'ssh_tunnel_mode': 'new@sdf.org', 'subscriber_timeout': 0})])
def test_multiserver_mode(pattern, options):
    if False:
        return 10
    "\n    Testing NetGear's Multi-Server Mode with three unique servers\n    "
    frame_server = None
    stream = None
    server_1 = None
    server_2 = None
    server_3 = None
    client = None
    client_frame_dict = {}
    try:
        stream = cv2.VideoCapture(return_testvideo_path())
        client = NetGear(port=['5556', '5557', '5558'], pattern=pattern, receive_mode=True, logging=True, **options)
        server_1 = NetGear(pattern=pattern, port='5556', logging=True, **options)
        server_2 = NetGear(pattern=pattern, port='5557', logging=True, **options)
        server_3 = NetGear(pattern=pattern, port='5558', logging=True, **options)
        i = 0
        while i < random.randint(10, 100):
            (grabbed, frame_server) = stream.read()
            i += 1
        assert not frame_server is None
        server_1.send(frame_server)
        (unique_address, frame) = client.recv(return_data='data' if 'bidirectional_mode' in options and pattern == 1 else '')
        client_frame_dict[unique_address] = frame
        server_2.send(frame_server)
        (unique_address, frame) = client.recv(return_data='data' if 'bidirectional_mode' in options and pattern == 1 else '')
        client_frame_dict[unique_address] = frame
        server_3.send(frame_server)
        (unique_address, frame) = client.recv(return_data='data' if 'bidirectional_mode' in options and pattern == 1 else '')
        client_frame_dict[unique_address] = frame
        for key in client_frame_dict.keys():
            assert np.array_equal(frame_server, client_frame_dict[key])
    except Exception as e:
        if isinstance(e, (ZMQError, ValueError, RuntimeError)):
            pytest.xfail(str(e))
        else:
            pytest.fail(str(e))
    finally:
        if not stream is None:
            stream.release()
        if not server_1 is None:
            server_1.close()
        if not server_2 is None:
            server_2.close()
        if not server_3 is None:
            server_3.close()
        if not client is None:
            client.close()

@pytest.mark.parametrize('pattern', [0, 1])
def test_multiclient_mode(pattern):
    if False:
        while True:
            i = 10
    "\n    Testing NetGear's Multi-Client Mode with three unique clients\n    "
    options = {'multiclient_mode': True, 'bidirectional_mode': True, 'jpeg_compression': False}
    frame_client = None
    stream = None
    server = None
    client_1 = None
    client_2 = None
    client_3 = None
    try:
        options_gear = {'THREAD_TIMEOUT': 60}
        stream = VideoGear(source=return_testvideo_path(), **options_gear).start()
        server = NetGear(pattern=pattern, port=['5556', '5557', '5558'], logging=True, **options)
        client_1 = NetGear(port='5556', pattern=pattern, receive_mode=True, logging=True, **options)
        client_2 = NetGear(port='5557', pattern=pattern, receive_mode=True, logging=True, **options)
        client_3 = NetGear(port='5558', pattern=pattern, receive_mode=True, logging=True, **options)
        i = 0
        while i < random.randint(10, 100):
            frame_client = stream.read()
            i += 1
        assert not frame_client is None
        server.send(frame_client, message='data' if pattern == 1 else '')
        frame_1 = client_1.recv()
        server.send(frame_client, message='data' if pattern == 1 else '')
        frame_2 = client_2.recv()
        server.send(frame_client, message='data' if pattern == 1 else '')
        frame_3 = client_3.recv()
        assert np.array_equal(frame_1[1] if pattern == 1 else frame_1, frame_client)
        assert np.array_equal(frame_2[1] if pattern == 1 else frame_2, frame_client)
        assert np.array_equal(frame_3[1] if pattern == 1 else frame_3, frame_client)
    except Exception as e:
        if isinstance(e, (ZMQError, ValueError, RuntimeError, queue.Empty)):
            pytest.xfail(str(e))
        else:
            pytest.fail(str(e))
    finally:
        if not stream is None:
            stream.stop()
        if not server is None:
            server.close()
        if not client_1 is None:
            client_1.close()
        if not client_2 is None:
            client_1.close()
        if not client_3 is None:
            client_1.close()

@pytest.mark.parametrize('options', [{'max_retries': -1, 'request_timeout': 2}, {'max_retries': 2, 'request_timeout': 2, 'bidirectional_mode': True, 'ssh_tunnel_mode': '    new@sdf.org  ', 'ssh_tunnel_pwd': 'xyz', 'ssh_tunnel_keyfile': 'ok.txt'}, {'max_retries': 2, 'request_timeout': 4, 'multiclient_mode': True}, {'max_retries': 2, 'request_timeout': -1, 'multiserver_mode': True}, {'subscriber_timeout': 4}])
def test_client_reliablity(options):
    if False:
        return 10
    '\n    Testing validation function of NetGear API\n    '
    client = None
    frame_client = None
    try:
        client = NetGear(pattern=2 if 'subscriber_timeout' in options.keys() else 1, port=[5587] if 'multiserver_mode' in options.keys() else 6657, receive_mode=True, logging=True, **options)
        frame_client = client.recv()
        if frame_client is None:
            raise RuntimeError
    except Exception as e:
        if isinstance(e, RuntimeError):
            pytest.xfail('Reconnection ran successfully.')
        else:
            logger.exception(str(e))
    finally:
        if not client is None:
            client.close()

@pytest.mark.parametrize('options', [{'max_retries': 2, 'request_timeout': 2, 'bidirectional_mode': True}, {'max_retries': 2, 'request_timeout': 2, 'multiserver_mode': True}, {'max_retries': 2, 'request_timeout': 2, 'multiclient_mode': True}, {'ssh_tunnel_mode': 'localhost'}, {'ssh_tunnel_mode': 'localhost:47'}, {'max_retries': 2, 'request_timeout': 2, 'bidirectional_mode': True, 'ssh_tunnel_mode': 'git@github.com'}, {'max_retries': 2, 'request_timeout': 2, 'ssh_tunnel_mode': 'git@github.com'}])
def test_server_reliablity(options):
    if False:
        i = 10
        return i + 15
    '\n    Testing validation function of NetGear API\n    '
    server = None
    stream = None
    frame_client = None
    try:
        server = NetGear(address='127.0.0.1' if 'ssh_tunnel_mode' in options else None, pattern=1, port=[5585] if 'multiclient_mode' in options.keys() else 6654, logging=True, **options)
        stream = cv2.VideoCapture(return_testvideo_path())
        i = 0
        while i < random.randint(10, 100):
            (grabbed, frame_client) = stream.read()
            i += 1
        assert not frame_client is None
        server.send(frame_client)
        server.send(frame_client)
    except Exception as e:
        if isinstance(e, RuntimeError):
            pytest.xfail('Reconnection ran successfully.')
        else:
            logger.exception(str(e))
    finally:
        if not stream is None:
            stream.release()
        if not server is None:
            server.close()

@pytest.mark.parametrize('server_ports, client_ports, options', [(None, 5555, {'multiserver_mode': True}), (5555, None, {'multiclient_mode': True})])
@pytest.mark.xfail(raises=ValueError)
def test_ports(server_ports, client_ports, options):
    if False:
        return 10
    '\n    Test made to fail on wrong port values\n    '
    if server_ports:
        server = NetGear(pattern=1, port=server_ports, logging=True, **options)
        server.close()
    else:
        client = NetGear(port=client_ports, pattern=1, receive_mode=True, logging=True, **options)
        client.close()