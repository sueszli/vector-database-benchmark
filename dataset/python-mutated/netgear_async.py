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
import cv2
import sys
import numpy as np
import asyncio
import inspect
import logging as log
import string
import secrets
import platform
from collections import deque
from ..helper import logger_handler, import_dependency_safe, logcurr_vidgear_ver
from ..videogear import VideoGear
zmq = import_dependency_safe('zmq', pkg_name='pyzmq', error='silent', min_version='4.0')
if not zmq is None:
    import zmq.asyncio
msgpack = import_dependency_safe('msgpack', error='silent')
m = import_dependency_safe('msgpack_numpy', error='silent')
uvloop = import_dependency_safe('uvloop', error='silent')
logger = log.getLogger('NetGear_Async')
logger.propagate = False
logger.addHandler(logger_handler())
logger.setLevel(log.DEBUG)

class NetGear_Async:
    """
    NetGear_Async can generate the same performance as NetGear API at about one-third the memory consumption, and also provide complete server-client handling with various
    options to use variable protocols/patterns similar to NetGear, but lacks in term of flexibility as it supports only a few NetGear's Exclusive Modes.

    NetGear_Async is built on `zmq.asyncio`, and powered by a high-performance asyncio event loop called uvloop to achieve unwatchable high-speed and lag-free video streaming
    over the network with minimal resource constraints. NetGear_Async can transfer thousands of frames in just a few seconds without causing any significant load on your
    system.

    NetGear_Async provides complete server-client handling and options to use variable protocols/patterns similar to NetGear API. Furthermore, NetGear_Async allows us to define
     our custom Server as source to transform frames easily before sending them across the network.

    NetGear_Async now supports additional **bidirectional data transmission** between receiver(client) and sender(server) while transferring frames.
    Users can easily build complex applications such as like _Real-Time Video Chat_ in just few lines of code.

    In addition to all this, NetGear_Async API also provides internal wrapper around VideoGear, which itself provides internal access to both CamGear and PiGear APIs, thereby
    granting it exclusive power for transferring frames incoming from any source to the network.

    NetGear_Async as of now supports four ZeroMQ messaging patterns:

    - `zmq.PAIR` _(ZMQ Pair Pattern)_
    - `zmq.REQ/zmq.REP` _(ZMQ Request/Reply Pattern)_
    - `zmq.PUB/zmq.SUB` _(ZMQ Publish/Subscribe Pattern)_
    - `zmq.PUSH/zmq.PULL` _(ZMQ Push/Pull Pattern)_

    Whereas supported protocol are: `tcp` and `ipc`.
    """

    def __init__(self, address=None, port=None, protocol='tcp', pattern=0, receive_mode=False, timeout=0.0, enablePiCamera=False, stabilize=False, source=None, camera_num=0, stream_mode=False, backend=0, colorspace=None, resolution=(640, 480), framerate=25, time_delay=0, logging=False, **options):
        if False:
            print('Hello World!')
        "\n        This constructor method initializes the object state and attributes of the NetGear_Async class.\n\n        Parameters:\n            address (str): sets the valid network address of the Server/Client.\n            port (str): sets the valid Network Port of the Server/Client.\n            protocol (str): sets the valid messaging protocol between Server/Client.\n            pattern (int): sets the supported messaging pattern(flow of communication) between Server/Client\n            receive_mode (bool): select the NetGear_Async's Mode of operation.\n            timeout (int/float): controls the maximum waiting time(in sec) after which Client throws `TimeoutError`.\n            enablePiCamera (bool): provide access to PiGear(if True) or CamGear(if False) APIs respectively.\n            stabilize (bool): enable access to Stabilizer Class for stabilizing frames.\n            camera_num (int): selects the camera module index which will be used as Rpi source.\n            resolution (tuple): sets the resolution (i.e. `(width,height)`) of the Rpi source.\n            framerate (int/float): sets the framerate of the Rpi source.\n            source (based on input): defines the source for the input stream.\n            stream_mode (bool): controls the exclusive YouTube Mode.\n            backend (int): selects the backend for OpenCV's VideoCapture class.\n            colorspace (str): selects the colorspace of the input stream.\n            logging (bool): enables/disables logging.\n            time_delay (int): time delay (in sec) before start reading the frames.\n            options (dict): provides ability to alter Tweak Parameters of NetGear_Async, CamGear, PiGear & Stabilizer.\n        "
        logcurr_vidgear_ver(logging=logging)
        import_dependency_safe('zmq' if zmq is None else '', min_version='4.0', pkg_name='pyzmq')
        import_dependency_safe('msgpack' if msgpack is None else '')
        import_dependency_safe('msgpack_numpy' if m is None else '')
        self.__logging = logging
        valid_messaging_patterns = {0: (zmq.PAIR, zmq.PAIR), 1: (zmq.REQ, zmq.REP), 2: (zmq.PUB, zmq.SUB), 3: (zmq.PUSH, zmq.PULL)}
        if isinstance(pattern, int) and pattern in valid_messaging_patterns:
            self.__msg_pattern = pattern
            self.__pattern = valid_messaging_patterns[pattern]
        else:
            self.__msg_pattern = 0
            self.__pattern = valid_messaging_patterns[self.__msg_pattern]
            if self.__logging:
                logger.warning('Invalid pattern {pattern}. Defaulting to `zmq.PAIR`!'.format(pattern=pattern))
        if isinstance(protocol, str) and protocol in ['tcp', 'ipc']:
            self.__protocol = protocol
        else:
            self.__protocol = 'tcp'
            if self.__logging:
                logger.warning('Invalid protocol. Defaulting to `tcp`!')
        self.__terminate = False
        self.__receive_mode = receive_mode
        self.__stream = None
        self.__msg_socket = None
        self.config = {}
        self.__queue = None
        self.__bi_mode = False
        if timeout and isinstance(timeout, (int, float)):
            self.__timeout = float(timeout)
        else:
            self.__timeout = 15.0
        self.__id = ''.join((secrets.choice(string.ascii_uppercase + string.digits) for i in range(8)))
        options = {str(k).strip(): v for (k, v) in options.items()}
        if 'bidirectional_mode' in options:
            value = options['bidirectional_mode']
            if isinstance(value, bool) and pattern < 2 and (source is None):
                self.__bi_mode = value
            else:
                self.__bi_mode = False
                logger.warning('Bidirectional data transmission is disabled!')
            if pattern >= 2:
                raise ValueError('[NetGear_Async:ERROR] :: `{}` pattern is not valid when Bidirectional Mode is enabled. Kindly refer Docs for more Information!'.format(pattern))
            elif not source is None:
                raise ValueError('[NetGear_Async:ERROR] :: Custom source must be used when Bidirectional Mode is enabled. Kindly refer Docs for more Information!'.format(pattern))
            elif isinstance(value, bool) and self.__logging:
                logger.debug('Bidirectional Data Transmission is {} for this connection!'.format('enabled' if value else 'disabled'))
            else:
                logger.error('`bidirectional_mode` value is invalid!')
            del options['bidirectional_mode']
        self.__msg_context = zmq.asyncio.Context()
        if receive_mode:
            if address is None:
                self.__address = '*'
            else:
                self.__address = address
            if port is None:
                self.__port = '5555'
            else:
                self.__port = port
        else:
            if source is None:
                self.config = {'generator': None}
                if self.__logging:
                    logger.warning('Given source is of NoneType!')
            else:
                self.__stream = VideoGear(enablePiCamera=enablePiCamera, stabilize=stabilize, source=source, camera_num=camera_num, stream_mode=stream_mode, backend=backend, colorspace=colorspace, resolution=resolution, framerate=framerate, logging=logging, time_delay=time_delay, **options)
                self.config = {'generator': self.__frame_generator()}
            if address is None:
                self.__address = 'localhost'
            else:
                self.__address = address
            if port is None:
                self.__port = '5555'
            else:
                self.__port = port
            self.task = None
        if platform.system() == 'Windows':
            if sys.version_info[:2] >= (3, 8):
                asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
        elif not uvloop is None:
            asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
        else:
            import_dependency_safe('uvloop', error='log')
        self.loop = asyncio.get_event_loop()
        self.__queue = asyncio.Queue() if self.__bi_mode else None
        if self.__logging:
            logger.info('Using `{}` event loop for this process.'.format(self.loop.__class__.__name__))

    def launch(self):
        if False:
            i = 10
            return i + 15
        '\n        Launches an asynchronous generators and loop executors for respective task.\n        '
        if self.__receive_mode:
            if self.__logging:
                logger.debug('Launching NetGear_Async asynchronous generator!')
            self.loop.run_in_executor(None, self.recv_generator)
        else:
            if self.__logging:
                logger.debug('Creating NetGear_Async asynchronous server handler!')
            self.task = asyncio.ensure_future(self.__server_handler())
        return self

    async def __server_handler(self):
        """
        Handles various Server-end processes/tasks.
        """
        if isinstance(self.config, dict) and 'generator' in self.config:
            if self.config['generator'] is None or not inspect.isasyncgen(self.config['generator']):
                raise ValueError('[NetGear_Async:ERROR] :: Invalid configuration. Assigned generator must be a asynchronous generator function/method only!')
        else:
            raise RuntimeError('[NetGear_Async:ERROR] :: Assigned NetGear_Async configuration is invalid!')
        self.__msg_socket = self.__msg_context.socket(self.__pattern[0])
        if self.__msg_pattern == 1:
            self.__msg_socket.REQ_RELAXED = True
            self.__msg_socket.REQ_CORRELATE = True
        if self.__msg_pattern == 2:
            self.__msg_socket.set_hwm(1)
        try:
            self.__msg_socket.connect(self.__protocol + '://' + str(self.__address) + ':' + str(self.__port))
            if self.__logging:
                logger.debug('Successfully connected to address: {} with pattern: {}.'.format(self.__protocol + '://' + str(self.__address) + ':' + str(self.__port), self.__msg_pattern))
            logger.critical('Send Mode is successfully activated and ready to send data!')
        except Exception as e:
            logger.exception(str(e))
            if self.__bi_mode:
                logger.error('Failed to activate Bidirectional Mode for this connection!')
            raise ValueError('[NetGear_Async:ERROR] :: Failed to connect address: {} and pattern: {}!'.format(self.__protocol + '://' + str(self.__address) + ':' + str(self.__port), self.__msg_pattern))
        async for dataframe in self.config['generator']:
            if self.__bi_mode and len(dataframe) == 2:
                (data, frame) = dataframe
                if not data is None and isinstance(data, np.ndarray):
                    logger.warning('Skipped unsupported `data` of datatype: {}!'.format(type(data).__name__))
                    data = None
                assert isinstance(frame, np.ndarray), '[NetGear_Async:ERROR] :: Invalid data received from server end!'
            elif self.__bi_mode:
                raise ValueError('[NetGear_Async:ERROR] :: Send Mode only accepts tuple(data, frame) as input in Bidirectional Mode.                     Kindly refer vidgear docs!')
            else:
                frame = np.copy(dataframe)
                data = None
            if not frame.flags['C_CONTIGUOUS']:
                frame = np.ascontiguousarray(frame, dtype=frame.dtype)
            data_dict = dict(terminate=False, bi_mode=self.__bi_mode, data=data if not data is None else '')
            data_enc = msgpack.packb(data_dict)
            await self.__msg_socket.send(data_enc, flags=zmq.SNDMORE)
            frame_enc = msgpack.packb(frame, default=m.encode)
            await self.__msg_socket.send_multipart([frame_enc])
            if self.__msg_pattern < 2:
                if self.__bi_mode:
                    recvdmsg_encoded = await asyncio.wait_for(self.__msg_socket.recv(), timeout=self.__timeout)
                    recvd_data = msgpack.unpackb(recvdmsg_encoded, use_list=False)
                    if recvd_data['return_type'] == 'ndarray':
                        recvdframe_encoded = await asyncio.wait_for(self.__msg_socket.recv_multipart(), timeout=self.__timeout)
                        await self.__queue.put(msgpack.unpackb(recvdframe_encoded[0], use_list=False, object_hook=m.decode))
                    else:
                        await self.__queue.put(recvd_data['return_data'] if recvd_data['return_data'] else None)
                else:
                    recv_confirmation = await asyncio.wait_for(self.__msg_socket.recv(), timeout=self.__timeout)
                    if self.__logging:
                        logger.debug(recv_confirmation)

    async def recv_generator(self):
        """
        A default Asynchronous Frame Generator for NetGear_Async's Receiver-end.
        """
        if not self.__receive_mode:
            self.__terminate = True
            raise ValueError('[NetGear_Async:ERROR] :: `recv_generator()` function cannot be accessed while `receive_mode` is disabled. Kindly refer vidgear docs!')
        self.__msg_socket = self.__msg_context.socket(self.__pattern[1])
        if self.__msg_pattern == 2:
            self.__msg_socket.set_hwm(1)
            self.__msg_socket.setsockopt(zmq.SUBSCRIBE, b'')
        try:
            self.__msg_socket.bind(self.__protocol + '://' + str(self.__address) + ':' + str(self.__port))
            if self.__logging:
                logger.debug('Successfully binded to address: {} with pattern: {}.'.format(self.__protocol + '://' + str(self.__address) + ':' + str(self.__port), self.__msg_pattern))
            logger.critical('Receive Mode is activated successfully!')
        except Exception as e:
            logger.exception(str(e))
            raise RuntimeError('[NetGear_Async:ERROR] :: Failed to bind address: {} and pattern: {}{}!'.format(self.__protocol + '://' + str(self.__address) + ':' + str(self.__port), self.__msg_pattern, ' and Bidirectional Mode enabled' if self.__bi_mode else ''))
        while not self.__terminate:
            datamsg_encoded = await asyncio.wait_for(self.__msg_socket.recv(), timeout=self.__timeout)
            data = msgpack.unpackb(datamsg_encoded, use_list=False)
            if data['terminate']:
                if self.__msg_pattern < 2:
                    return_dict = dict(terminated='Client-`{}` successfully terminated!'.format(self.__id))
                    retdata_enc = msgpack.packb(return_dict)
                    await self.__msg_socket.send(retdata_enc)
                if self.__logging:
                    logger.info('Termination signal received from server!')
                self.__terminate = True
                break
            framemsg_encoded = await asyncio.wait_for(self.__msg_socket.recv_multipart(), timeout=self.__timeout)
            frame = msgpack.unpackb(framemsg_encoded[0], use_list=False, object_hook=m.decode)
            if self.__msg_pattern < 2:
                if self.__bi_mode and data['bi_mode']:
                    if not self.__queue.empty():
                        return_data = await self.__queue.get()
                        self.__queue.task_done()
                    else:
                        return_data = None
                    if not return_data is None and isinstance(return_data, np.ndarray):
                        if not return_data.flags['C_CONTIGUOUS']:
                            return_data = np.ascontiguousarray(return_data, dtype=return_data.dtype)
                        rettype_dict = dict(return_type=type(return_data).__name__, return_data=None)
                        rettype_enc = msgpack.packb(rettype_dict)
                        await self.__msg_socket.send(rettype_enc, flags=zmq.SNDMORE)
                        retframe_enc = msgpack.packb(return_data, default=m.encode)
                        await self.__msg_socket.send_multipart([retframe_enc])
                    else:
                        return_dict = dict(return_type=type(return_data).__name__, return_data=return_data if not return_data is None else '')
                        retdata_enc = msgpack.packb(return_dict)
                        await self.__msg_socket.send(retdata_enc)
                elif self.__bi_mode or data['bi_mode']:
                    raise RuntimeError('[NetGear_Async:ERROR] :: Invalid configuration! Bidirectional Mode is not activate on {} end.'.format('client' if self.__bi_mode else 'server'))
                else:
                    await self.__msg_socket.send(bytes('Data received on client: {} !'.format(self.__id), 'utf-8'))
            if self.__bi_mode:
                yield ((data['data'], frame) if data['data'] else (None, frame))
            else:
                yield frame
            await asyncio.sleep(0)

    async def __frame_generator(self):
        """
        Returns a default frame-generator for NetGear_Async's Server Handler.
        """
        self.__stream.start()
        while not self.__terminate:
            frame = self.__stream.read()
            if frame is None:
                break
            yield frame
            await asyncio.sleep(0)

    async def transceive_data(self, data=None):
        """
        Bidirectional Mode exclusive method to Transmit data _(in Receive mode)_ and Receive data _(in Send mode)_.

        Parameters:
            data (any): inputs data _(of any datatype)_ for sending back to Server.
        """
        recvd_data = None
        if not self.__terminate:
            if self.__bi_mode:
                if self.__receive_mode:
                    await self.__queue.put(data)
                elif not self.__queue.empty():
                    recvd_data = await self.__queue.get()
                    self.__queue.task_done()
            else:
                logger.error('`transceive_data()` function cannot be used when Bidirectional Mode is disabled.')
        return recvd_data

    async def __terminate_connection(self, disable_confirmation=False):
        """
        Internal asyncio method to safely terminate ZMQ connection and queues

        Parameters:
            disable_confirmation (boolean): Force disable termination confirmation from client in bidirectional patterns.
        """
        if self.__logging:
            logger.debug('Terminating various {} Processes. Please wait.'.format('Receive Mode' if self.__receive_mode else 'Send Mode'))
        if self.__receive_mode:
            self.__terminate = True
        else:
            self.__terminate = True
            if not self.__stream is None:
                self.__stream.stop()
            data_dict = dict(terminate=True)
            data_enc = msgpack.packb(data_dict)
            await self.__msg_socket.send(data_enc)
            if self.__msg_pattern < 2 and (not disable_confirmation):
                recv_confirmation = await self.__msg_socket.recv()
                recvd_conf = msgpack.unpackb(recv_confirmation, use_list=False)
                if self.__logging and 'terminated' in recvd_conf:
                    logger.debug(recvd_conf['terminated'])
        self.__msg_socket.setsockopt(zmq.LINGER, 0)
        self.__msg_socket.close()
        if self.__bi_mode:
            while not self.__queue.empty():
                try:
                    self.__queue.get_nowait()
                except asyncio.QueueEmpty:
                    continue
                self.__queue.task_done()
            await self.__queue.join()
        logger.critical('{} successfully terminated!'.format('Receive Mode' if self.__receive_mode else 'Send Mode'))

    def close(self, skip_loop=False):
        if False:
            print('Hello World!')
        "\n        Terminates all NetGear_Async Asynchronous processes gracefully.\n\n        Parameters:\n            skip_loop (Boolean): (optional)used only if don't want to close eventloop(required in pytest).\n        "
        if not skip_loop:
            self.loop.run_until_complete(self.__terminate_connection())
            self.loop.close()
        else:
            asyncio.ensure_future(self.__terminate_connection(disable_confirmation=True))