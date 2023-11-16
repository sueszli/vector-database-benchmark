"""
Simple command line client for Testing/debugging
a Freqtrade bot's message websocket

Should not import anything from freqtrade,
so it can be used as a standalone script.
"""
import argparse
import asyncio
import logging
import socket
import sys
import time
from pathlib import Path
import orjson
import pandas
import rapidjson
import websockets
logger = logging.getLogger('WebSocketClient')

def setup_logging(filename: str):
    if False:
        while True:
            i = 10
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', handlers=[logging.FileHandler(filename), logging.StreamHandler()])

def parse_args():
    if False:
        i = 10
        return i + 15
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', help='Specify configuration file (default: %(default)s). ', dest='config', type=str, metavar='PATH', default='config.json')
    parser.add_argument('-l', '--logfile', help='The filename to log to.', dest='logfile', type=str, default='ws_client.log')
    args = parser.parse_args()
    return vars(args)

def load_config(configfile):
    if False:
        return 10
    file = Path(configfile)
    if file.is_file():
        with file.open('r') as f:
            config = rapidjson.load(f, parse_mode=rapidjson.PM_COMMENTS | rapidjson.PM_TRAILING_COMMAS)
        return config
    else:
        logger.warning(f'Could not load config file {file}.')
        sys.exit(1)

def readable_timedelta(delta):
    if False:
        return 10
    '\n    Convert a millisecond delta to a readable format\n\n    :param delta: A delta between two timestamps in milliseconds\n    :returns: The readable time difference string\n    '
    (seconds, milliseconds) = divmod(delta, 1000)
    (minutes, seconds) = divmod(seconds, 60)
    return f'{int(minutes)}:{int(seconds)}.{int(milliseconds)}'

def json_serialize(message):
    if False:
        for i in range(10):
            print('nop')
    '\n    Serialize a message to JSON using orjson\n    :param message: The message to serialize\n    '
    return str(orjson.dumps(message), 'utf-8')

def json_deserialize(message):
    if False:
        return 10
    '\n    Deserialize JSON to a dict\n    :param message: The message to deserialize\n    '

    def json_to_dataframe(data: str) -> pandas.DataFrame:
        if False:
            while True:
                i = 10
        dataframe = pandas.read_json(data, orient='split')
        if 'date' in dataframe.columns:
            dataframe['date'] = pandas.to_datetime(dataframe['date'], unit='ms', utc=True)
        return dataframe

    def _json_object_hook(z):
        if False:
            return 10
        if z.get('__type__') == 'dataframe':
            return json_to_dataframe(z.get('__value__'))
        return z
    return rapidjson.loads(message, object_hook=_json_object_hook)

class ClientProtocol:
    logger = logging.getLogger('WebSocketClient.Protocol')
    _MESSAGE_COUNT = 0
    _LAST_RECEIVED_AT = 0

    async def on_connect(self, websocket):
        initial_requests = [{'type': 'subscribe', 'data': ['analyzed_df', 'whitelist']}, {'type': 'whitelist', 'data': None}, {'type': 'analyzed_df', 'data': {'limit': 1500}}]
        for request in initial_requests:
            await websocket.send(json_serialize(request))

    async def on_message(self, websocket, name, message):
        deserialized = json_deserialize(message)
        message_size = sys.getsizeof(message)
        message_type = deserialized.get('type')
        message_data = deserialized.get('data')
        self.logger.info(f'Received message of type {message_type} [{message_size} bytes] @ [{name}]')
        time_difference = self._calculate_time_difference()
        if self._MESSAGE_COUNT > 0:
            self.logger.info(f'Time since last message: {time_difference}')
        message_handler = getattr(self, f'_handle_{message_type}', None) or self._handle_default
        await message_handler(name, message_type, message_data)
        self._MESSAGE_COUNT += 1
        self.logger.info(f'[{self._MESSAGE_COUNT}] total messages..')
        self.logger.info('-' * 80)

    def _calculate_time_difference(self):
        if False:
            print('Hello World!')
        old_last_received_at = self._LAST_RECEIVED_AT
        self._LAST_RECEIVED_AT = time.time() * 1000.0
        time_delta = self._LAST_RECEIVED_AT - old_last_received_at
        return readable_timedelta(time_delta)

    async def _handle_whitelist(self, name, type, data):
        self.logger.info(data)

    async def _handle_analyzed_df(self, name, type, data):
        (key, la, df) = (data['key'], data['la'], data['df'])
        if not df.empty:
            columns = ', '.join([str(column) for column in df.columns])
            self.logger.info(key)
            self.logger.info(f'Last analyzed datetime: {la}')
            self.logger.info(f"Latest candle datetime: {df.iloc[-1]['date']}")
            self.logger.info(f'DataFrame length: {len(df)}')
            self.logger.info(f'DataFrame columns: {columns}')
        else:
            self.logger.info('Empty DataFrame')

    async def _handle_default(self, name, type, data):
        self.logger.info('Unkown message of type {type} received...')
        self.logger.info(data)

async def create_client(host, port, token, scheme='ws', name='default', protocol=ClientProtocol(), sleep_time=10, ping_timeout=10, wait_timeout=30, **kwargs):
    """
    Create a websocket client and listen for messages
    :param host: The host
    :param port: The port
    :param token: The websocket auth token
    :param scheme: `ws` for most connections, `wss` for ssl
    :param name: The name of the producer
    :param **kwargs: Any extra kwargs passed to websockets.connect
    """
    while 1:
        try:
            websocket_url = f'{scheme}://{host}:{port}/api/v1/message/ws?token={token}'
            logger.info(f'Attempting to connect to {name} @ {host}:{port}')
            async with websockets.connect(websocket_url, **kwargs) as ws:
                logger.info('Connection successful...')
                await protocol.on_connect(ws)
                while 1:
                    try:
                        message = await asyncio.wait_for(ws.recv(), timeout=wait_timeout)
                        await protocol.on_message(ws, name, message)
                    except (asyncio.TimeoutError, websockets.exceptions.WebSocketException):
                        try:
                            pong = await ws.ping()
                            latency = await asyncio.wait_for(pong, timeout=ping_timeout) * 1000
                            logger.info(f'Connection still alive, latency: {latency}ms')
                            continue
                        except asyncio.TimeoutError:
                            logger.error(f'Ping timed out, retrying in {sleep_time}s')
                            await asyncio.sleep(sleep_time)
                            break
        except (socket.gaierror, ConnectionRefusedError, websockets.exceptions.InvalidStatusCode, websockets.exceptions.InvalidMessage) as e:
            logger.error(f'Connection Refused - {e} retrying in {sleep_time}s')
            await asyncio.sleep(sleep_time)
            continue
        except (websockets.exceptions.ConnectionClosedError, websockets.exceptions.ConnectionClosedOK):
            logger.info('Connection was closed')
            await asyncio.sleep(sleep_time)
            continue
        except Exception as e:
            logger.error('Unexpected error has occurred:')
            logger.exception(e)
            await asyncio.sleep(sleep_time)
            continue

async def _main(args):
    setup_logging(args['logfile'])
    config = load_config(args['config'])
    emc_config = config.get('external_message_consumer', {})
    producers = emc_config.get('producers', [])
    producer = producers[0]
    wait_timeout = emc_config.get('wait_timeout', 30)
    ping_timeout = emc_config.get('ping_timeout', 10)
    sleep_time = emc_config.get('sleep_time', 10)
    message_size_limit = emc_config.get('message_size_limit', 8) << 20
    await create_client(producer['host'], producer['port'], producer['ws_token'], 'wss' if producer.get('secure', False) else 'ws', producer['name'], sleep_time=sleep_time, ping_timeout=ping_timeout, wait_timeout=wait_timeout, max_size=message_size_limit, ping_interval=None)

def main():
    if False:
        i = 10
        return i + 15
    args = parse_args()
    try:
        asyncio.run(_main(args))
    except KeyboardInterrupt:
        logger.info('Exiting...')
if __name__ == '__main__':
    main()