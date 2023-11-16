"""Used by test_client.TestClient.test_sigstop_sigcont."""
from __future__ import annotations
import logging
import os
import sys
sys.path[0:0] = ['']
from pymongo import monitoring
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
SERVER_API = None
MONGODB_API_VERSION = os.environ.get('MONGODB_API_VERSION')
if MONGODB_API_VERSION:
    SERVER_API = ServerApi(MONGODB_API_VERSION)

class HeartbeatLogger(monitoring.ServerHeartbeatListener):
    """Log events until the listener is closed."""

    def __init__(self):
        if False:
            i = 10
            return i + 15
        self.closed = False

    def close(self):
        if False:
            print('Hello World!')
        self.closed = True

    def started(self, event: monitoring.ServerHeartbeatStartedEvent) -> None:
        if False:
            i = 10
            return i + 15
        if self.closed:
            return
        logging.info('%s', event)

    def succeeded(self, event: monitoring.ServerHeartbeatSucceededEvent) -> None:
        if False:
            return 10
        if self.closed:
            return
        logging.info('%s', event)

    def failed(self, event: monitoring.ServerHeartbeatFailedEvent) -> None:
        if False:
            for i in range(10):
                print('nop')
        if self.closed:
            return
        logging.warning('%s', event)

def main(uri: str) -> None:
    if False:
        i = 10
        return i + 15
    heartbeat_logger = HeartbeatLogger()
    client = MongoClient(uri, event_listeners=[heartbeat_logger], heartbeatFrequencyMS=500, connectTimeoutMS=500, server_api=SERVER_API)
    client.admin.command('ping')
    logging.info('TEST STARTED')
    while True:
        try:
            data = input('Type "q" to quit: ')
        except EOFError:
            break
        if data == 'q':
            break
    client.admin.command('ping')
    logging.info('TEST COMPLETED')
    heartbeat_logger.close()
    client.close()
if __name__ == '__main__':
    if len(sys.argv) != 2:
        print('unknown or missing options')
        print(f"usage: python3 {sys.argv[0]} 'mongodb://localhost'")
        sys.exit(1)
    FORMAT = '%(asctime)s %(levelname)s %(message)s'
    logging.basicConfig(format=FORMAT, level=logging.INFO)
    main(sys.argv[1])