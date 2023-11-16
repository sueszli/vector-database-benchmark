from __future__ import annotations
import sys

def _is_rabbitmq_running(mq_urls: list[str]) -> bool:
    if False:
        while True:
            i = 10
    'Connect to rabbitmq with connection logic that mirrors the st2 code.\n\n    In particular, this is based on:\n      - st2common.transport.utils.get_connection()\n      - st2common.transport.bootstrap_utils.register_exchanges()\n\n    This should not import the st2 code as it should be self-contained.\n    '
    from kombu import Connection
    with Connection(mq_urls) as connection:
        try:
            connection.connect()
        except connection.connection_errors:
            return False
    return True
if __name__ == '__main__':
    mq_urls = list(sys.argv[1:])
    if not mq_urls:
        mq_urls = ['amqp://guest:guest@127.0.0.1:5672//']
    is_running = _is_rabbitmq_running(mq_urls)
    exit_code = 0 if is_running else 1
    sys.exit(exit_code)