import os
from argparse import ArgumentParser
from typing import Any, Set
import orjson
from django.conf import settings
from typing_extensions import override
from urllib3.util import Retry
from zerver.lib.management import ZulipBaseCommand
from zerver.lib.outgoing_http import OutgoingSession

class TorDataSession(OutgoingSession):

    def __init__(self, max_retries: int) -> None:
        if False:
            return 10
        Retry.DEFAULT_BACKOFF_MAX = 64
        retry = Retry(total=max_retries, backoff_factor=2.0, status_forcelist={429, 500, 502, 503})
        super().__init__(role='tor_data', timeout=3, max_retries=retry)

class Command(ZulipBaseCommand):
    help = 'Fetch the list of TOR exit nodes, and write the list of IP addresses\nto a file for access from Django for rate-limiting purposes.\n\nDoes nothing unless RATE_LIMIT_TOR_TOGETHER is enabled.\n'

    @override
    def add_arguments(self, parser: ArgumentParser) -> None:
        if False:
            print('Hello World!')
        parser.add_argument('--max-retries', type=int, default=10, help='Number of times to retry fetching data from TOR')

    @override
    def handle(self, *args: Any, **options: Any) -> None:
        if False:
            print('Hello World!')
        if not settings.RATE_LIMIT_TOR_TOGETHER:
            return
        certificates = os.environ.get('CUSTOM_CA_CERTIFICATES')
        session = TorDataSession(max_retries=options['max_retries'])
        response = session.get('https://check.torproject.org/exit-addresses', verify=certificates)
        response.raise_for_status()
        exit_nodes: Set[str] = set()
        for line in response.text.splitlines():
            if line.startswith('ExitAddress '):
                exit_nodes.add(line.split()[1])
        with open(settings.TOR_EXIT_NODE_FILE_PATH + '.tmp', 'wb') as f:
            f.write(orjson.dumps(list(exit_nodes)))
        os.rename(settings.TOR_EXIT_NODE_FILE_PATH + '.tmp', settings.TOR_EXIT_NODE_FILE_PATH)