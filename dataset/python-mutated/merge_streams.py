from argparse import ArgumentParser
from typing import Any
from typing_extensions import override
from zerver.actions.streams import merge_streams
from zerver.lib.management import ZulipBaseCommand
from zerver.models import get_stream

class Command(ZulipBaseCommand):
    help = 'Merge two streams.'

    @override
    def add_arguments(self, parser: ArgumentParser) -> None:
        if False:
            while True:
                i = 10
        parser.add_argument('stream_to_keep', help='name of stream to keep')
        parser.add_argument('stream_to_destroy', help='name of stream to merge into the stream being kept')
        self.add_realm_args(parser, required=True)

    @override
    def handle(self, *args: Any, **options: str) -> None:
        if False:
            i = 10
            return i + 15
        realm = self.get_realm(options)
        assert realm is not None
        stream_to_keep = get_stream(options['stream_to_keep'], realm)
        stream_to_destroy = get_stream(options['stream_to_destroy'], realm)
        stats = merge_streams(realm, stream_to_keep, stream_to_destroy)
        print(f'Added {stats[0]} subscriptions')
        print(f'Moved {stats[1]} messages')
        print(f'Deactivated {stats[2]} subscriptions')