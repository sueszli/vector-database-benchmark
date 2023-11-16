import argparse
from ._common import with_repository, Highlander
from ..constants import *
from ..helpers import EXIT_SUCCESS
from ..manifest import Manifest
from ..logger import create_logger
logger = create_logger()

class CompactMixIn:

    @with_repository(manifest=False, exclusive=True)
    def do_compact(self, args, repository):
        if False:
            i = 10
            return i + 15
        'compact segment files in the repository'
        data = repository.get(Manifest.MANIFEST_ID)
        repository.put(Manifest.MANIFEST_ID, data)
        threshold = args.threshold / 100
        repository.commit(compact=True, threshold=threshold)
        return EXIT_SUCCESS

    def build_parser_compact(self, subparsers, common_parser, mid_common_parser):
        if False:
            i = 10
            return i + 15
        from ._common import process_epilog
        compact_epilog = process_epilog('\n        This command frees repository space by compacting segments.\n\n        Use this regularly to avoid running out of space - you do not need to use this\n        after each borg command though. It is especially useful after deleting archives,\n        because only compaction will really free repository space.\n\n        borg compact does not need a key, so it is possible to invoke it from the\n        client or also from the server.\n\n        Depending on the amount of segments that need compaction, it may take a while,\n        so consider using the ``--progress`` option.\n\n        A segment is compacted if the amount of saved space is above the percentage value\n        given by the ``--threshold`` option. If omitted, a threshold of 10% is used.\n        When using ``--verbose``, borg will output an estimate of the freed space.\n\n        See :ref:`separate_compaction` in Additional Notes for more details.\n        ')
        subparser = subparsers.add_parser('compact', parents=[common_parser], add_help=False, description=self.do_compact.__doc__, epilog=compact_epilog, formatter_class=argparse.RawDescriptionHelpFormatter, help='compact segment files / free space in repo')
        subparser.set_defaults(func=self.do_compact)
        subparser.add_argument('--threshold', metavar='PERCENT', dest='threshold', type=int, default=10, action=Highlander, help='set minimum threshold for saved space in PERCENT (Default: 10)')