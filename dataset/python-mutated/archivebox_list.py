__package__ = 'archivebox.cli'
__command__ = 'archivebox list'
import sys
import argparse
from typing import Optional, List, IO
from ..main import list_all
from ..util import docstring
from ..config import OUTPUT_DIR
from ..index import LINK_FILTERS, get_indexed_folders, get_archived_folders, get_unarchived_folders, get_present_folders, get_valid_folders, get_invalid_folders, get_duplicate_folders, get_orphaned_folders, get_corrupted_folders, get_unrecognized_folders
from ..logging_util import SmartFormatter, reject_stdin, stderr

@docstring(list_all.__doc__)
def main(args: Optional[List[str]]=None, stdin: Optional[IO]=None, pwd: Optional[str]=None) -> None:
    if False:
        return 10
    parser = argparse.ArgumentParser(prog=__command__, description=list_all.__doc__, add_help=True, formatter_class=SmartFormatter)
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--csv', type=str, help='Print the output in CSV format with the given columns, e.g.: timestamp,url,extension', default=None)
    group.add_argument('--json', action='store_true', help='Print the output in JSON format with all columns included')
    group.add_argument('--html', action='store_true', help='Print the output in HTML format')
    parser.add_argument('--with-headers', action='store_true', help='Include the headers in the output document')
    parser.add_argument('--sort', type=str, help='List the links sorted using the given key, e.g. timestamp or updated', default=None)
    parser.add_argument('--before', type=float, help='List only links bookmarked before (less than) the given timestamp', default=None)
    parser.add_argument('--after', type=float, help='List only links bookmarked after (greater than or equal to) the given timestamp', default=None)
    parser.add_argument('--status', type=str, choices=('indexed', 'archived', 'unarchived', 'present', 'valid', 'invalid', 'duplicate', 'orphaned', 'corrupted', 'unrecognized'), default='indexed', help=f'List only links or data directories that have the given status\n    indexed       {get_indexed_folders.__doc__} (the default)\n    archived      {get_archived_folders.__doc__}\n    unarchived    {get_unarchived_folders.__doc__}\n\n    present       {get_present_folders.__doc__}\n    valid         {get_valid_folders.__doc__}\n    invalid       {get_invalid_folders.__doc__}\n\n    duplicate     {get_duplicate_folders.__doc__}\n    orphaned      {get_orphaned_folders.__doc__}\n    corrupted     {get_corrupted_folders.__doc__}\n    unrecognized  {get_unrecognized_folders.__doc__}\n')
    parser.add_argument('--filter-type', '-t', type=str, choices=(*LINK_FILTERS.keys(), 'search'), default='exact', help='Type of pattern matching to use when filtering URLs')
    parser.add_argument('filter_patterns', nargs='*', type=str, default=None, help='List only URLs matching these filter patterns')
    command = parser.parse_args(args or ())
    reject_stdin(stdin)
    if command.with_headers and (not (command.json or command.html or command.csv)):
        stderr('[X] --with-headers can only be used with --json, --html or --csv options\n', color='red')
        raise SystemExit(2)
    matching_folders = list_all(filter_patterns=command.filter_patterns, filter_type=command.filter_type, status=command.status, after=command.after, before=command.before, sort=command.sort, csv=command.csv, json=command.json, html=command.html, with_headers=command.with_headers, out_dir=pwd or OUTPUT_DIR)
    raise SystemExit(not matching_folders)
if __name__ == '__main__':
    main(args=sys.argv[1:], stdin=sys.stdin)