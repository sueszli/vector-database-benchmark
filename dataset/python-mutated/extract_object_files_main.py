"""Command line tool version of the extract_object_files module.

This command line tool version takes the archive file path and the destination
directory path as the positional command line arguments.
"""
import sys
from typing import Sequence
from tensorflow.lite.ios import extract_object_files

def main(argv: Sequence[str]) -> None:
    if False:
        print('Hello World!')
    if len(argv) != 3:
        raise RuntimeError('Usage: {} <archive_file> <dest_dir>'.format(argv[0]))
    archive_path = argv[1]
    dest_dir = argv[2]
    with open(archive_path, 'rb') as archive_file:
        extract_object_files.extract_object_files(archive_file, dest_dir)
if __name__ == '__main__':
    main(sys.argv)