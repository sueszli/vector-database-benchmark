import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from devscripts.utils import get_filename_args, write_file
from yt_dlp.extractor import list_extractor_classes

def main():
    if False:
        return 10
    out = '\n'.join((ie.description() for ie in list_extractor_classes() if ie.IE_DESC is not False))
    write_file(get_filename_args(), f'# Supported sites\n{out}\n')
if __name__ == '__main__':
    main()