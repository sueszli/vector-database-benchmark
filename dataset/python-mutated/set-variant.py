import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import argparse
import functools
import re
from devscripts.utils import compose_functions, read_file, write_file
VERSION_FILE = 'yt_dlp/version.py'

def parse_options():
    if False:
        while True:
            i = 10
    parser = argparse.ArgumentParser(description='Set the build variant of the package')
    parser.add_argument('variant', help='Name of the variant')
    parser.add_argument('-M', '--update-message', default=None, help='Message to show in -U')
    return parser.parse_args()

def property_setter(name, value):
    if False:
        for i in range(10):
            print('nop')
    return functools.partial(re.sub, f'(?m)^{name}\\s*=\\s*.+$', f'{name} = {value!r}')
opts = parse_options()
transform = compose_functions(property_setter('VARIANT', opts.variant), property_setter('UPDATE_HINT', opts.update_message))
write_file(VERSION_FILE, transform(read_file(VERSION_FILE)))