"""Implements the transport for the file:// schema."""
import io
import os.path
SCHEME = 'file'
URI_EXAMPLES = ('./local/path/file', '~/local/path/file', 'local/path/file', './local/path/file.gz', 'file:///home/user/file', 'file:///home/user/file.bz2')
open = io.open

def parse_uri(uri_as_string):
    if False:
        i = 10
        return i + 15
    local_path = extract_local_path(uri_as_string)
    return dict(scheme=SCHEME, uri_path=local_path)

def open_uri(uri_as_string, mode, transport_params):
    if False:
        return 10
    parsed_uri = parse_uri(uri_as_string)
    fobj = io.open(parsed_uri['uri_path'], mode)
    return fobj

def extract_local_path(uri_as_string):
    if False:
        while True:
            i = 10
    if uri_as_string.startswith('file://'):
        local_path = uri_as_string.replace('file://', '', 1)
    else:
        local_path = uri_as_string
    return os.path.expanduser(local_path)