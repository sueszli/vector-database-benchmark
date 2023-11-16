from collections import namedtuple
from picard.config import get_config
from picard.const import MUSICBRAINZ_SERVERS
from picard.util import build_qurl
ServerTuple = namedtuple('ServerTuple', ('host', 'port'))

def is_official_server(host):
    if False:
        i = 10
        return i + 15
    'Returns True, if host is an official MusicBrainz server for the primary database.\n\n    Args:\n        host: the hostname\n\n    Returns: True, if host is an official MusicBrainz server, False otherwise\n    '
    return host in MUSICBRAINZ_SERVERS

def get_submission_server():
    if False:
        i = 10
        return i + 15
    "Returns the host and port used for data submission.\n\n    Data submission usually should be done against the primary database. This function\n    will return the hostname configured as `server_host` if it is an official MusicBrainz\n    server, otherwise it will return the primary official server.\n\n    Returns: Tuple of hostname and port number, e.g. `('musicbrainz.org', 443)`\n    "
    config = get_config()
    host = config.setting['server_host']
    if is_official_server(host):
        return ServerTuple(host, 443)
    elif host and config.setting['use_server_for_submission']:
        port = config.setting['server_port']
        return ServerTuple(host, port)
    else:
        return ServerTuple(MUSICBRAINZ_SERVERS[0], 443)

def build_submission_url(path=None, query_args=None):
    if False:
        while True:
            i = 10
    'Builds a submission URL with path and query parameters.\n\n    Args:\n        path: The path for the URL\n        query_args: A dict of query parameters\n\n    Returns: The submission URL as a string\n    '
    server = get_submission_server()
    url = build_qurl(server.host, server.port, path, query_args)
    return url.toString()