"""Handle media queries.

https://www.w3.org/TR/mediaqueries-4/

"""
import tinycss2
from ..logger import LOGGER
from .utils import remove_whitespace, split_on_comma

def evaluate_media_query(query_list, device_media_type):
    if False:
        i = 10
        return i + 15
    'Return the boolean evaluation of `query_list` for the given\n    `device_media_type`.\n\n    :attr query_list: a cssutilts.stlysheets.MediaList\n    :attr device_media_type: a media type string (for now)\n\n    '
    return 'all' in query_list or device_media_type in query_list

def parse_media_query(tokens):
    if False:
        i = 10
        return i + 15
    tokens = remove_whitespace(tokens)
    if not tokens:
        return ['all']
    else:
        media = []
        for part in split_on_comma(tokens):
            types = [token.type for token in part]
            if types == ['ident']:
                media.append(part[0].lower_value)
            else:
                LOGGER.warning('Expected a media type, got %r', tinycss2.serialize(part))
                return
        return media