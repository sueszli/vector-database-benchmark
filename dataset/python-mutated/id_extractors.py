"""Helpers around the extraction of album/track ID's from metadata sources."""
import re
spotify_id_regex = {'pattern': '(^|open\\.spotify\\.com/{}/)([0-9A-Za-z]{{22}})', 'match_group': 2}
deezer_id_regex = {'pattern': '(^|deezer\\.com/)([a-z]*/)?({}/)?(\\d+)', 'match_group': 4}
beatport_id_regex = {'pattern': '(^|beatport\\.com/release/.+/)(\\d+)$', 'match_group': 2}

def extract_discogs_id_regex(album_id):
    if False:
        i = 10
        return i + 15
    'Returns the Discogs_id or None.'
    for pattern in ['^\\[?r?(?P<id>\\d+)\\]?$', 'discogs\\.com/release/(?P<id>\\d+)-?', 'discogs\\.com/[^/]+/release/(?P<id>\\d+)']:
        match = re.search(pattern, album_id)
        if match:
            return int(match.group('id'))
    return None