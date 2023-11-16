"""Functions to interact with Trakt API"""
from __future__ import absolute_import, unicode_literals
from . import api_utils, settings
try:
    from typing import Text, Optional, Union, List, Dict, Any
except ImportError:
    pass
HEADERS = (('User-Agent', 'Kodi TV Show scraper by Team Kodi; contact pkscout@kodi.tv'), ('Accept', 'application/json'), ('trakt-api-key', settings.TRAKT_CLOWNCAR), ('trakt-api-version', '2'), ('Content-Type', 'application/json'))
api_utils.set_headers(dict(HEADERS))
SHOW_URL = 'https://api.trakt.tv/shows/{}'
EP_URL = SHOW_URL + '/seasons/{}/episodes/{}/ratings'

def get_details(imdb_id, season=None, episode=None):
    if False:
        return 10
    '\n    get the Trakt ratings\n\n    :param imdb_id:\n    :param season:\n    :param episode:\n    :return: trackt ratings\n    '
    result = {}
    if season and episode:
        url = EP_URL.format(imdb_id, season, episode)
        params = None
    else:
        url = SHOW_URL.format(imdb_id)
        params = {'extended': 'full'}
    resp = api_utils.load_info(url, params=params, default={}, verboselog=settings.VERBOSELOG)
    rating = resp.get('rating')
    votes = resp.get('votes')
    if votes and rating:
        result['ratings'] = {'trakt': {'votes': votes, 'rating': rating}}
    return result