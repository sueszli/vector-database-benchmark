"""Functions to interact with TMDb API."""
from . import api_utils
import xbmc
try:
    from typing import Optional, Text, Dict, List, Any
    InfoType = Dict[Text, Any]
except ImportError:
    pass
HEADERS = (('User-Agent', 'Kodi Movie scraper by Team Kodi'), ('Accept', 'application/json'))
api_utils.set_headers(dict(HEADERS))
TMDB_PARAMS = {'api_key': 'f090bb54758cabf231fb605d3e3e0468'}
BASE_URL = 'https://api.themoviedb.org/3/{}'
SEARCH_URL = BASE_URL.format('search/movie')
FIND_URL = BASE_URL.format('find/{}')
MOVIE_URL = BASE_URL.format('movie/{}')
COLLECTION_URL = BASE_URL.format('collection/{}')
CONFIG_URL = BASE_URL.format('configuration')

def search_movie(query, year=None, language=None):
    if False:
        return 10
    '\n    Search for a movie\n\n    :param title: movie title to search\n    :param year: the year to search (optional)\n    :param language: the language filter for TMDb (optional)\n    :return: a list with found movies\n    '
    xbmc.log('using title of %s to find movie' % query, xbmc.LOGDEBUG)
    theurl = SEARCH_URL
    params = _set_params(None, language)
    params['query'] = query
    if year is not None:
        params['year'] = str(year)
    return api_utils.load_info(theurl, params=params)

def find_movie_by_external_id(external_id, language=None):
    if False:
        for i in range(10):
            print('nop')
    '\n    Find movie based on external ID\n\n    :param mid: external ID\n    :param language: the language filter for TMDb (optional)\n    :return: the movie or error\n    '
    xbmc.log('using external id of %s to find movie' % external_id, xbmc.LOGDEBUG)
    theurl = FIND_URL.format(external_id)
    params = _set_params(None, language)
    params['external_source'] = 'imdb_id'
    return api_utils.load_info(theurl, params=params)

def get_movie(mid, language=None, append_to_response=None):
    if False:
        return 10
    '\n    Get movie details\n\n    :param mid: TMDb movie ID\n    :param language: the language filter for TMDb (optional)\n    :append_to_response: the additional data to get from TMDb (optional)\n    :return: the movie or error\n    '
    xbmc.log('using movie id of %s to get movie details' % mid, xbmc.LOGDEBUG)
    theurl = MOVIE_URL.format(mid)
    return api_utils.load_info(theurl, params=_set_params(append_to_response, language))

def get_collection(collection_id, language=None, append_to_response=None):
    if False:
        i = 10
        return i + 15
    '\n    Get movie collection information\n\n    :param collection_id: TMDb collection ID\n    :param language: the language filter for TMDb (optional)\n    :append_to_response: the additional data to get from TMDb (optional)\n    :return: the movie or error\n    '
    xbmc.log('using collection id of %s to get collection details' % collection_id, xbmc.LOGDEBUG)
    theurl = COLLECTION_URL.format(collection_id)
    return api_utils.load_info(theurl, params=_set_params(append_to_response, language))

def get_configuration():
    if False:
        print('Hello World!')
    '\n    Get configuration information\n\n    :return: configuration details or error\n    '
    xbmc.log('getting configuration details', xbmc.LOGDEBUG)
    return api_utils.load_info(CONFIG_URL, params=TMDB_PARAMS.copy())

def _set_params(append_to_response, language):
    if False:
        print('Hello World!')
    params = TMDB_PARAMS.copy()
    img_lang = 'en,null'
    if language is not None:
        params['language'] = language
        img_lang = '%s,en,null' % language[0:2]
    if append_to_response is not None:
        params['append_to_response'] = append_to_response
        if 'images' in append_to_response:
            params['include_image_language'] = img_lang
    return params