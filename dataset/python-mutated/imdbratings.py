import re
import json
from . import api_utils
from . import settings
try:
    from typing import Optional, Tuple, Text, Dict, List, Any
except ImportError:
    pass
IMDB_RATINGS_URL = 'https://www.imdb.com/title/{}/'
IMDB_JSON_REGEX = re.compile('<script type="application\\/ld\\+json">(.*?)<\\/script>')

def get_details(imdb_id):
    if False:
        return 10
    'get the IMDB ratings details'
    if not imdb_id:
        return {}
    (votes, rating) = _get_ratinginfo(imdb_id)
    return _assemble_imdb_result(votes, rating)

def _get_ratinginfo(imdb_id):
    if False:
        i = 10
        return i + 15
    'get the IMDB ratings details'
    response = api_utils.load_info(IMDB_RATINGS_URL.format(imdb_id), default='', resp_type='text', verboselog=settings.VERBOSELOG)
    return _parse_imdb_result(response)

def _assemble_imdb_result(votes, rating):
    if False:
        i = 10
        return i + 15
    'assemble to IMDB ratings into a Dict'
    result = {}
    if votes and rating:
        result['ratings'] = {'imdb': {'votes': votes, 'rating': rating}}
    return result

def _parse_imdb_result(input_html):
    if False:
        i = 10
        return i + 15
    'parse the IMDB ratings from the JSON in the raw HTML'
    match = re.search(IMDB_JSON_REGEX, input_html)
    if not match:
        return (None, None)
    imdb_json = json.loads(match.group(1))
    imdb_ratings = imdb_json.get('aggregateRating', {})
    rating = imdb_ratings.get('ratingValue', None)
    votes = imdb_ratings.get('ratingCount', None)
    return (votes, rating)