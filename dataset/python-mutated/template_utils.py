import functools
import os.path
import re
from datetime import datetime
from email.utils import formatdate
import flask
from werkzeug.urls import url_encode
from nyaa.backend import get_category_id_map
from nyaa.torrents import create_magnet
app = flask.current_app
bp = flask.Blueprint('template-utils', __name__)
_static_cache = {}

@bp.app_context_processor
def create_magnet_from_es_torrent():
    if False:
        i = 10
        return i + 15
    return dict(create_magnet_from_es_torrent=create_magnet)
flask_url_for = flask.url_for

@functools.lru_cache(maxsize=1024 * 4)
def _caching_url_for(endpoint, **values):
    if False:
        return 10
    return flask_url_for(endpoint, **values)

@bp.app_template_global()
def caching_url_for(*args, **kwargs):
    if False:
        print('Hello World!')
    try:
        return _caching_url_for(*args, **kwargs)
    except TypeError:
        return flask_url_for(*args, **kwargs)

@bp.app_template_global()
def static_cachebuster(filename):
    if False:
        for i in range(10):
            print('nop')
    ' Adds a ?t=<mtime> cachebuster to the given path, if the file exists.\n        Results are cached in memory and persist until app restart! '
    if app.debug:
        return flask.url_for('static', filename=filename)
    if filename not in _static_cache:
        file_path = os.path.join(app.static_folder, filename)
        file_mtime = None
        if os.path.exists(file_path):
            file_mtime = int(os.path.getmtime(file_path))
        _static_cache[filename] = file_mtime
    return flask.url_for('static', filename=filename, t=_static_cache[filename])

@bp.app_template_global()
def modify_query(**new_values):
    if False:
        for i in range(10):
            print('nop')
    args = flask.request.args.copy()
    args.pop('p', None)
    for (key, value) in new_values.items():
        args[key] = value
    return '{}?{}'.format(flask.request.path, url_encode(args))

@bp.app_template_global()
def filter_truthy(input_list):
    if False:
        i = 10
        return i + 15
    " Jinja2 can't into list comprehension so this is for\n        the search_results.html template "
    return [item for item in input_list if item]

@bp.app_template_global()
def category_name(cat_id):
    if False:
        for i in range(10):
            print('nop')
    ' Given a category id (eg. 1_2), returns a category name (eg. Anime - English-translated) '
    return ' - '.join(get_category_id_map().get(cat_id, ['???']))

@bp.app_template_filter('utc_time')
def get_utc_timestamp(datetime_str):
    if False:
        return 10
    ' Returns a UTC POSIX timestamp, as seconds '
    UTC_EPOCH = datetime.utcfromtimestamp(0)
    return int((datetime.strptime(datetime_str, '%Y-%m-%dT%H:%M:%S') - UTC_EPOCH).total_seconds())

@bp.app_template_filter('utc_timestamp')
def get_utc_timestamp_seconds(datetime_instance):
    if False:
        for i in range(10):
            print('nop')
    ' Returns a UTC POSIX timestamp, as seconds '
    UTC_EPOCH = datetime.utcfromtimestamp(0)
    return int((datetime_instance - UTC_EPOCH).total_seconds())

@bp.app_template_filter('display_time')
def get_display_time(datetime_str):
    if False:
        print('Hello World!')
    return datetime.strptime(datetime_str, '%Y-%m-%dT%H:%M:%S').strftime('%Y-%m-%d %H:%M')

@bp.app_template_filter('rfc822')
def _jinja2_filter_rfc822(date, fmt=None):
    if False:
        print('Hello World!')
    return formatdate(date.timestamp())

@bp.app_template_filter('rfc822_es')
def _jinja2_filter_rfc822_es(datestr, fmt=None):
    if False:
        i = 10
        return i + 15
    return formatdate(datetime.strptime(datestr, '%Y-%m-%dT%H:%M:%S').timestamp())

@bp.app_template_filter()
def timesince(dt, default='just now'):
    if False:
        for i in range(10):
            print('nop')
    '\n    Returns string representing "time since" e.g.\n    3 minutes ago, 5 hours ago etc.\n    Date and time (UTC) are returned if older than 1 day.\n    '
    now = datetime.utcnow()
    diff = now - dt
    periods = ((diff.days, 'day', 'days'), (diff.seconds / 3600, 'hour', 'hours'), (diff.seconds / 60, 'minute', 'minutes'), (diff.seconds, 'second', 'seconds'))
    if diff.days >= 1:
        return dt.strftime('%Y-%m-%d %H:%M UTC')
    else:
        for (period, singular, plural) in periods:
            if period >= 1:
                return '%d %s ago' % (period, singular if int(period) == 1 else plural)
    return default

@bp.app_template_filter()
def regex_replace(s, find, replace):
    if False:
        while True:
            i = 10
    'A non-optimal implementation of a regex filter'
    return re.sub(find, replace, s)