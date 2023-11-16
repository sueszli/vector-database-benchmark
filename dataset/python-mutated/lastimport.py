import pylast
from pylast import TopItem, _extract, _number
from beets import config, dbcore, plugins, ui
from beets.dbcore import types
API_URL = 'https://ws.audioscrobbler.com/2.0/'

class LastImportPlugin(plugins.BeetsPlugin):

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        super().__init__()
        config['lastfm'].add({'user': '', 'api_key': plugins.LASTFM_KEY})
        config['lastfm']['api_key'].redact = True
        self.config.add({'per_page': 500, 'retry_limit': 3})
        self.item_types = {'play_count': types.INTEGER}

    def commands(self):
        if False:
            print('Hello World!')
        cmd = ui.Subcommand('lastimport', help='import last.fm play-count')

        def func(lib, opts, args):
            if False:
                for i in range(10):
                    print('nop')
            import_lastfm(lib, self._log)
        cmd.func = func
        return [cmd]

class CustomUser(pylast.User):
    """Custom user class derived from pylast.User, and overriding the
    _get_things method to return MBID and album. Also introduces new
    get_top_tracks_by_page method to allow access to more than one page of top
    tracks.
    """

    def __init__(self, *args, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        super().__init__(*args, **kwargs)

    def _get_things(self, method, thing, thing_type, params=None, cacheable=True):
        if False:
            i = 10
            return i + 15
        'Returns a list of the most played thing_types by this thing, in a\n        tuple with the total number of pages of results. Includes an MBID, if\n        found.\n        '
        doc = self._request(self.ws_prefix + '.' + method, cacheable, params)
        toptracks_node = doc.getElementsByTagName('toptracks')[0]
        total_pages = int(toptracks_node.getAttribute('totalPages'))
        seq = []
        for node in doc.getElementsByTagName(thing):
            title = _extract(node, 'name')
            artist = _extract(node, 'name', 1)
            mbid = _extract(node, 'mbid')
            playcount = _number(_extract(node, 'playcount'))
            thing = thing_type(artist, title, self.network)
            thing.mbid = mbid
            seq.append(TopItem(thing, playcount))
        return (seq, total_pages)

    def get_top_tracks_by_page(self, period=pylast.PERIOD_OVERALL, limit=None, page=1, cacheable=True):
        if False:
            for i in range(10):
                print('nop')
        'Returns the top tracks played by a user, in a tuple with the total\n        number of pages of results.\n        * period: The period of time. Possible values:\n          o PERIOD_OVERALL\n          o PERIOD_7DAYS\n          o PERIOD_1MONTH\n          o PERIOD_3MONTHS\n          o PERIOD_6MONTHS\n          o PERIOD_12MONTHS\n        '
        params = self._get_params()
        params['period'] = period
        params['page'] = page
        if limit:
            params['limit'] = limit
        return self._get_things('getTopTracks', 'track', pylast.Track, params, cacheable)

def import_lastfm(lib, log):
    if False:
        print('Hello World!')
    user = config['lastfm']['user'].as_str()
    per_page = config['lastimport']['per_page'].get(int)
    if not user:
        raise ui.UserError('You must specify a user name for lastimport')
    log.info('Fetching last.fm library for @{0}', user)
    page_total = 1
    page_current = 0
    found_total = 0
    unknown_total = 0
    retry_limit = config['lastimport']['retry_limit'].get(int)
    while page_current < page_total:
        log.info('Querying page #{0}{1}...', page_current + 1, f'/{page_total}' if page_total > 1 else '')
        for retry in range(0, retry_limit):
            (tracks, page_total) = fetch_tracks(user, page_current + 1, per_page)
            if page_total < 1:
                raise ui.UserError('Last.fm reported no data.')
            if tracks:
                (found, unknown) = process_tracks(lib, tracks, log)
                found_total += found
                unknown_total += unknown
                break
            else:
                log.error('ERROR: unable to read page #{0}', page_current + 1)
                if retry < retry_limit:
                    log.info('Retrying page #{0}... ({1}/{2} retry)', page_current + 1, retry + 1, retry_limit)
                else:
                    log.error('FAIL: unable to fetch page #{0}, ', 'tried {1} times', page_current, retry + 1)
        page_current += 1
    log.info('... done!')
    log.info('finished processing {0} song pages', page_total)
    log.info('{0} unknown play-counts', unknown_total)
    log.info('{0} play-counts imported', found_total)

def fetch_tracks(user, page, limit):
    if False:
        print('Hello World!')
    'JSON format:\n    [\n        {\n            "mbid": "...",\n            "artist": "...",\n            "title": "...",\n            "playcount": "..."\n        }\n    ]\n    '
    network = pylast.LastFMNetwork(api_key=config['lastfm']['api_key'])
    user_obj = CustomUser(user, network)
    (results, total_pages) = user_obj.get_top_tracks_by_page(limit=limit, page=page)
    return ([{'mbid': track.item.mbid if track.item.mbid else '', 'artist': {'name': track.item.artist.name}, 'name': track.item.title, 'playcount': track.weight} for track in results], total_pages)

def process_tracks(lib, tracks, log):
    if False:
        i = 10
        return i + 15
    total = len(tracks)
    total_found = 0
    total_fails = 0
    log.info('Received {0} tracks in this page, processing...', total)
    for num in range(0, total):
        song = None
        trackid = tracks[num]['mbid'].strip()
        artist = tracks[num]['artist'].get('name', '').strip()
        title = tracks[num]['name'].strip()
        album = ''
        if 'album' in tracks[num]:
            album = tracks[num]['album'].get('name', '').strip()
        log.debug('query: {0} - {1} ({2})', artist, title, album)
        if trackid:
            song = lib.items(dbcore.query.MatchQuery('mb_trackid', trackid)).get()
        if song is None:
            log.debug('no album match, trying by artist/title')
            query = dbcore.AndQuery([dbcore.query.SubstringQuery('artist', artist), dbcore.query.SubstringQuery('title', title)])
            song = lib.items(query).get()
        if song is None:
            title = title.replace("'", '’')
            log.debug('no title match, trying utf-8 single quote')
            query = dbcore.AndQuery([dbcore.query.SubstringQuery('artist', artist), dbcore.query.SubstringQuery('title', title)])
            song = lib.items(query).get()
        if song is not None:
            count = int(song.get('play_count', 0))
            new_count = int(tracks[num]['playcount'])
            log.debug('match: {0} - {1} ({2}) updating: play_count {3} => {4}', song.artist, song.title, song.album, count, new_count)
            song['play_count'] = new_count
            song.store()
            total_found += 1
        else:
            total_fails += 1
            log.info('  - No match: {0} - {1} ({2})', artist, title, album)
    if total_fails > 0:
        log.info('Acquired {0}/{1} play-counts ({2} unknown)', total_found, total, total_fails)
    return (total_found, total_fails)