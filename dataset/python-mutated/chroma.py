"""Adds Chromaprint/Acoustid acoustic fingerprinting support to the
autotagger. Requires the pyacoustid library.
"""
import re
from collections import defaultdict
from functools import partial
import acoustid
import confuse
from beets import config, plugins, ui, util
from beets.autotag import hooks
API_KEY = '1vOwZtEn'
SCORE_THRESH = 0.5
TRACK_ID_WEIGHT = 10.0
COMMON_REL_THRESH = 0.6
MAX_RECORDINGS = 5
MAX_RELEASES = 5
_matches = {}
_fingerprints = {}
_acoustids = {}

def prefix(it, count):
    if False:
        for i in range(10):
            print('nop')
    'Truncate an iterable to at most `count` items.'
    for (i, v) in enumerate(it):
        if i >= count:
            break
        yield v

def releases_key(release, countries, original_year):
    if False:
        i = 10
        return i + 15
    'Used as a key to sort releases by date then preferred country'
    date = release.get('date')
    if date and original_year:
        year = date.get('year', 9999)
        month = date.get('month', 99)
        day = date.get('day', 99)
    else:
        year = 9999
        month = 99
        day = 99
    country_key = 99
    if release.get('country'):
        for (i, country) in enumerate(countries):
            if country.match(release['country']):
                country_key = i
                break
    return (year, month, day, country_key)

def acoustid_match(log, path):
    if False:
        i = 10
        return i + 15
    'Gets metadata for a file from Acoustid and populates the\n    _matches, _fingerprints, and _acoustids dictionaries accordingly.\n    '
    try:
        (duration, fp) = acoustid.fingerprint_file(util.syspath(path))
    except acoustid.FingerprintGenerationError as exc:
        log.error('fingerprinting of {0} failed: {1}', util.displayable_path(repr(path)), exc)
        return None
    fp = fp.decode()
    _fingerprints[path] = fp
    try:
        res = acoustid.lookup(API_KEY, fp, duration, meta='recordings releases')
    except acoustid.AcoustidError as exc:
        log.debug('fingerprint matching {0} failed: {1}', util.displayable_path(repr(path)), exc)
        return None
    log.debug('chroma: fingerprinted {0}', util.displayable_path(repr(path)))
    if res['status'] != 'ok' or not res.get('results'):
        log.debug('no match found')
        return None
    result = res['results'][0]
    if result['score'] < SCORE_THRESH:
        log.debug('no results above threshold')
        return None
    _acoustids[path] = result['id']
    if not result.get('recordings'):
        log.debug('no recordings found')
        return None
    recording_ids = []
    releases = []
    for recording in result['recordings']:
        recording_ids.append(recording['id'])
        if 'releases' in recording:
            releases.extend(recording['releases'])
    country_patterns = config['match']['preferred']['countries'].as_str_seq()
    countries = [re.compile(pat, re.I) for pat in country_patterns]
    original_year = config['match']['preferred']['original_year']
    releases.sort(key=partial(releases_key, countries=countries, original_year=original_year))
    release_ids = [rel['id'] for rel in releases]
    log.debug('matched recordings {0} on releases {1}', recording_ids, release_ids)
    _matches[path] = (recording_ids, release_ids)

def _all_releases(items):
    if False:
        for i in range(10):
            print('nop')
    'Given an iterable of Items, determines (according to Acoustid)\n    which releases the items have in common. Generates release IDs.\n    '
    relcounts = defaultdict(int)
    for item in items:
        if item.path not in _matches:
            continue
        (_, release_ids) = _matches[item.path]
        for release_id in release_ids:
            relcounts[release_id] += 1
    for (release_id, count) in relcounts.items():
        if float(count) / len(items) > COMMON_REL_THRESH:
            yield release_id

class AcoustidPlugin(plugins.BeetsPlugin):

    def __init__(self):
        if False:
            while True:
                i = 10
        super().__init__()
        self.config.add({'auto': True})
        config['acoustid']['apikey'].redact = True
        if self.config['auto']:
            self.register_listener('import_task_start', self.fingerprint_task)
        self.register_listener('import_task_apply', apply_acoustid_metadata)

    def fingerprint_task(self, task, session):
        if False:
            i = 10
            return i + 15
        return fingerprint_task(self._log, task, session)

    def track_distance(self, item, info):
        if False:
            i = 10
            return i + 15
        dist = hooks.Distance()
        if item.path not in _matches or not info.track_id:
            return dist
        (recording_ids, _) = _matches[item.path]
        dist.add_expr('track_id', info.track_id not in recording_ids)
        return dist

    def candidates(self, items, artist, album, va_likely, extra_tags=None):
        if False:
            i = 10
            return i + 15
        albums = []
        for relid in prefix(_all_releases(items), MAX_RELEASES):
            album = hooks.album_for_mbid(relid)
            if album:
                albums.append(album)
        self._log.debug('acoustid album candidates: {0}', len(albums))
        return albums

    def item_candidates(self, item, artist, title):
        if False:
            for i in range(10):
                print('nop')
        if item.path not in _matches:
            return []
        (recording_ids, _) = _matches[item.path]
        tracks = []
        for recording_id in prefix(recording_ids, MAX_RECORDINGS):
            track = hooks.track_for_mbid(recording_id)
            if track:
                tracks.append(track)
        self._log.debug('acoustid item candidates: {0}', len(tracks))
        return tracks

    def commands(self):
        if False:
            for i in range(10):
                print('nop')
        submit_cmd = ui.Subcommand('submit', help='submit Acoustid fingerprints')

        def submit_cmd_func(lib, opts, args):
            if False:
                for i in range(10):
                    print('nop')
            try:
                apikey = config['acoustid']['apikey'].as_str()
            except confuse.NotFoundError:
                raise ui.UserError('no Acoustid user API key provided')
            submit_items(self._log, apikey, lib.items(ui.decargs(args)))
        submit_cmd.func = submit_cmd_func
        fingerprint_cmd = ui.Subcommand('fingerprint', help='generate fingerprints for items without them')

        def fingerprint_cmd_func(lib, opts, args):
            if False:
                return 10
            for item in lib.items(ui.decargs(args)):
                fingerprint_item(self._log, item, write=ui.should_write())
        fingerprint_cmd.func = fingerprint_cmd_func
        return [submit_cmd, fingerprint_cmd]

def fingerprint_task(log, task, session):
    if False:
        for i in range(10):
            print('nop')
    'Fingerprint each item in the task for later use during the\n    autotagging candidate search.\n    '
    items = task.items if task.is_album else [task.item]
    for item in items:
        acoustid_match(log, item.path)

def apply_acoustid_metadata(task, session):
    if False:
        return 10
    "Apply Acoustid metadata (fingerprint and ID) to the task's items."
    for item in task.imported_items():
        if item.path in _fingerprints:
            item.acoustid_fingerprint = _fingerprints[item.path]
        if item.path in _acoustids:
            item.acoustid_id = _acoustids[item.path]

def submit_items(log, userkey, items, chunksize=64):
    if False:
        i = 10
        return i + 15
    'Submit fingerprints for the items to the Acoustid server.'
    data = []

    def submit_chunk():
        if False:
            while True:
                i = 10
        'Submit the current accumulated fingerprint data.'
        log.info('submitting {0} fingerprints', len(data))
        try:
            acoustid.submit(API_KEY, userkey, data)
        except acoustid.AcoustidError as exc:
            log.warning('acoustid submission error: {0}', exc)
        del data[:]
    for item in items:
        fp = fingerprint_item(log, item, write=ui.should_write())
        item_data = {'duration': int(item.length), 'fingerprint': fp}
        if item.mb_trackid:
            item_data['mbid'] = item.mb_trackid
            log.debug('submitting MBID')
        else:
            item_data.update({'track': item.title, 'artist': item.artist, 'album': item.album, 'albumartist': item.albumartist, 'year': item.year, 'trackno': item.track, 'discno': item.disc})
            log.debug('submitting textual metadata')
        data.append(item_data)
        if len(data) >= chunksize:
            submit_chunk()
    if data:
        submit_chunk()

def fingerprint_item(log, item, write=False):
    if False:
        print('Hello World!')
    "Get the fingerprint for an Item. If the item already has a\n    fingerprint, it is not regenerated. If fingerprint generation fails,\n    return None. If the items are associated with a library, they are\n    saved to the database. If `write` is set, then the new fingerprints\n    are also written to files' metadata.\n    "
    if not item.length:
        log.info('{0}: no duration available', util.displayable_path(item.path))
    elif item.acoustid_fingerprint:
        if write:
            log.info('{0}: fingerprint exists, skipping', util.displayable_path(item.path))
        else:
            log.info('{0}: using existing fingerprint', util.displayable_path(item.path))
        return item.acoustid_fingerprint
    else:
        log.info('{0}: fingerprinting', util.displayable_path(item.path))
        try:
            (_, fp) = acoustid.fingerprint_file(util.syspath(item.path))
            item.acoustid_fingerprint = fp.decode()
            if write:
                log.info('{0}: writing fingerprint', util.displayable_path(item.path))
                item.try_write()
            if item._db:
                item.store()
            return item.acoustid_fingerprint
        except acoustid.FingerprintGenerationError as exc:
            log.info('fingerprint generation failed: {0}', exc)