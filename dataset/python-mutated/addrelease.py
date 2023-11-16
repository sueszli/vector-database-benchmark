from html import escape
from secrets import token_bytes
from PyQt6.QtCore import QCoreApplication
from picard import log
from picard.util import format_time
from picard.util.mbserver import build_submission_url
from picard.util.webbrowser2 import open
try:
    import jwt
    import jwt.exceptions
except ImportError:
    log.debug('PyJWT not available, addrelease functionality disabled')
    jwt = None
__key = token_bytes()
__algorithm = 'HS256'
_form_template = '<!doctype html>\n<meta charset="UTF-8">\n<html>\n<head>\n    <title>{title}</title>\n</head>\n<body>\n    <form action="{action}" method="post">\n        {form_data}\n        <input type="submit" value="{submit_label}">\n    </form>\n    <script>document.forms[0].submit()</script>\n</body>\n'
_form_input_template = '<input type="hidden" name="{name}" value="{value}" >'

class InvalidTokenError(Exception):
    pass

class NotFoundError(Exception):
    pass

def is_available():
    if False:
        i = 10
        return i + 15
    return jwt is not None

def is_enabled():
    if False:
        print('Hello World!')
    tagger = QCoreApplication.instance()
    return tagger.browser_integration.is_running

def submit_cluster(cluster):
    if False:
        for i in range(10):
            print('nop')
    _open_url_with_token({'cluster': hash(cluster)})

def submit_file(file, as_release=False):
    if False:
        for i in range(10):
            print('nop')
    _open_url_with_token({'file': file.filename, 'as_release': as_release})

def serve_form(token):
    if False:
        print('Hello World!')
    try:
        payload = jwt.decode(token, __key, algorithms=__algorithm)
        log.debug('received JWT token %r', payload)
        tagger = QCoreApplication.instance()
        tport = tagger.browser_integration.port
        if 'cluster' in payload:
            cluster = _find_cluster(tagger, payload['cluster'])
            if not cluster:
                raise NotFoundError('Cluster not found')
            return _get_cluster_form(cluster, tport)
        elif 'file' in payload:
            file = _find_file(tagger, payload['file'])
            if not file:
                raise NotFoundError('File not found')
            if payload.get('as_release', False):
                return _get_file_as_release_form(file, tport)
            else:
                return _get_file_as_recording_form(file, tport)
        else:
            raise InvalidTokenError
    except jwt.exceptions.InvalidTokenError:
        raise InvalidTokenError

def extract_discnumber(metadata):
    if False:
        for i in range(10):
            print('nop')
    try:
        discnumber = metadata.get('discnumber', '1').split('/')[0]
        return int(discnumber)
    except ValueError:
        return 1

def _open_url_with_token(payload):
    if False:
        i = 10
        return i + 15
    token = jwt.encode(payload, __key, algorithm=__algorithm)
    if isinstance(token, bytes):
        token = token.decode()
    browser_integration = QCoreApplication.instance().browser_integration
    url = f'http://127.0.0.1:{browser_integration.port}/add?token={token}'
    open(url)

def _find_cluster(tagger, cluster_hash):
    if False:
        for i in range(10):
            print('nop')
    for cluster in tagger.clusters:
        if hash(cluster) == cluster_hash:
            return cluster
    return None

def _find_file(tagger, path):
    if False:
        while True:
            i = 10
    return tagger.files.get(path, None)

def _get_cluster_form(cluster, tport):
    if False:
        return 10
    return _get_form(_('Add cluster as release'), '/release/add', _('Add cluster as release…'), _get_cluster_data(cluster), {'tport': tport})

def _get_file_as_release_form(file, tport):
    if False:
        i = 10
        return i + 15
    return _get_form(_('Add file as release'), '/release/add', _('Add file as release…'), _get_file_as_release_data(file), {'tport': tport})

def _get_file_as_recording_form(file, tport):
    if False:
        print('Hello World!')
    return _get_form(_('Add file as recording'), '/recording/create', _('Add file as recording…'), _get_file_as_recording_data(file), {'tport': tport})

def _get_cluster_data(cluster):
    if False:
        i = 10
        return i + 15
    metadata = cluster.metadata
    data = {'name': metadata['album'], 'artist_credit.names.0.artist.name': metadata['albumartist']}
    _add_track_data(data, cluster.files)
    return data

def _get_file_as_release_data(file):
    if False:
        while True:
            i = 10
    metadata = file.metadata
    data = {'name': metadata['album'] or metadata['title'], 'artist_credit.names.0.artist.name': metadata['albumartist'] or metadata['artist']}
    _add_track_data(data, [file])
    return data

def _get_file_as_recording_data(file):
    if False:
        print('Hello World!')
    metadata = file.metadata
    data = {'edit-recording.name': metadata['title'], 'edit-recording.artist_credit.names.0.artist.name': metadata['artist'], 'edit-recording.length': format_time(file.metadata.length)}
    return data

def _add_track_data(data, files):
    if False:
        for i in range(10):
            print('nop')

    def mkey(disc, track, name):
        if False:
            print('Hello World!')
        return 'mediums.%i.track.%i.%s' % (disc, track, name)
    labels = set()
    barcode = None
    disc_counter = 0
    track_counter = 0
    last_discnumber = None
    for f in files:
        m = f.metadata
        discnumber = extract_discnumber(m)
        if last_discnumber is not None and discnumber != last_discnumber:
            disc_counter += 1
            track_counter = 0
        last_discnumber = discnumber
        if m['label'] or m['catalognumber']:
            labels.add((m['label'], m['catalognumber']))
        if m['barcode']:
            barcode = m['barcode']
        data[mkey(disc_counter, track_counter, 'name')] = m['title']
        data[mkey(disc_counter, track_counter, 'artist_credit.names.0.name')] = m['artist']
        data[mkey(disc_counter, track_counter, 'number')] = m['tracknumber'] or str(track_counter + 1)
        data[mkey(disc_counter, track_counter, 'length')] = str(m.length)
        if m['musicbrainz_recordingid']:
            data[mkey(disc_counter, track_counter, 'recording')] = m['musicbrainz_recordingid']
        track_counter += 1
    for (i, label) in enumerate(labels):
        (label, catalog_number) = label
        data['labels.%i.name' % i] = label
        data['labels.%i.catalog_number' % i] = catalog_number
    if barcode:
        data['barcode'] = barcode

def _get_form(title, action, label, form_data, query_args=None):
    if False:
        return 10
    return _form_template.format(title=escape(title), submit_label=escape(label), action=escape(build_submission_url(action, query_args)), form_data=_format_form_data(form_data))

def _format_form_data(data):
    if False:
        return 10
    return ''.join((_form_input_template.format(name=escape(name), value=escape(value)) for (name, value) in data.items()))