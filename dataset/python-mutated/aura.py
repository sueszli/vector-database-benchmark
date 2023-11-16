"""An AURA server using Flask."""
import os.path
import re
from mimetypes import guess_type
from os.path import getsize, isfile
from flask import Blueprint, Flask, current_app, make_response, request, send_file
from beets import config
from beets.dbcore.query import AndQuery, FixedFieldSort, MatchQuery, MultipleSort, NotQuery, RegexpQuery, SlowFieldSort
from beets.library import Album, Item
from beets.plugins import BeetsPlugin
from beets.ui import Subcommand, _open_library
from beets.util import py3_path
SERVER_INFO = {'aura-version': '0', 'server': 'beets-aura', 'server-version': '0.1', 'auth-required': False, 'features': ['albums', 'artists', 'images']}
TRACK_ATTR_MAP = {'title': 'title', 'artist': 'artist', 'album': 'album', 'track': 'track', 'tracktotal': 'tracktotal', 'disc': 'disc', 'disctotal': 'disctotal', 'year': 'year', 'month': 'month', 'day': 'day', 'bpm': 'bpm', 'genre': 'genre', 'recording-mbid': 'mb_trackid', 'track-mbid': 'mb_releasetrackid', 'composer': 'composer', 'albumartist': 'albumartist', 'comments': 'comments', 'duration': 'length', 'framerate': 'samplerate', 'channels': 'channels', 'bitrate': 'bitrate', 'bitdepth': 'bitdepth', 'size': 'filesize'}
ALBUM_ATTR_MAP = {'title': 'album', 'artist': 'albumartist', 'tracktotal': 'albumtotal', 'disctotal': 'disctotal', 'year': 'year', 'month': 'month', 'day': 'day', 'genre': 'genre', 'release-mbid': 'mb_albumid', 'release-group-mbid': 'mb_releasegroupid'}
ARTIST_ATTR_MAP = {'name': 'artist', 'artist-mbid': 'mb_artistid'}

class AURADocument:
    """Base class for building AURA documents."""

    @staticmethod
    def error(status, title, detail):
        if False:
            return 10
        'Make a response for an error following the JSON:API spec.\n\n        Args:\n            status: An HTTP status code string, e.g. "404 Not Found".\n            title: A short, human-readable summary of the problem.\n            detail: A human-readable explanation specific to this\n                occurrence of the problem.\n        '
        document = {'errors': [{'status': status, 'title': title, 'detail': detail}]}
        return make_response(document, status)

    def translate_filters(self):
        if False:
            while True:
                i = 10
        'Translate filters from request arguments to a beets Query.'
        pattern = re.compile('filter\\[(?P<attribute>[a-zA-Z0-9_-]+)\\]')
        queries = []
        for (key, value) in request.args.items():
            match = pattern.match(key)
            if match:
                aura_attr = match.group('attribute')
                beets_attr = self.attribute_map.get(aura_attr, aura_attr)
                converter = self.get_attribute_converter(beets_attr)
                value = converter(value)
                queries.append(MatchQuery(beets_attr, value, fast=False))
        return AndQuery(queries)

    def translate_sorts(self, sort_arg):
        if False:
            i = 10
            return i + 15
        'Translate an AURA sort parameter into a beets Sort.\n\n        Args:\n            sort_arg: The value of the \'sort\' query parameter; a comma\n                separated list of fields to sort by, in order.\n                E.g. "-year,title".\n        '
        aura_sorts = sort_arg.strip(',').split(',')
        sorts = []
        for aura_attr in aura_sorts:
            if aura_attr[0] == '-':
                ascending = False
                aura_attr = aura_attr[1:]
            else:
                ascending = True
            beets_attr = self.attribute_map.get(aura_attr, aura_attr)
            sorts.append(SlowFieldSort(beets_attr, ascending=ascending))
        return MultipleSort(sorts)

    def paginate(self, collection):
        if False:
            for i in range(10):
                print('nop')
        'Get a page of the collection and the URL to the next page.\n\n        Args:\n            collection: The raw data from which resource objects can be\n                built. Could be an sqlite3.Cursor object (tracks and\n                albums) or a list of strings (artists).\n        '
        page = request.args.get('page', 0, int)
        default_limit = config['aura']['page_limit'].get(int)
        limit = request.args.get('limit', default_limit, int)
        start = page * limit
        end = start + limit
        if end > len(collection):
            end = len(collection)
            next_url = None
        elif not request.args:
            next_url = request.url + '?page=1'
        elif not request.args.get('page', None):
            next_url = request.url + '&page=1'
        else:
            next_url = request.url.replace(f'page={page}', 'page={}'.format(page + 1))
        data = [self.resource_object(collection[i]) for i in range(start, end)]
        return (data, next_url)

    def get_included(self, data, include_str):
        if False:
            print('Hello World!')
        'Build a list of resource objects for inclusion.\n\n        Args:\n            data: An array of dicts in the form of resource objects.\n            include_str: A comma separated list of resource types to\n                include. E.g. "tracks,images".\n        '
        to_include = include_str.strip(',').split(',')
        unique_identifiers = []
        for res_obj in data:
            for (rel_name, rel_obj) in res_obj['relationships'].items():
                if rel_name in to_include:
                    for identifier in rel_obj['data']:
                        if identifier not in unique_identifiers:
                            unique_identifiers.append(identifier)
        included = []
        for identifier in unique_identifiers:
            res_type = identifier['type']
            if res_type == 'track':
                track_id = int(identifier['id'])
                track = current_app.config['lib'].get_item(track_id)
                included.append(TrackDocument.resource_object(track))
            elif res_type == 'album':
                album_id = int(identifier['id'])
                album = current_app.config['lib'].get_album(album_id)
                included.append(AlbumDocument.resource_object(album))
            elif res_type == 'artist':
                artist_id = identifier['id']
                included.append(ArtistDocument.resource_object(artist_id))
            elif res_type == 'image':
                image_id = identifier['id']
                included.append(ImageDocument.resource_object(image_id))
            else:
                raise ValueError(f'Invalid resource type: {res_type}')
        return included

    def all_resources(self):
        if False:
            return 10
        'Build document for /tracks, /albums or /artists.'
        query = self.translate_filters()
        sort_arg = request.args.get('sort', None)
        if sort_arg:
            sort = self.translate_sorts(sort_arg)
            for s in sort.sorts:
                query.subqueries.append(NotQuery(RegexpQuery(s.field, '(^$|^0$)', fast=False)))
        else:
            sort = None
        collection = self.get_collection(query=query, sort=sort)
        (data, next_url) = self.paginate(collection)
        document = {'data': data}
        if next_url:
            document['links'] = {'next': next_url}
        include_str = request.args.get('include', None)
        if include_str:
            document['included'] = self.get_included(data, include_str)
        return document

    def single_resource_document(self, resource_object):
        if False:
            return 10
        'Build document for a specific requested resource.\n\n        Args:\n            resource_object: A dictionary in the form of a JSON:API\n                resource object.\n        '
        document = {'data': resource_object}
        include_str = request.args.get('include', None)
        if include_str:
            document['included'] = self.get_included([document['data']], include_str)
        return document

class TrackDocument(AURADocument):
    """Class for building documents for /tracks endpoints."""
    attribute_map = TRACK_ATTR_MAP

    def get_collection(self, query=None, sort=None):
        if False:
            return 10
        'Get Item objects from the library.\n\n        Args:\n            query: A beets Query object or a beets query string.\n            sort: A beets Sort object.\n        '
        return current_app.config['lib'].items(query, sort)

    def get_attribute_converter(self, beets_attr):
        if False:
            while True:
                i = 10
        'Work out what data type an attribute should be for beets.\n\n        Args:\n            beets_attr: The name of the beets attribute, e.g. "title".\n        '
        if beets_attr == 'filesize':
            converter = int
        else:
            try:
                converter = Item._fields[beets_attr].model_type
            except KeyError:
                converter = str
        return converter

    @staticmethod
    def resource_object(track):
        if False:
            print('Hello World!')
        'Construct a JSON:API resource object from a beets Item.\n\n        Args:\n            track: A beets Item object.\n        '
        attributes = {}
        for (aura_attr, beets_attr) in TRACK_ATTR_MAP.items():
            a = getattr(track, beets_attr)
            if a:
                attributes[aura_attr] = a
        relationships = {'artists': {'data': [{'type': 'artist', 'id': track.artist}]}}
        if not track.singleton:
            relationships['albums'] = {'data': [{'type': 'album', 'id': str(track.album_id)}]}
        return {'type': 'track', 'id': str(track.id), 'attributes': attributes, 'relationships': relationships}

    def single_resource(self, track_id):
        if False:
            i = 10
            return i + 15
        'Get track from the library and build a document.\n\n        Args:\n            track_id: The beets id of the track (integer).\n        '
        track = current_app.config['lib'].get_item(track_id)
        if not track:
            return self.error('404 Not Found', 'No track with the requested id.', 'There is no track with an id of {} in the library.'.format(track_id))
        return self.single_resource_document(self.resource_object(track))

class AlbumDocument(AURADocument):
    """Class for building documents for /albums endpoints."""
    attribute_map = ALBUM_ATTR_MAP

    def get_collection(self, query=None, sort=None):
        if False:
            return 10
        'Get Album objects from the library.\n\n        Args:\n            query: A beets Query object or a beets query string.\n            sort: A beets Sort object.\n        '
        return current_app.config['lib'].albums(query, sort)

    def get_attribute_converter(self, beets_attr):
        if False:
            return 10
        'Work out what data type an attribute should be for beets.\n\n        Args:\n            beets_attr: The name of the beets attribute, e.g. "title".\n        '
        try:
            converter = Album._fields[beets_attr].model_type
        except KeyError:
            converter = str
        return converter

    @staticmethod
    def resource_object(album):
        if False:
            return 10
        'Construct a JSON:API resource object from a beets Album.\n\n        Args:\n            album: A beets Album object.\n        '
        attributes = {}
        for (aura_attr, beets_attr) in ALBUM_ATTR_MAP.items():
            a = getattr(album, beets_attr)
            if a:
                attributes[aura_attr] = a
        query = MatchQuery('album_id', album.id)
        sort = FixedFieldSort('track', ascending=True)
        tracks = current_app.config['lib'].items(query, sort)
        relationships = {'tracks': {'data': [{'type': 'track', 'id': str(t.id)} for t in tracks]}}
        if album.artpath:
            path = py3_path(album.artpath)
            filename = path.split('/')[-1]
            image_id = f'album-{album.id}-{filename}'
            relationships['images'] = {'data': [{'type': 'image', 'id': image_id}]}
        if album.albumartist in [t.artist for t in tracks]:
            relationships['artists'] = {'data': [{'type': 'artist', 'id': album.albumartist}]}
        return {'type': 'album', 'id': str(album.id), 'attributes': attributes, 'relationships': relationships}

    def single_resource(self, album_id):
        if False:
            for i in range(10):
                print('nop')
        'Get album from the library and build a document.\n\n        Args:\n            album_id: The beets id of the album (integer).\n        '
        album = current_app.config['lib'].get_album(album_id)
        if not album:
            return self.error('404 Not Found', 'No album with the requested id.', 'There is no album with an id of {} in the library.'.format(album_id))
        return self.single_resource_document(self.resource_object(album))

class ArtistDocument(AURADocument):
    """Class for building documents for /artists endpoints."""
    attribute_map = ARTIST_ATTR_MAP

    def get_collection(self, query=None, sort=None):
        if False:
            i = 10
            return i + 15
        'Get a list of artist names from the library.\n\n        Args:\n            query: A beets Query object or a beets query string.\n            sort: A beets Sort object.\n        '
        tracks = current_app.config['lib'].items(query, sort)
        collection = []
        for track in tracks:
            if track.artist not in collection:
                collection.append(track.artist)
        return collection

    def get_attribute_converter(self, beets_attr):
        if False:
            return 10
        'Work out what data type an attribute should be for beets.\n\n        Args:\n            beets_attr: The name of the beets attribute, e.g. "artist".\n        '
        try:
            converter = Item._fields[beets_attr].model_type
        except KeyError:
            converter = str
        return converter

    @staticmethod
    def resource_object(artist_id):
        if False:
            while True:
                i = 10
        "Construct a JSON:API resource object for the given artist.\n\n        Args:\n            artist_id: A string which is the artist's name.\n        "
        query = MatchQuery('artist', artist_id)
        tracks = current_app.config['lib'].items(query)
        if not tracks:
            return None
        attributes = {}
        for (aura_attr, beets_attr) in ARTIST_ATTR_MAP.items():
            a = getattr(tracks[0], beets_attr)
            if a:
                attributes[aura_attr] = a
        relationships = {'tracks': {'data': [{'type': 'track', 'id': str(t.id)} for t in tracks]}}
        album_query = MatchQuery('albumartist', artist_id)
        albums = current_app.config['lib'].albums(query=album_query)
        if len(albums) != 0:
            relationships['albums'] = {'data': [{'type': 'album', 'id': str(a.id)} for a in albums]}
        return {'type': 'artist', 'id': artist_id, 'attributes': attributes, 'relationships': relationships}

    def single_resource(self, artist_id):
        if False:
            i = 10
            return i + 15
        "Get info for the requested artist and build a document.\n\n        Args:\n            artist_id: A string which is the artist's name.\n        "
        artist_resource = self.resource_object(artist_id)
        if not artist_resource:
            return self.error('404 Not Found', 'No artist with the requested id.', 'There is no artist with an id of {} in the library.'.format(artist_id))
        return self.single_resource_document(artist_resource)

def safe_filename(fn):
    if False:
        print('Hello World!')
    'Check whether a string is a simple (non-path) filename.\n\n    For example, `foo.txt` is safe because it is a "plain" filename. But\n    `foo/bar.txt` and `../foo.txt` and `.` are all non-safe because they\n    can traverse to other directories other than the current one.\n    '
    if os.path.basename(fn) != fn:
        return False
    if fn in ('.', '..'):
        return False
    return True

class ImageDocument(AURADocument):
    """Class for building documents for /images/(id) endpoints."""

    @staticmethod
    def get_image_path(image_id):
        if False:
            while True:
                i = 10
        'Works out the full path to the image with the given id.\n\n        Returns None if there is no such image.\n\n        Args:\n            image_id: A string in the form\n                "<parent_type>-<parent_id>-<img_filename>".\n        '
        id_split = image_id.split('-')
        if len(id_split) < 3:
            return None
        parent_type = id_split[0]
        parent_id = id_split[1]
        img_filename = '-'.join(id_split[2:])
        if not safe_filename(img_filename):
            return None
        if parent_type == 'album':
            album = current_app.config['lib'].get_album(int(parent_id))
            if not album or not album.artpath:
                return None
            artpath = py3_path(album.artpath)
            dir_path = '/'.join(artpath.split('/')[:-1])
        else:
            return None
        img_path = os.path.join(dir_path, img_filename)
        if isfile(img_path):
            return img_path
        else:
            return None

    @staticmethod
    def resource_object(image_id):
        if False:
            for i in range(10):
                print('nop')
        'Construct a JSON:API resource object for the given image.\n\n        Args:\n            image_id: A string in the form\n                "<parent_type>-<parent_id>-<img_filename>".\n        '
        image_path = ImageDocument.get_image_path(image_id)
        if not image_path:
            return None
        attributes = {'role': 'cover', 'mimetype': guess_type(image_path)[0], 'size': getsize(image_path)}
        try:
            from PIL import Image
        except ImportError:
            pass
        else:
            im = Image.open(image_path)
            attributes['width'] = im.width
            attributes['height'] = im.height
        relationships = {}
        id_split = image_id.split('-')
        relationships[id_split[0] + 's'] = {'data': [{'type': id_split[0], 'id': id_split[1]}]}
        return {'id': image_id, 'type': 'image', 'attributes': {k: v for (k, v) in attributes.items() if v}, 'relationships': relationships}

    def single_resource(self, image_id):
        if False:
            while True:
                i = 10
        'Get info for the requested image and build a document.\n\n        Args:\n            image_id: A string in the form\n                "<parent_type>-<parent_id>-<img_filename>".\n        '
        image_resource = self.resource_object(image_id)
        if not image_resource:
            return self.error('404 Not Found', 'No image with the requested id.', 'There is no image with an id of {} in the library.'.format(image_id))
        return self.single_resource_document(image_resource)
aura_bp = Blueprint('aura_bp', __name__)

@aura_bp.route('/server')
def server_info():
    if False:
        for i in range(10):
            print('nop')
    'Respond with info about the server.'
    return {'data': {'type': 'server', 'id': '0', 'attributes': SERVER_INFO}}

@aura_bp.route('/tracks')
def all_tracks():
    if False:
        i = 10
        return i + 15
    'Respond with a list of all tracks and related information.'
    doc = TrackDocument()
    return doc.all_resources()

@aura_bp.route('/tracks/<int:track_id>')
def single_track(track_id):
    if False:
        for i in range(10):
            print('nop')
    'Respond with info about the specified track.\n\n    Args:\n        track_id: The id of the track provided in the URL (integer).\n    '
    doc = TrackDocument()
    return doc.single_resource(track_id)

@aura_bp.route('/tracks/<int:track_id>/audio')
def audio_file(track_id):
    if False:
        for i in range(10):
            print('nop')
    'Supply an audio file for the specified track.\n\n    Args:\n        track_id: The id of the track provided in the URL (integer).\n    '
    track = current_app.config['lib'].get_item(track_id)
    if not track:
        return AURADocument.error('404 Not Found', 'No track with the requested id.', 'There is no track with an id of {} in the library.'.format(track_id))
    path = py3_path(track.path)
    if not isfile(path):
        return AURADocument.error('404 Not Found', 'No audio file for the requested track.', 'There is no audio file for track {} at the expected location'.format(track_id))
    file_mimetype = guess_type(path)[0]
    if not file_mimetype:
        return AURADocument.error('500 Internal Server Error', 'Requested audio file has an unknown mimetype.', 'The audio file for track {} has an unknown mimetype. Its file extension is {}.'.format(track_id, path.split('.')[-1]))
    if not request.accept_mimetypes.best_match([file_mimetype]):
        return AURADocument.error('406 Not Acceptable', 'Unsupported MIME type or bitrate parameter in Accept header.', 'The audio file for track {} is only available as {} and bitrate parameters are not supported.'.format(track_id, file_mimetype))
    return send_file(path, mimetype=file_mimetype, as_attachment=True, conditional=True)

@aura_bp.route('/albums')
def all_albums():
    if False:
        print('Hello World!')
    'Respond with a list of all albums and related information.'
    doc = AlbumDocument()
    return doc.all_resources()

@aura_bp.route('/albums/<int:album_id>')
def single_album(album_id):
    if False:
        print('Hello World!')
    'Respond with info about the specified album.\n\n    Args:\n        album_id: The id of the album provided in the URL (integer).\n    '
    doc = AlbumDocument()
    return doc.single_resource(album_id)

@aura_bp.route('/artists')
def all_artists():
    if False:
        while True:
            i = 10
    'Respond with a list of all artists and related information.'
    doc = ArtistDocument()
    return doc.all_resources()

@aura_bp.route('/artists/<path:artist_id>')
def single_artist(artist_id):
    if False:
        return 10
    "Respond with info about the specified artist.\n\n    Args:\n        artist_id: The id of the artist provided in the URL. A string\n            which is the artist's name.\n    "
    doc = ArtistDocument()
    return doc.single_resource(artist_id)

@aura_bp.route('/images/<string:image_id>')
def single_image(image_id):
    if False:
        return 10
    'Respond with info about the specified image.\n\n    Args:\n        image_id: The id of the image provided in the URL. A string in\n            the form "<parent_type>-<parent_id>-<img_filename>".\n    '
    doc = ImageDocument()
    return doc.single_resource(image_id)

@aura_bp.route('/images/<string:image_id>/file')
def image_file(image_id):
    if False:
        return 10
    'Supply an image file for the specified image.\n\n    Args:\n        image_id: The id of the image provided in the URL. A string in\n            the form "<parent_type>-<parent_id>-<img_filename>".\n    '
    img_path = ImageDocument.get_image_path(image_id)
    if not img_path:
        return AURADocument.error('404 Not Found', 'No image with the requested id.', 'There is no image with an id of {} in the library'.format(image_id))
    return send_file(img_path)

def create_app():
    if False:
        i = 10
        return i + 15
    'An application factory for use by a WSGI server.'
    config['aura'].add({'host': '127.0.0.1', 'port': 8337, 'cors': [], 'cors_supports_credentials': False, 'page_limit': 500})
    app = Flask(__name__)
    app.register_blueprint(aura_bp, url_prefix='/aura')
    app.config['JSONIFY_MIMETYPE'] = 'application/vnd.api+json'
    app.config['JSON_SORT_KEYS'] = False
    app.config['lib'] = _open_library(config)
    cors = config['aura']['cors'].as_str_seq(list)
    if cors:
        from flask_cors import CORS
        app.config['CORS_ALLOW_HEADERS'] = 'Accept'
        app.config['CORS_RESOURCES'] = {'/aura/*': {'origins': cors}}
        app.config['CORS_SUPPORTS_CREDENTIALS'] = config['aura']['cors_supports_credentials'].get(bool)
        CORS(app)
    return app

class AURAPlugin(BeetsPlugin):
    """The BeetsPlugin subclass for the AURA server plugin."""

    def __init__(self):
        if False:
            while True:
                i = 10
        'Add configuration options for the AURA plugin.'
        super().__init__()

    def commands(self):
        if False:
            print('Hello World!')
        'Add subcommand used to run the AURA server.'

        def run_aura(lib, opts, args):
            if False:
                print('Hello World!')
            "Run the application using Flask's built in-server.\n\n            Args:\n                lib: A beets Library object (not used).\n                opts: Command line options. An optparse.Values object.\n                args: The list of arguments to process (not used).\n            "
            app = create_app()
            app.run(host=self.config['host'].get(str), port=self.config['port'].get(int), debug=opts.debug, threaded=True)
        run_aura_cmd = Subcommand('aura', help='run an AURA server')
        run_aura_cmd.parser.add_option('-d', '--debug', action='store_true', default=False, help='use Flask debug mode')
        run_aura_cmd.func = run_aura
        return [run_aura_cmd]