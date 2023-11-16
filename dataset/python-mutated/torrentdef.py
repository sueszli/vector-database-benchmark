"""
Author(s): Arno Bakker
"""
import itertools
import logging
from asyncio import get_running_loop
from hashlib import sha1
import aiohttp
from tribler.core.components.libtorrent.utils.libtorrent_helper import libtorrent as lt
from tribler.core.components.libtorrent.utils import torrent_utils
from tribler.core.utilities import maketorrent, path_util
from tribler.core.utilities.path_util import Path
from tribler.core.utilities.simpledefs import INFOHASH_LENGTH
from tribler.core.utilities.unicode import ensure_unicode
from tribler.core.utilities.utilities import bdecode_compat, is_valid_url, parse_magnetlink

def escape_as_utf8(string, encoding='utf8'):
    if False:
        i = 10
        return i + 15
    '\n    Make a string UTF-8 compliant, destroying characters if necessary.\n\n    :param string: the string to convert\n    :type string: str\n    :return: the utf-8 string derivative\n    :rtype: str\n    '
    try:
        return string.decode(encoding).encode('utf8').decode('utf8')
    except (LookupError, TypeError, ValueError):
        try:
            return string.decode('latin1').encode('utf8', 'ignore').decode('utf8')
        except (TypeError, ValueError):
            return string.encode('utf8', 'ignore').decode('utf8')

class TorrentDef:
    """
    This object acts as a wrapper around some libtorrent metadata.
    It can be used to create new torrents, or analyze existing ones.
    """

    def __init__(self, metainfo=None, torrent_parameters=None, ignore_validation=False):
        if False:
            print('Hello World!')
        '\n        Create a new TorrentDef object, possibly based on existing data.\n        :param metainfo: A dictionary with metainfo, i.e. from a .torrent file.\n        :param torrent_parameters: User-defined parameters for the new TorrentDef.\n        :param ignore_validation: Whether we ignore the libtorrent validation.\n        '
        self._logger = logging.getLogger(self.__class__.__name__)
        self.torrent_parameters = {}
        self.metainfo = None
        self.files_list = []
        self.infohash = None
        if metainfo is not None:
            if not ignore_validation:
                try:
                    lt.torrent_info(metainfo)
                except RuntimeError as exc:
                    raise ValueError from exc
            self.metainfo = metainfo
            self.infohash = sha1(lt.bencode(self.metainfo[b'info'])).digest()
            self.copy_metainfo_to_torrent_parameters()
        elif torrent_parameters:
            self.torrent_parameters.update(torrent_parameters)

    def copy_metainfo_to_torrent_parameters(self):
        if False:
            print('Hello World!')
        '\n        Populate the torrent_parameters dictionary with information from the metainfo.\n        '
        for key in [b'comment', b'created by', b'creation date', b'announce', b'announce-list', b'nodes', b'httpseeds', b'urllist']:
            if self.metainfo and key in self.metainfo:
                self.torrent_parameters[key] = self.metainfo[key]
        infokeys = [b'name', b'piece length']
        for key in infokeys:
            if self.metainfo and key in self.metainfo[b'info']:
                self.torrent_parameters[key] = self.metainfo[b'info'][key]

    @staticmethod
    def _threaded_load_job(filepath):
        if False:
            print('Hello World!')
        "\n        Perform the actual loading of the torrent.\n\n        Called from a thread: don't call this directly!\n        "
        with open(filepath, 'rb') as torrent_file:
            file_content = torrent_file.read()
        return TorrentDef.load_from_memory(file_content)

    @staticmethod
    async def load(filepath):
        """
        Create a TorrentDef object from a .torrent file
        :param filepath: The path to the .torrent file
        """
        return await get_running_loop().run_in_executor(None, TorrentDef._threaded_load_job, filepath)

    @staticmethod
    def load_from_memory(bencoded_data):
        if False:
            while True:
                i = 10
        '\n        Load some bencoded data into a TorrentDef.\n        :param bencoded_data: The bencoded data to decode and use as metainfo\n        '
        metainfo = bdecode_compat(bencoded_data)
        if metainfo is None:
            raise ValueError('Data is not a bencoded string')
        return TorrentDef.load_from_dict(metainfo)

    @staticmethod
    def load_from_dict(metainfo):
        if False:
            for i in range(10):
                print('nop')
        '\n        Load a metainfo dictionary into a TorrentDef object.\n        :param metainfo: The metainfo dictionary\n        '
        return TorrentDef(metainfo=metainfo)

    @staticmethod
    async def load_from_url(url):
        """
        Create a TorrentDef with information from a remote source.
        :param url: The HTTP/HTTPS url where to fetch the torrent info from.
        """
        async with aiohttp.ClientSession(raise_for_status=True) as session:
            response = await session.get(url)
            body = await response.read()
        return TorrentDef.load_from_memory(body)

    def _filter_characters(self, name: bytes) -> str:
        if False:
            i = 10
            return i + 15
        "\n        Sanitize the names in path to unicode by replacing out all\n        characters that may -even remotely- cause problems with the '?'\n        character.\n\n        :param name: the name to sanitize\n        :type name: bytes\n        :return: the sanitized string\n        :rtype: str\n        "

        def filter_character(char: int) -> str:
            if False:
                while True:
                    i = 10
            if 0 < char < 128:
                return chr(char)
            self._logger.debug('Bad character 0x%X', char)
            return '?'
        return ''.join(map(filter_character, name))

    def add_content(self, file_path):
        if False:
            for i in range(10):
                print('nop')
        '\n        Add some content to the torrent file.\n        :param file_path: The (absolute) path of the file to add.\n        '
        self.files_list.append(Path(file_path).absolute())

    def set_encoding(self, enc):
        if False:
            i = 10
            return i + 15
        "\n        Set the character encoding for e.g. the 'name' field\n        :param enc: The new encoding of the file.\n        "
        self.torrent_parameters[b'encoding'] = enc

    def get_encoding(self):
        if False:
            while True:
                i = 10
        '\n        Returns the used encoding of the TorrentDef.\n        '
        return ensure_unicode(self.torrent_parameters.get(b'encoding', b'utf-8'), 'utf-8')

    def set_tracker(self, url):
        if False:
            print('Hello World!')
        '\n        Set the tracker of this torrent, according to a given URL.\n        :param url: The tracker url.\n        '
        if not is_valid_url(url):
            raise ValueError('Invalid URL')
        if url.endswith('/'):
            url = url[:-1]
        self.torrent_parameters[b'announce'] = url

    def get_tracker(self):
        if False:
            while True:
                i = 10
        '\n        Returns the torrent announce URL.\n        '
        return self.torrent_parameters.get(b'announce', None)

    def get_tracker_hierarchy(self):
        if False:
            return 10
        '\n        Returns the hierarchy of trackers.\n        '
        return self.torrent_parameters.get(b'announce-list', [])

    def get_trackers(self) -> set:
        if False:
            while True:
                i = 10
        '\n        Returns a flat set of all known trackers.\n\n        :return: all known trackers\n        :rtype: set\n        '
        if self.get_tracker_hierarchy():
            trackers = itertools.chain.from_iterable(self.get_tracker_hierarchy())
            return set(filter(None, trackers))
        tracker = self.get_tracker()
        if tracker:
            return {tracker}
        return set()

    def set_piece_length(self, piece_length):
        if False:
            for i in range(10):
                print('nop')
        '\n        Set the size of the pieces in which the content is traded.\n        The piece size must be a multiple of the chunk size, the unit in which\n        it is transmitted, which is 16K by default. The default is automatic (value 0).\n        :param piece_length: The piece length.\n        '
        if not isinstance(piece_length, int):
            raise ValueError('Piece length not an int/long')
        self.torrent_parameters[b'piece length'] = piece_length

    def get_piece_length(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Returns the piece size.\n        '
        return self.torrent_parameters.get(b'piece length', 0)

    def get_nr_pieces(self):
        if False:
            return 10
        '\n        Returns the number of pieces.\n        '
        if not self.metainfo:
            return 0
        return len(self.metainfo[b'info'][b'pieces']) // 20

    def get_pieces(self):
        if False:
            return 10
        '\n        Returns the pieces.\n        '
        if not self.metainfo:
            return []
        return self.metainfo[b'info'][b'pieces'][:]

    def get_infohash(self):
        if False:
            return 10
        '\n        Returns the infohash of the torrent, if metainfo is provided. Might be None if no metainfo is provided.\n        '
        return self.infohash

    def get_metainfo(self):
        if False:
            while True:
                i = 10
        '\n        Returns the metainfo of the torrent. Might be None if no metainfo is provided.\n        '
        return self.metainfo

    def get_name(self):
        if False:
            while True:
                i = 10
        '\n        Returns the name as raw string of bytes.\n        '
        return self.torrent_parameters[b'name']

    def get_name_utf8(self):
        if False:
            while True:
                i = 10
        '\n        Not all names are utf-8, attempt to construct it as utf-8 anyway.\n        '
        return escape_as_utf8(self.get_name(), self.get_encoding())

    def set_name(self, name):
        if False:
            while True:
                i = 10
        '\n        Set the name of this torrent.\n        :param name: The new name of the torrent\n        '
        self.torrent_parameters[b'name'] = name

    def get_name_as_unicode(self):
        if False:
            while True:
                i = 10
        " Returns the info['name'] field as Unicode string.\n        @return Unicode string. "
        if self.metainfo and b'name.utf-8' in self.metainfo[b'info']:
            try:
                return ensure_unicode(self.metainfo[b'info'][b'name.utf-8'], 'UTF-8')
            except UnicodeError:
                pass
        if self.metainfo and b'name' in self.metainfo[b'info']:
            if 'encoding' in self.metainfo:
                try:
                    return ensure_unicode(self.metainfo[b'info'][b'name'], self.metainfo[b'encoding'])
                except UnicodeError:
                    pass
                except LookupError:
                    pass
            try:
                return ensure_unicode(self.metainfo[b'info'][b'name'], 'UTF-8')
            except UnicodeError:
                pass
            try:
                return self._filter_characters(self.metainfo[b'info'][b'name'])
            except UnicodeError:
                pass
        return ''

    def save(self, torrent_filepath=None):
        if False:
            while True:
                i = 10
        '\n        Generate the metainfo and save the torrent file.\n        :param torrent_filepath: An optional absolute path to where to save the generated .torrent file.\n        '
        torrent_dict = torrent_utils.create_torrent_file(self.files_list, self.torrent_parameters, torrent_filepath=torrent_filepath)
        self.metainfo = bdecode_compat(torrent_dict['metainfo'])
        self.copy_metainfo_to_torrent_parameters()
        self.infohash = torrent_dict['infohash']

    def _get_all_files_as_unicode_with_length(self):
        if False:
            print('Hello World!')
        ' Get a generator for files in the torrent def. No filtering\n        is possible and all tricks are allowed to obtain a unicode\n        list of filenames.\n        @return A unicode filename generator.\n        '
        if self.metainfo and b'files' in self.metainfo[b'info']:
            files = self.metainfo[b'info'][b'files']
            for file_dict in files:
                if b'path.utf-8' in file_dict:
                    try:
                        yield (Path(*(ensure_unicode(element, 'UTF-8') for element in file_dict[b'path.utf-8'])), file_dict[b'length'])
                        continue
                    except UnicodeError:
                        pass
                if b'path' in file_dict:
                    if b'encoding' in self.metainfo:
                        encoding = ensure_unicode(self.metainfo[b'encoding'], 'utf8')
                        try:
                            yield (Path(*(ensure_unicode(element, encoding) for element in file_dict[b'path'])), file_dict[b'length'])
                            continue
                        except UnicodeError:
                            pass
                        except LookupError:
                            pass
                    try:
                        yield (Path(*(ensure_unicode(element, 'UTF-8') for element in file_dict[b'path'])), file_dict[b'length'])
                        continue
                    except UnicodeError:
                        pass
                    try:
                        yield (Path(*map(self._filter_characters, file_dict[b'path'])), file_dict[b'length'])
                        continue
                    except UnicodeError:
                        pass
        elif self.metainfo:
            yield (self.get_name_as_unicode(), self.metainfo[b'info'][b'length'])

    def get_files_with_length(self, exts=None):
        if False:
            return 10
        ' The list of files in the torrent def.\n        @param exts (Optional) list of filename extensions (without leading .)\n        to search for.\n        @return A list of filenames.\n        '
        videofiles = []
        for (filename, length) in self._get_all_files_as_unicode_with_length():
            ext = path_util.Path(filename).suffix
            if ext != '' and ext[0] == '.':
                ext = ext[1:]
            if exts is None or ext.lower() in exts:
                videofiles.append((filename, length))
        return videofiles

    def get_files(self, exts=None):
        if False:
            for i in range(10):
                print('nop')
        return [filename for (filename, _) in self.get_files_with_length(exts)]

    def get_length(self, selectedfiles=None):
        if False:
            while True:
                i = 10
        ' Returns the total size of the content in the torrent. If the\n        optional selectedfiles argument is specified, the method returns\n        the total size of only those files.\n        @return A length (long)\n        '
        if self.metainfo:
            return maketorrent.get_length_from_metainfo(self.metainfo, selectedfiles)
        return 0

    def get_creation_date(self):
        if False:
            while True:
                i = 10
        '\n        Returns the creation date of the torrent.\n        '
        return self.metainfo.get(b'creation date', 0) if self.metainfo else 0

    def is_multifile_torrent(self):
        if False:
            print('Hello World!')
        '\n        Returns whether this TorrentDef is a multi-file torrent.\n        '
        if self.metainfo:
            return b'files' in self.metainfo[b'info']
        return False

    def is_private(self) -> bool:
        if False:
            i = 10
            return i + 15
        '\n        Returns whether this TorrentDef is a private torrent (and is not announced in the DHT).\n        '
        try:
            private = int(self.metainfo[b'info'].get(b'private', 0)) if self.metainfo else 0
        except (ValueError, KeyError) as e:
            self._logger.warning(f'{e.__class__.__name__}: {e}')
            private = 0
        return private == 1

    def get_index_of_file_in_files(self, file):
        if False:
            print('Hello World!')
        if not self.metainfo:
            raise ValueError('TorrentDef does not have metainfo')
        info = self.metainfo[b'info']
        if file is not None and b'files' in info:
            for i in range(len(info[b'files'])):
                file_dict = info[b'files'][i]
                if b'path.utf-8' in file_dict:
                    intorrentpath = maketorrent.pathlist2filename(file_dict[b'path.utf-8'])
                else:
                    intorrentpath = maketorrent.pathlist2filename(file_dict[b'path'])
                if intorrentpath == path_util.Path(ensure_unicode(file, 'utf8')):
                    return i
            raise ValueError('File not found in torrent')
        else:
            raise ValueError('File not found in single-file torrent')

class TorrentDefNoMetainfo:
    """
    Instances of this class are used when working with a torrent def that contains no metainfo (yet), for instance,
    when starting a download with only an infohash. Other methods that are using this class do not distinguish between
    a TorrentDef with and without data and may still expect this class to have various methods in TorrentDef
    implemented.
    """

    def __init__(self, infohash, name, url=None):
        if False:
            i = 10
            return i + 15
        assert isinstance(infohash, bytes), f'INFOHASH has invalid type: {type(infohash)}'
        assert len(infohash) == INFOHASH_LENGTH, 'INFOHASH has invalid length: %d' % len(infohash)
        self.infohash = infohash
        self.name = name
        self.url = url

    def get_name(self):
        if False:
            return 10
        return self.name

    def get_infohash(self):
        if False:
            return 10
        return self.infohash

    def get_length(self, selectedfiles=None):
        if False:
            return 10
        return 0

    def get_metainfo(self):
        if False:
            for i in range(10):
                print('nop')
        return None

    def get_url(self):
        if False:
            return 10
        return self.url

    def is_multifile_torrent(self):
        if False:
            while True:
                i = 10
        return False

    def get_name_utf8(self):
        if False:
            print('Hello World!')
        '\n        Not all names are utf-8, attempt to construct it as utf-8 anyway.\n        '
        return escape_as_utf8(self.name.encode('utf-8 ') if isinstance(self.name, str) else self.name)

    def get_name_as_unicode(self):
        if False:
            while True:
                i = 10
        return ensure_unicode(self.name, 'utf-8')

    def get_files(self, exts=None):
        if False:
            print('Hello World!')
        return []

    def get_files_with_length(self, exts=None):
        if False:
            for i in range(10):
                print('nop')
        return []

    def get_trackers(self) -> set:
        if False:
            for i in range(10):
                print('nop')
        '\n        Returns a flat set of all known trackers.\n\n        :return: all known trackers\n        :rtype: set\n        '
        if self.url and self.url.startswith('magnet:'):
            trackers = parse_magnetlink(self.url)[2]
            return set(trackers)
        return set()

    def is_private(self):
        if False:
            while True:
                i = 10
        return False

    def get_nr_pieces(self):
        if False:
            return 10
        return 0