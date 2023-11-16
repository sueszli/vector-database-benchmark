"""Synchronize information from iTunes's library
"""
import os
import plistlib
import shutil
import tempfile
from contextlib import contextmanager
from time import mktime
from urllib.parse import unquote, urlparse
from confuse import ConfigValueError
from beets import util
from beets.dbcore import types
from beets.library import DateType
from beets.util import bytestring_path, syspath
from beetsplug.metasync import MetaSource

@contextmanager
def create_temporary_copy(path):
    if False:
        i = 10
        return i + 15
    temp_dir = bytestring_path(tempfile.mkdtemp())
    temp_path = os.path.join(temp_dir, b'temp_itunes_lib')
    shutil.copyfile(syspath(path), syspath(temp_path))
    try:
        yield temp_path
    finally:
        shutil.rmtree(syspath(temp_dir))

def _norm_itunes_path(path):
    if False:
        for i in range(10):
            print('nop')
    return util.bytestring_path(os.path.normpath(unquote(urlparse(path).path)).lstrip('\\')).lower()

class Itunes(MetaSource):
    item_types = {'itunes_rating': types.INTEGER, 'itunes_playcount': types.INTEGER, 'itunes_skipcount': types.INTEGER, 'itunes_lastplayed': DateType(), 'itunes_lastskipped': DateType(), 'itunes_dateadded': DateType()}

    def __init__(self, config, log):
        if False:
            for i in range(10):
                print('nop')
        super().__init__(config, log)
        config.add({'itunes': {'library': '~/Music/iTunes/iTunes Library.xml'}})
        library_path = config['itunes']['library'].as_filename()
        try:
            self._log.debug(f'loading iTunes library from {library_path}')
            with create_temporary_copy(library_path) as library_copy:
                with open(library_copy, 'rb') as library_copy_f:
                    raw_library = plistlib.load(library_copy_f)
        except OSError as e:
            raise ConfigValueError('invalid iTunes library: ' + e.strerror)
        except Exception:
            if os.path.splitext(library_path)[1].lower() != '.xml':
                hint = ': please ensure that the configured path points to the .XML library'
            else:
                hint = ''
            raise ConfigValueError('invalid iTunes library' + hint)
        self.collection = {_norm_itunes_path(track['Location']): track for track in raw_library['Tracks'].values() if 'Location' in track}

    def sync_from_source(self, item):
        if False:
            return 10
        result = self.collection.get(util.bytestring_path(item.path).lower())
        if not result:
            self._log.warning(f'no iTunes match found for {item}')
            return
        item.itunes_rating = result.get('Rating')
        item.itunes_playcount = result.get('Play Count')
        item.itunes_skipcount = result.get('Skip Count')
        if result.get('Play Date UTC'):
            item.itunes_lastplayed = mktime(result.get('Play Date UTC').timetuple())
        if result.get('Skip Date'):
            item.itunes_lastskipped = mktime(result.get('Skip Date').timetuple())
        if result.get('Date Added'):
            item.itunes_dateadded = mktime(result.get('Date Added').timetuple())