from __future__ import absolute_import
from os.path import isfile
import re
import mutagen.apev2
import mutagen.monkeysaudio
import mutagen.musepack
import mutagen.optimfrog
import mutagen.wavpack
from picard import log
from picard.config import get_config
from picard.coverart.image import CoverArtImageError, TagCoverArtImage
from picard.file import File
from picard.metadata import Metadata
from picard.util import encode_filename, sanitize_date
from picard.util.filenaming import get_available_filename, move_ensure_casing, replace_extension
from .mutagenext import aac, tak
INVALID_CHARS = re.compile('[^ -~]')
DISALLOWED_KEYS = {'ID3', 'TAG', 'OggS', 'MP+'}
UNSUPPORTED_TAGS = {'gapless', 'musicip_fingerprint', 'podcast', 'podcasturl', 'show', 'showsort', 'r128_album_gain', 'r128_track_gain'}

def is_valid_key(key):
    if False:
        i = 10
        return i + 15
    '\n    Return true if a string is a valid APE tag key.\n    APE tag item keys can have a length of 2 (including) up to 255 (including)\n    characters in the range from 0x20 (Space) until 0x7E (Tilde).\n    Not allowed are the following keys: ID3, TAG, OggS and MP+.\n\n    See http://wiki.hydrogenaud.io/index.php?title=APE_key\n    '
    return key and 2 <= len(key) <= 255 and (key not in DISALLOWED_KEYS) and (INVALID_CHARS.search(key) is None)

class APEv2File(File):
    """Generic APEv2-based file."""
    _File = None
    __translate = {'albumartist': 'Album Artist', 'remixer': 'MixArtist', 'director': 'Director', 'website': 'Weblink', 'discsubtitle': 'DiscSubtitle', 'bpm': 'BPM', 'isrc': 'ISRC', 'catalognumber': 'CatalogNumber', 'barcode': 'Barcode', 'encodedby': 'EncodedBy', 'language': 'Language', 'movementnumber': 'MOVEMENT', 'movement': 'MOVEMENTNAME', 'movementtotal': 'MOVEMENTTOTAL', 'showmovement': 'SHOWMOVEMENT', 'releasestatus': 'MUSICBRAINZ_ALBUMSTATUS', 'releasetype': 'MUSICBRAINZ_ALBUMTYPE', 'musicbrainz_recordingid': 'musicbrainz_trackid', 'musicbrainz_trackid': 'musicbrainz_releasetrackid', 'originalartist': 'Original Artist', 'replaygain_album_gain': 'REPLAYGAIN_ALBUM_GAIN', 'replaygain_album_peak': 'REPLAYGAIN_ALBUM_PEAK', 'replaygain_album_range': 'REPLAYGAIN_ALBUM_RANGE', 'replaygain_track_gain': 'REPLAYGAIN_TRACK_GAIN', 'replaygain_track_peak': 'REPLAYGAIN_TRACK_PEAK', 'replaygain_track_range': 'REPLAYGAIN_TRACK_RANGE', 'replaygain_reference_loudness': 'REPLAYGAIN_REFERENCE_LOUDNESS'}
    __rtranslate = {v.lower(): k for (k, v) in __translate.items()}

    def __init__(self, filename):
        if False:
            while True:
                i = 10
        super().__init__(filename)
        self.__casemap = {}

    def _load(self, filename):
        if False:
            return 10
        log.debug('Loading file %r', filename)
        self.__casemap = {}
        file = self._File(encode_filename(filename))
        metadata = Metadata()
        if file.tags:
            for (origname, values) in file.tags.items():
                name_lower = origname.lower()
                if values.kind == mutagen.apev2.BINARY and name_lower.startswith('cover art'):
                    if b'\x00' in values.value:
                        (descr, data) = values.value.split(b'\x00', 1)
                        try:
                            coverartimage = TagCoverArtImage(file=filename, tag=name_lower, data=data)
                        except CoverArtImageError as e:
                            log.error('Cannot load image from %r: %s', filename, e)
                        else:
                            metadata.images.append(coverartimage)
                if values.kind != mutagen.apev2.TEXT:
                    continue
                for value in values:
                    name = name_lower
                    if name == 'year':
                        name = 'date'
                        value = sanitize_date(value)
                    elif name == 'track':
                        name = 'tracknumber'
                        track = value.split('/')
                        if len(track) > 1:
                            metadata['totaltracks'] = track[1]
                            value = track[0]
                    elif name == 'disc':
                        name = 'discnumber'
                        disc = value.split('/')
                        if len(disc) > 1:
                            metadata['totaldiscs'] = disc[1]
                            value = disc[0]
                    elif name in {'performer', 'comment'}:
                        if value.endswith(')'):
                            start = value.rfind(' (')
                            if start > 0:
                                name += ':' + value[start + 2:-1]
                                value = value[:start]
                    elif name in self.__rtranslate:
                        name = self.__rtranslate[name]
                    self.__casemap[name] = origname
                    metadata.add(name, value)
        self._info(metadata, file)
        return metadata

    def _save(self, filename, metadata):
        if False:
            return 10
        'Save metadata to the file.'
        log.debug('Saving file %r', filename)
        config = get_config()
        try:
            tags = mutagen.apev2.APEv2(encode_filename(filename))
        except mutagen.apev2.APENoHeaderError:
            tags = mutagen.apev2.APEv2()
        images_to_save = list(metadata.images.to_be_saved_to_tags())
        if config.setting['clear_existing_tags']:
            preserved = []
            if config.setting['preserve_images']:
                preserved = list(self._iter_cover_art_tags(tags))
            tags.clear()
            for (name, value) in preserved:
                tags[name] = value
        elif images_to_save:
            for (name, value) in self._iter_cover_art_tags(tags):
                del tags[name]
        temp = {}
        for (name, value) in metadata.items():
            if name.startswith('~') or not self.supports_tag(name):
                continue
            real_name = self._get_tag_name(name)
            if name == 'tracknumber':
                if 'totaltracks' in metadata:
                    value = '%s/%s' % (value, metadata['totaltracks'])
            elif name == 'discnumber':
                if 'totaldiscs' in metadata:
                    value = '%s/%s' % (value, metadata['totaldiscs'])
            elif name in {'totaltracks', 'totaldiscs'}:
                continue
            elif name.startswith('performer:') or name.startswith('comment:'):
                (name, desc) = name.split(':', 1)
                if desc:
                    value += ' (%s)' % desc
            temp.setdefault(real_name, []).append(value)
        for (name, values) in temp.items():
            tags[name] = values
        for image in images_to_save:
            cover_filename = 'Cover Art (Front)'
            cover_filename += image.extension
            tags['Cover Art (Front)'] = mutagen.apev2.APEValue(cover_filename.encode('ascii') + b'\x00' + image.data, mutagen.apev2.BINARY)
            break
        self._remove_deleted_tags(metadata, tags)
        tags.save(encode_filename(filename))

    def _remove_deleted_tags(self, metadata, tags):
        if False:
            return 10
        'Remove the tags from the file that were deleted in the UI'
        for tag in metadata.deleted_tags:
            real_name = self._get_tag_name(tag)
            if real_name in {'Lyrics', 'Comment', 'Performer'}:
                parts = tag.split(':', 1)
                if len(parts) == 2:
                    tag_type_regex = re.compile('\\(%s\\)$' % re.escape(parts[1]))
                else:
                    tag_type_regex = re.compile('[^)]$')
                existing_tags = tags.get(real_name, [])
                for item in existing_tags:
                    if re.search(tag_type_regex, item):
                        existing_tags.remove(item)
                tags[real_name] = existing_tags
            elif tag in {'totaltracks', 'totaldiscs'}:
                tagstr = real_name.lower() + 'number'
                if tagstr in metadata:
                    tags[real_name] = metadata[tagstr]
            elif real_name in tags:
                del tags[real_name]

    def _get_tag_name(self, name):
        if False:
            while True:
                i = 10
        if name in self.__casemap:
            return self.__casemap[name]
        elif name.startswith('lyrics:'):
            return 'Lyrics'
        elif name == 'date':
            return 'Year'
        elif name in {'tracknumber', 'totaltracks'}:
            return 'Track'
        elif name in {'discnumber', 'totaldiscs'}:
            return 'Disc'
        elif name.startswith('performer:') or name.startswith('comment:'):
            return name.split(':', 1)[0].title()
        elif name in self.__translate:
            return self.__translate[name]
        else:
            return name.title()

    @staticmethod
    def _iter_cover_art_tags(tags):
        if False:
            print('Hello World!')
        for (name, value) in tags.items():
            if value.kind == mutagen.apev2.BINARY and name.lower().startswith('cover art'):
                yield (name, value)

    @classmethod
    def supports_tag(cls, name):
        if False:
            return 10
        return bool(name) and name not in UNSUPPORTED_TAGS and (not name.startswith('~')) and (is_valid_key(name) or name.startswith('comment:') or name.startswith('lyrics:') or name.startswith('performer:'))

class MusepackFile(APEv2File):
    """Musepack file."""
    EXTENSIONS = ['.mpc', '.mp+']
    NAME = 'Musepack'
    _File = mutagen.musepack.Musepack

    def _info(self, metadata, file):
        if False:
            for i in range(10):
                print('nop')
        super()._info(metadata, file)
        metadata['~format'] = 'Musepack, SV%d' % file.info.version

class WavPackFile(APEv2File):
    """WavPack file."""
    EXTENSIONS = ['.wv']
    NAME = 'WavPack'
    _File = mutagen.wavpack.WavPack

    def _move_or_rename_wvc(self, old_filename, new_filename):
        if False:
            for i in range(10):
                print('nop')
        wvc_filename = replace_extension(old_filename, '.wvc')
        if not isfile(wvc_filename):
            return
        wvc_new_filename = replace_extension(new_filename, '.wvc')
        wvc_new_filename = get_available_filename(wvc_new_filename, wvc_filename)
        log.debug('Moving Wavepack correction file %r => %r', wvc_filename, wvc_new_filename)
        move_ensure_casing(wvc_filename, wvc_new_filename)

    def _move_additional_files(self, old_filename, new_filename, config):
        if False:
            while True:
                i = 10
        'Includes an additional check for WavPack correction files'
        if config.setting['rename_files'] or config.setting['move_files']:
            self._move_or_rename_wvc(old_filename, new_filename)
        return super()._move_additional_files(old_filename, new_filename, config)

class OptimFROGFile(APEv2File):
    """OptimFROG file."""
    EXTENSIONS = ['.ofr', '.ofs']
    NAME = 'OptimFROG'
    _File = mutagen.optimfrog.OptimFROG

    def _info(self, metadata, file):
        if False:
            while True:
                i = 10
        super()._info(metadata, file)
        filename = file.filename
        if isinstance(filename, bytes):
            filename = filename.decode()
        if filename.lower().endswith('.ofs'):
            metadata['~format'] = 'OptimFROG DualStream Audio'
        else:
            metadata['~format'] = 'OptimFROG Lossless Audio'

class MonkeysAudioFile(APEv2File):
    """Monkey's Audio file."""
    EXTENSIONS = ['.ape']
    NAME = "Monkey's Audio"
    _File = mutagen.monkeysaudio.MonkeysAudio

class TAKFile(APEv2File):
    """TAK file."""
    EXTENSIONS = ['.tak']
    NAME = "Tom's lossless Audio Kompressor"
    _File = tak.TAK

class AACFile(APEv2File):
    EXTENSIONS = ['.aac']
    NAME = 'AAC'
    _File = aac.AACAPEv2

    def _info(self, metadata, file):
        if False:
            print('Hello World!')
        super()._info(metadata, file)
        if file.tags:
            metadata['~format'] = '%s (APEv2)' % self.NAME

    def _save(self, filename, metadata):
        if False:
            print('Hello World!')
        config = get_config()
        if config.setting['aac_save_ape']:
            super()._save(filename, metadata)
        elif config.setting['remove_ape_from_aac']:
            try:
                mutagen.apev2.delete(encode_filename(filename))
            except BaseException:
                log.exception('Error removing APEv2 tags from %s', filename)

    @classmethod
    def supports_tag(cls, name):
        if False:
            return 10
        config = get_config()
        if config.setting['aac_save_ape']:
            return APEv2File.supports_tag(name)
        else:
            return False