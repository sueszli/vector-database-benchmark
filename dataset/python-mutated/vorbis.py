import base64
import re
import mutagen.flac
import mutagen.ogg
import mutagen.oggflac
import mutagen.oggopus
import mutagen.oggspeex
import mutagen.oggtheora
import mutagen.oggvorbis
from picard import log
from picard.config import get_config
from picard.coverart.image import CoverArtImageError, TagCoverArtImage
from picard.coverart.utils import types_from_id3
from picard.file import File
from picard.formats.util import guess_format
from picard.metadata import Metadata
from picard.util import encode_filename, sanitize_date
FLAC_MAX_BLOCK_SIZE = 2 ** 24 - 1
INVALID_CHARS = re.compile('([^ -}]|=)')
UNSUPPORTED_TAGS = {'r128_album_gain', 'r128_track_gain'}

def sanitize_key(key):
    if False:
        return 10
    '\n    Remove characters from key which are invalid for a Vorbis comment field name.\n    See https://www.xiph.org/vorbis/doc/v-comment.html#vectorformat\n    '
    return INVALID_CHARS.sub('', key)

def is_valid_key(key):
    if False:
        while True:
            i = 10
    "\n    Return true if a string is a valid Vorbis comment key.\n    Valid characters for Vorbis comment field names are\n    ASCII 0x20 through 0x7D, 0x3D ('=') excluded.\n    "
    return key and INVALID_CHARS.search(key) is None

def flac_sort_pics_after_tags(metadata_blocks):
    if False:
        for i in range(10):
            print('nop')
    '\n    Reorder the metadata_blocks so that all picture blocks are located after\n    the first Vorbis comment block.\n\n    Windows fails to read FLAC tags if the picture blocks are located before\n    the Vorbis comments. Reordering the blocks fixes this.\n    '
    tagindex = 0
    picblocks = []
    for block in metadata_blocks:
        if block.code == mutagen.flac.VCFLACDict.code:
            tagindex = metadata_blocks.index(block)
            break
        elif block.code == mutagen.flac.Picture.code:
            picblocks.append(block)
    else:
        return
    for pic in picblocks:
        metadata_blocks.remove(pic)
        metadata_blocks.insert(tagindex, pic)

def flac_remove_empty_seektable(file):
    if False:
        for i in range(10):
            print('nop')
    'Removes an existing but empty seektable from the Flac file.\n\n    Some software has issues with files that contain an empty seek table. Since\n    no seektable is also valid, remove it.\n    '
    seektable = file.seektable
    if seektable and (not seektable.seekpoints):
        file.metadata_blocks = [b for b in file.metadata_blocks if b != file.seektable]
        file.seektable = None

class VCommentFile(File):
    """Generic VComment-based file."""
    _File = None
    __translate = {'movement': 'movementnumber', 'movementname': 'movement', 'musicbrainz_releasetrackid': 'musicbrainz_trackid', 'musicbrainz_trackid': 'musicbrainz_recordingid', 'waveformatextensible_channel_mask': '~waveformatextensible_channel_mask'}
    __rtranslate = {v: k for (k, v) in __translate.items()}

    def _load(self, filename):
        if False:
            while True:
                i = 10
        log.debug('Loading file %r', filename)
        config = get_config()
        file = self._File(encode_filename(filename))
        file.tags = file.tags or {}
        metadata = Metadata()
        for (origname, values) in file.tags.items():
            for value in values:
                value = value.rstrip('\x00')
                name = origname
                if name in {'date', 'originaldate', 'releasedate'}:
                    value = sanitize_date(value)
                elif name == 'performer' or name == 'comment':
                    name += ':'
                    if value.endswith(')'):
                        start = len(value) - 2
                        count = 1
                        while count > 0 and start > 0:
                            if value[start] == ')':
                                count += 1
                            elif value[start] == '(':
                                count -= 1
                            start -= 1
                        if start > 0:
                            name += value[start + 2:-1]
                            value = value[:start]
                elif name.startswith('rating'):
                    try:
                        (name, email) = name.split(':', 1)
                    except ValueError:
                        email = ''
                    if email != sanitize_key(config.setting['rating_user_email']):
                        continue
                    name = '~rating'
                    try:
                        value = str(round(float(value) * (config.setting['rating_steps'] - 1)))
                    except ValueError:
                        log.warning('Invalid rating value in %r: %s', filename, value)
                elif name == 'fingerprint' and value.startswith('MusicMagic Fingerprint'):
                    name = 'musicip_fingerprint'
                    value = value[22:]
                elif name == 'tracktotal':
                    if 'totaltracks' in file.tags:
                        continue
                    name = 'totaltracks'
                elif name == 'disctotal':
                    if 'totaldiscs' in file.tags:
                        continue
                    name = 'totaldiscs'
                elif name == 'metadata_block_picture':
                    try:
                        image = mutagen.flac.Picture(base64.standard_b64decode(value))
                        coverartimage = TagCoverArtImage(file=filename, tag=name, types=types_from_id3(image.type), comment=image.desc, support_types=True, data=image.data, id3_type=image.type)
                    except (CoverArtImageError, TypeError, ValueError, mutagen.flac.error) as e:
                        log.error('Cannot load image from %r: %s', filename, e)
                    else:
                        metadata.images.append(coverartimage)
                    continue
                elif name in self.__translate:
                    name = self.__translate[name]
                metadata.add(name, value)
        if self._File == mutagen.flac.FLAC:
            for image in file.pictures:
                try:
                    coverartimage = TagCoverArtImage(file=filename, tag='FLAC/PICTURE', types=types_from_id3(image.type), comment=image.desc, support_types=True, data=image.data, id3_type=image.type)
                except CoverArtImageError as e:
                    log.error('Cannot load image from %r: %s', filename, e)
                else:
                    metadata.images.append(coverartimage)
        if 'metadata_block_picture' not in file.tags:
            try:
                for data in file['COVERART']:
                    try:
                        coverartimage = TagCoverArtImage(file=filename, tag='COVERART', data=base64.standard_b64decode(data))
                    except (CoverArtImageError, TypeError, ValueError) as e:
                        log.error('Cannot load image from %r: %s', filename, e)
                    else:
                        metadata.images.append(coverartimage)
            except KeyError:
                pass
        self._info(metadata, file)
        return metadata

    def _save(self, filename, metadata):
        if False:
            while True:
                i = 10
        'Save metadata to the file.'
        log.debug('Saving file %r', filename)
        config = get_config()
        is_flac = self._File == mutagen.flac.FLAC
        file = self._File(encode_filename(filename))
        if file.tags is None:
            file.add_tags()
        if config.setting['clear_existing_tags']:
            preserve_tags = ['waveformatextensible_channel_mask']
            if not is_flac and config.setting['preserve_images']:
                preserve_tags.append('metadata_block_picture')
                preserve_tags.append('coverart')
            preserved_values = {}
            for name in preserve_tags:
                if name in file.tags and file.tags[name]:
                    preserved_values[name] = file.tags[name]
            file.tags.clear()
            for (name, value) in preserved_values.items():
                file.tags[name] = value
        images_to_save = list(metadata.images.to_be_saved_to_tags())
        if is_flac and (images_to_save or (config.setting['clear_existing_tags'] and (not config.setting['preserve_images']))):
            file.clear_pictures()
        tags = {}
        for (name, value) in metadata.items():
            if name == '~rating':
                user_email = sanitize_key(config.setting['rating_user_email'])
                if user_email:
                    name = 'rating:%s' % user_email
                else:
                    name = 'rating'
                value = str(float(value) / (config.setting['rating_steps'] - 1))
            elif name.startswith('~') or not self.supports_tag(name):
                continue
            elif name.startswith('lyrics:'):
                name = 'lyrics'
            elif name in {'date', 'originaldate', 'releasedate'}:
                value = sanitize_date(value)
            elif name.startswith('performer:') or name.startswith('comment:'):
                (name, desc) = name.split(':', 1)
                if desc:
                    value += ' (%s)' % desc
            elif name == 'musicip_fingerprint':
                name = 'fingerprint'
                value = 'MusicMagic Fingerprint%s' % value
            elif name in self.__rtranslate:
                name = self.__rtranslate[name]
            tags.setdefault(name.upper(), []).append(value.rstrip('\x00'))
        if 'totaltracks' in metadata:
            tags.setdefault('TRACKTOTAL', []).append(metadata['totaltracks'])
        if 'totaldiscs' in metadata:
            tags.setdefault('DISCTOTAL', []).append(metadata['totaldiscs'])
        for image in images_to_save:
            picture = mutagen.flac.Picture()
            picture.data = image.data
            picture.mime = image.mimetype
            picture.desc = image.comment
            picture.width = image.width
            picture.height = image.height
            picture.type = image.id3_type
            if is_flac:
                expected_block_size = 8 * 4 + len(picture.data) + len(picture.mime) + len(picture.desc.encode('UTF-8'))
                if expected_block_size > FLAC_MAX_BLOCK_SIZE:
                    log.error('Failed saving image to %r: Image size of %d bytes exceeds maximum FLAC block size of %d bytes', filename, expected_block_size, FLAC_MAX_BLOCK_SIZE)
                    continue
                file.add_picture(picture)
            else:
                tags.setdefault('METADATA_BLOCK_PICTURE', []).append(base64.b64encode(picture.write()).decode('ascii'))
        file.tags.update(tags)
        self._remove_deleted_tags(metadata, file.tags)
        kwargs = {}
        if is_flac:
            flac_sort_pics_after_tags(file.metadata_blocks)
            if config.setting['fix_missing_seekpoints_flac']:
                flac_remove_empty_seektable(file)
            if config.setting['remove_id3_from_flac']:
                kwargs['deleteid3'] = True
        try:
            file.save(**kwargs)
        except TypeError:
            file.save()

    def _remove_deleted_tags(self, metadata, tags):
        if False:
            for i in range(10):
                print('nop')
        'Remove the tags from the file that were deleted in the UI'
        for tag in metadata.deleted_tags:
            real_name = self._get_tag_name(tag)
            if is_valid_key(real_name) and real_name in tags:
                if real_name in {'performer', 'comment'}:
                    parts = tag.split(':', 1)
                    if len(parts) == 2:
                        tag_type_regex = re.compile('\\(%s\\)$' % re.escape(parts[1]))
                    else:
                        tag_type_regex = re.compile('[^)]$')
                    existing_tags = tags.get(real_name)
                    for item in existing_tags:
                        if re.search(tag_type_regex, item):
                            existing_tags.remove(item)
                    tags[real_name] = existing_tags
                else:
                    if tag in {'totaldiscs', 'totaltracks'} and tag in tags:
                        del tags[tag]
                    del tags[real_name]

    def _get_tag_name(self, name):
        if False:
            print('Hello World!')
        if name == '~rating':
            config = get_config()
            if config.setting['rating_user_email']:
                return 'rating:%s' % config.setting['rating_user_email']
            else:
                return 'rating'
        elif name.startswith('~'):
            return None
        elif name.startswith('lyrics:'):
            return 'lyrics'
        elif name.startswith('performer:') or name.startswith('comment:'):
            return name.split(':', 1)[0]
        elif name == 'musicip_fingerprint':
            return 'fingerprint'
        elif name == 'totaltracks':
            return 'tracktotal'
        elif name == 'totaldiscs':
            return 'disctotal'
        elif name in self.__rtranslate:
            return self.__rtranslate[name]
        else:
            return name

    @classmethod
    def supports_tag(cls, name):
        if False:
            while True:
                i = 10
        return bool(name) and name not in UNSUPPORTED_TAGS and (is_valid_key(name) or name.startswith('comment:') or name.startswith('lyrics:') or name.startswith('performer:'))

class FLACFile(VCommentFile):
    """FLAC file."""
    EXTENSIONS = ['.flac']
    NAME = 'FLAC'
    _File = mutagen.flac.FLAC

class OggFLACFile(VCommentFile):
    """FLAC file."""
    EXTENSIONS = ['.oggflac']
    NAME = 'Ogg FLAC'
    _File = mutagen.oggflac.OggFLAC

class OggSpeexFile(VCommentFile):
    """Ogg Speex file."""
    EXTENSIONS = ['.spx']
    NAME = 'Speex'
    _File = mutagen.oggspeex.OggSpeex

class OggTheoraFile(VCommentFile):
    """Ogg Theora file."""
    EXTENSIONS = ['.oggtheora']
    NAME = 'Ogg Theora'
    _File = mutagen.oggtheora.OggTheora

    def _info(self, metadata, file):
        if False:
            i = 10
            return i + 15
        super()._info(metadata, file)
        metadata['~video'] = '1'

class OggVorbisFile(VCommentFile):
    """Ogg Vorbis file."""
    EXTENSIONS = []
    NAME = 'Ogg Vorbis'
    _File = mutagen.oggvorbis.OggVorbis

class OggOpusFile(VCommentFile):
    """Ogg Opus file."""
    EXTENSIONS = ['.opus']
    NAME = 'Ogg Opus'
    _File = mutagen.oggopus.OggOpus

    @classmethod
    def supports_tag(cls, name):
        if False:
            return 10
        if name.startswith('r128_'):
            return True
        return VCommentFile.supports_tag(name)

def OggAudioFile(filename):
    if False:
        for i in range(10):
            print('nop')
    'Generic Ogg audio file.'
    options = [OggFLACFile, OggOpusFile, OggSpeexFile, OggVorbisFile]
    return guess_format(filename, options)
OggAudioFile.EXTENSIONS = ['.oga']
OggAudioFile.NAME = 'Ogg Audio'
OggAudioFile.supports_tag = VCommentFile.supports_tag

def OggVideoFile(filename):
    if False:
        for i in range(10):
            print('nop')
    'Generic Ogg video file.'
    options = [OggTheoraFile]
    return guess_format(filename, options)
OggVideoFile.EXTENSIONS = ['.ogv']
OggVideoFile.NAME = 'Ogg Video'
OggVideoFile.supports_tag = VCommentFile.supports_tag

def OggContainerFile(filename):
    if False:
        for i in range(10):
            print('nop')
    'Generic Ogg file.'
    options = [OggFLACFile, OggOpusFile, OggSpeexFile, OggTheoraFile, OggVorbisFile]
    return guess_format(filename, options)
OggContainerFile.EXTENSIONS = ['.ogg']
OggContainerFile.NAME = 'Ogg'
OggContainerFile.supports_tag = VCommentFile.supports_tag