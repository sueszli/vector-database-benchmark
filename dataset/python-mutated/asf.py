import struct
from mutagen.asf import ASF, ASFByteArrayAttribute
from picard import log
from picard.config import get_config
from picard.coverart.image import CoverArtImageError, TagCoverArtImage
from picard.coverart.utils import types_from_id3
from picard.file import File
from picard.formats.mutagenext import delall_ci
from picard.metadata import Metadata
from picard.util import encode_filename

def unpack_image(data):
    if False:
        while True:
            i = 10
    '\n    Helper function to unpack image data from a WM/Picture tag.\n\n    The data has the following format:\n    1 byte: Picture type (0-20), see ID3 APIC frame specification at\n            http://www.id3.org/id3v2.4.0-frames\n    4 bytes: Picture data length in LE format\n    MIME type, null terminated UTF-16-LE string\n    Description, null terminated UTF-16-LE string\n    The image data in the given length\n    '
    try:
        (type_, size) = struct.unpack_from('<bi', data)
    except struct.error as e:
        raise ValueError(e)
    data = data[5:]
    mime = b''
    while data:
        (char, data) = (data[:2], data[2:])
        if char == b'\x00\x00':
            break
        mime += char
    else:
        raise ValueError('mime: missing data')
    mime = mime.decode('utf-16-le')
    description = b''
    while data:
        (char, data) = (data[:2], data[2:])
        if char == b'\x00\x00':
            break
        description += char
    else:
        raise ValueError('desc: missing data')
    description = description.decode('utf-16-le')
    if size != len(data):
        raise ValueError('image data size mismatch')
    return (mime, data, type_, description)

def pack_image(mime, data, image_type=3, description=''):
    if False:
        i = 10
        return i + 15
    '\n    Helper function to pack image data for a WM/Picture tag.\n    See unpack_image for a description of the data format.\n    '
    tag_data = struct.pack('<bi', image_type, len(data))
    tag_data += mime.encode('utf-16-le') + b'\x00\x00'
    tag_data += description.encode('utf-16-le') + b'\x00\x00'
    tag_data += data
    return tag_data

class ASFFile(File):
    """
    ASF (WMA) metadata reader/writer
    See http://msdn.microsoft.com/en-us/library/ms867702.aspx for official
    WMA tag specifications.
    """
    EXTENSIONS = ['.wma', '.wmv', '.asf']
    NAME = 'Windows Media Audio'
    _File = ASF
    __TRANS = {'album': 'WM/AlbumTitle', 'title': 'Title', 'artist': 'Author', 'albumartist': 'WM/AlbumArtist', 'date': 'WM/Year', 'originalalbum': 'WM/OriginalAlbumTitle', 'originalartist': 'WM/OriginalArtist', 'originaldate': 'WM/OriginalReleaseTime', 'originalyear': 'WM/OriginalReleaseYear', 'originalfilename': 'WM/OriginalFilename', 'composer': 'WM/Composer', 'lyricist': 'WM/Writer', 'conductor': 'WM/Conductor', 'remixer': 'WM/ModifiedBy', 'producer': 'WM/Producer', 'grouping': 'WM/ContentGroupDescription', 'subtitle': 'WM/SubTitle', 'discsubtitle': 'WM/SetSubTitle', 'tracknumber': 'WM/TrackNumber', 'discnumber': 'WM/PartOfSet', 'comment': 'Description', 'genre': 'WM/Genre', 'bpm': 'WM/BeatsPerMinute', 'key': 'WM/InitialKey', 'script': 'WM/Script', 'language': 'WM/Language', 'mood': 'WM/Mood', 'isrc': 'WM/ISRC', 'copyright': 'Copyright', 'lyrics': 'WM/Lyrics', '~rating': 'WM/SharedUserRating', 'media': 'WM/Media', 'barcode': 'WM/Barcode', 'catalognumber': 'WM/CatalogNo', 'label': 'WM/Publisher', 'encodedby': 'WM/EncodedBy', 'encodersettings': 'WM/EncodingSettings', 'albumsort': 'WM/AlbumSortOrder', 'albumartistsort': 'WM/AlbumArtistSortOrder', 'artistsort': 'WM/ArtistSortOrder', 'titlesort': 'WM/TitleSortOrder', 'composersort': 'WM/ComposerSortOrder', 'musicbrainz_recordingid': 'MusicBrainz/Track Id', 'musicbrainz_trackid': 'MusicBrainz/Release Track Id', 'musicbrainz_albumid': 'MusicBrainz/Album Id', 'musicbrainz_artistid': 'MusicBrainz/Artist Id', 'musicbrainz_albumartistid': 'MusicBrainz/Album Artist Id', 'musicbrainz_trmid': 'MusicBrainz/TRM Id', 'musicbrainz_discid': 'MusicBrainz/Disc Id', 'musicbrainz_workid': 'MusicBrainz/Work Id', 'musicbrainz_releasegroupid': 'MusicBrainz/Release Group Id', 'musicbrainz_originalalbumid': 'MusicBrainz/Original Album Id', 'musicbrainz_originalartistid': 'MusicBrainz/Original Artist Id', 'musicip_puid': 'MusicIP/PUID', 'releasestatus': 'MusicBrainz/Album Status', 'releasetype': 'MusicBrainz/Album Type', 'releasecountry': 'MusicBrainz/Album Release Country', 'acoustid_id': 'Acoustid/Id', 'acoustid_fingerprint': 'Acoustid/Fingerprint', 'compilation': 'WM/IsCompilation', 'engineer': 'WM/Engineer', 'asin': 'ASIN', 'djmixer': 'WM/DJMixer', 'mixer': 'WM/Mixer', 'artists': 'WM/ARTISTS', 'director': 'WM/Director', 'work': 'WM/Work', 'website': 'WM/AuthorURL'}
    __RTRANS = {b: a for (a, b) in __TRANS.items()}
    __TRANS_CI = {'replaygain_album_gain': 'REPLAYGAIN_ALBUM_GAIN', 'replaygain_album_peak': 'REPLAYGAIN_ALBUM_PEAK', 'replaygain_album_range': 'REPLAYGAIN_ALBUM_RANGE', 'replaygain_track_gain': 'REPLAYGAIN_TRACK_GAIN', 'replaygain_track_peak': 'REPLAYGAIN_TRACK_PEAK', 'replaygain_track_range': 'REPLAYGAIN_TRACK_RANGE', 'replaygain_reference_loudness': 'REPLAYGAIN_REFERENCE_LOUDNESS'}
    __RTRANS_CI = {b.lower(): a for (a, b) in __TRANS_CI.items()}

    def __init__(self, filename):
        if False:
            for i in range(10):
                print('nop')
        super().__init__(filename)
        self.__casemap = {}

    def _load(self, filename):
        if False:
            i = 10
            return i + 15
        log.debug('Loading file %r', filename)
        config = get_config()
        self.__casemap = {}
        file = ASF(encode_filename(filename))
        metadata = Metadata()
        for (name, values) in file.tags.items():
            if name == 'WM/Picture':
                for image in values:
                    try:
                        (mime, data, image_type, description) = unpack_image(image.value)
                    except ValueError as e:
                        log.warning('Cannot unpack image from %r: %s', filename, e)
                        continue
                    try:
                        coverartimage = TagCoverArtImage(file=filename, tag=name, types=types_from_id3(image_type), comment=description, support_types=True, data=data, id3_type=image_type)
                    except CoverArtImageError as e:
                        log.error('Cannot load image from %r: %s', filename, e)
                    else:
                        metadata.images.append(coverartimage)
                continue
            elif name == 'WM/SharedUserRating':
                values[0] = int(round(int(str(values[0])) / 99.0 * (config.setting['rating_steps'] - 1)))
            elif name == 'WM/PartOfSet':
                disc = str(values[0]).split('/')
                if len(disc) > 1:
                    metadata['totaldiscs'] = disc[1]
                    values[0] = disc[0]
            name_lower = name.lower()
            if name in self.__RTRANS:
                name = self.__RTRANS[name]
            elif name_lower in self.__RTRANS_CI:
                orig_name = name
                name = self.__RTRANS_CI[name_lower]
                self.__casemap[name] = orig_name
            else:
                continue
            values = [str(value) for value in values if value]
            if values:
                metadata[name] = values
        self._info(metadata, file)
        return metadata

    def _save(self, filename, metadata):
        if False:
            return 10
        log.debug('Saving file %r', filename)
        config = get_config()
        file = ASF(encode_filename(filename))
        tags = file.tags
        if config.setting['clear_existing_tags']:
            cover = tags.get('WM/Picture') if config.setting['preserve_images'] else None
            tags.clear()
            if cover:
                tags['WM/Picture'] = cover
        cover = []
        for image in metadata.images.to_be_saved_to_tags():
            tag_data = pack_image(image.mimetype, image.data, image.id3_type, image.comment)
            cover.append(ASFByteArrayAttribute(tag_data))
        if cover:
            tags['WM/Picture'] = cover
        for (name, values) in metadata.rawitems():
            if name.startswith('lyrics:'):
                name = 'lyrics'
            elif name == '~rating':
                values = [int(values[0]) * 99 // (config.setting['rating_steps'] - 1)]
            elif name == 'discnumber' and 'totaldiscs' in metadata:
                values = ['%s/%s' % (metadata['discnumber'], metadata['totaldiscs'])]
            if name in self.__TRANS:
                name = self.__TRANS[name]
            elif name in self.__TRANS_CI:
                if name in self.__casemap:
                    name = self.__casemap[name]
                else:
                    name = self.__TRANS_CI[name]
                delall_ci(tags, name)
            else:
                continue
            tags[name] = values
        self._remove_deleted_tags(metadata, tags)
        file.save()

    def _remove_deleted_tags(self, metadata, tags):
        if False:
            i = 10
            return i + 15
        'Remove the tags from the file that were deleted in the UI'
        for tag in metadata.deleted_tags:
            real_name = self._get_tag_name(tag)
            if real_name and real_name in tags:
                del tags[real_name]

    @classmethod
    def supports_tag(cls, name):
        if False:
            print('Hello World!')
        return name in cls.__TRANS or name in cls.__TRANS_CI or name in {'~rating', 'totaldiscs'} or name.startswith('lyrics:')

    def _get_tag_name(self, name):
        if False:
            return 10
        if name.startswith('lyrics:'):
            name = 'lyrics'
        if name == 'totaldiscs':
            return self.__TRANS['discnumber']
        elif name in self.__TRANS:
            return self.__TRANS[name]
        else:
            return None

    def _info(self, metadata, file):
        if False:
            while True:
                i = 10
        super()._info(metadata, file)
        filename = file.filename
        if isinstance(filename, bytes):
            filename = filename.decode()
        if filename.lower().endswith('.wmv'):
            metadata['~video'] = '1'