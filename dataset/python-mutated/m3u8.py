import logging
import math
import re
from binascii import Error as BinasciiError, unhexlify
from datetime import datetime, timedelta
from typing import Callable, ClassVar, Dict, Generic, Iterator, List, Mapping, Optional, Tuple, Type, TypeVar, Union
from urllib.parse import urljoin, urlparse
from isodate import ISO8601Error, parse_datetime
from requests import Response
from streamlink.logger import ALL, StreamlinkLogger
from streamlink.stream.hls.segment import ByteRange, DateRange, ExtInf, HLSPlaylist, HLSSegment, IFrameStreamInfo, Key, Map, Media, Resolution, Start, StreamInfo
try:
    from typing import Self
except ImportError:
    from typing_extensions import Self
log: StreamlinkLogger = logging.getLogger(__name__)
THLSSegment_co = TypeVar('THLSSegment_co', bound=HLSSegment, covariant=True)
THLSPlaylist_co = TypeVar('THLSPlaylist_co', bound=HLSPlaylist, covariant=True)

class M3U8(Generic[THLSSegment_co, THLSPlaylist_co]):

    def __init__(self, uri: Optional[str]=None):
        if False:
            while True:
                i = 10
        self.uri = uri
        self.is_endlist: bool = False
        self.is_master: bool = False
        self.allow_cache: Optional[bool] = None
        self.discontinuity_sequence: Optional[int] = None
        self.iframes_only: Optional[bool] = None
        self.media_sequence: Optional[int] = None
        self.playlist_type: Optional[str] = None
        self.targetduration: Optional[float] = None
        self.start: Optional[Start] = None
        self.version: Optional[int] = None
        self.media: List[Media] = []
        self.dateranges: List[DateRange] = []
        self.playlists: List[THLSPlaylist_co] = []
        self.segments: List[THLSSegment_co] = []

    @classmethod
    def is_date_in_daterange(cls, date: Optional[datetime], daterange: DateRange):
        if False:
            print('Hello World!')
        if date is None or daterange.start_date is None:
            return None
        if daterange.end_date is not None:
            return daterange.start_date <= date < daterange.end_date
        duration = daterange.duration or daterange.planned_duration
        if duration is not None:
            end = daterange.start_date + duration
            return daterange.start_date <= date < end
        return daterange.start_date <= date
TM3U8_co = TypeVar('TM3U8_co', bound=M3U8, covariant=True)
_symbol_tag_parser = '__PARSE_TAG_NAME'

def parse_tag(tag: str):
    if False:
        while True:
            i = 10

    def decorator(func: Callable[[str], None]) -> Callable[[str], None]:
        if False:
            for i in range(10):
                print('nop')
        setattr(func, _symbol_tag_parser, tag)
        return func
    return decorator

class M3U8ParserMeta(type):

    def __init__(cls, name, bases, namespace, **kwargs):
        if False:
            return 10
        super().__init__(name, bases, namespace, **kwargs)
        tags = dict(**getattr(cls, '_TAGS', {}))
        for member in namespace.values():
            tag = getattr(member, _symbol_tag_parser, None)
            if type(tag) is not str:
                continue
            tags[tag] = member
        cls._TAGS = tags

class M3U8Parser(Generic[TM3U8_co, THLSSegment_co, THLSPlaylist_co], metaclass=M3U8ParserMeta):
    __m3u8__: ClassVar[Type[M3U8[HLSSegment, HLSPlaylist]]] = M3U8
    __segment__: ClassVar[Type[HLSSegment]] = HLSSegment
    __playlist__: ClassVar[Type[HLSPlaylist]] = HLSPlaylist
    _TAGS: ClassVar[Mapping[str, Callable[[Self, str], None]]]
    _extinf_re = re.compile('(?P<duration>\\d+(\\.\\d+)?)(,(?P<title>.+))?')
    _attr_re = re.compile('\n        (?P<key>[A-Z0-9\\-]+)\n        =\n        (?P<value>\n            (?# decimal-integer)\n            \\d+\n            (?# hexadecimal-sequence)\n            |0[xX][0-9A-Fa-f]+\n            (?# decimal-floating-point and signed-decimal-floating-point)\n            |-?\\d+\\.\\d+\n            (?# quoted-string)\n            |\\"(?P<quoted>[^\\r\\n\\"]*)\\"\n            (?# enumerated-string)\n            |[^\\",\\s]+\n            (?# decimal-resolution)\n            |\\d+x\\d+\n        )\n        (?# be more lenient and allow spaces around attributes)\n        \\s*(?:,\\s*|$)\n    ', re.VERBOSE)
    _range_re = re.compile('(?P<range>\\d+)(?:@(?P<offset>\\d+))?')
    _tag_re = re.compile('#(?P<tag>[\\w-]+)(:(?P<value>.+))?')
    _res_re = re.compile('(\\d+)x(\\d+)')

    def __init__(self, base_uri: Optional[str]=None):
        if False:
            for i in range(10):
                print('nop')
        self.m3u8: TM3U8_co = self.__m3u8__(base_uri)
        self._expect_playlist: bool = False
        self._streaminf: Optional[Dict[str, str]] = None
        self._expect_segment: bool = False
        self._extinf: Optional[ExtInf] = None
        self._byterange: Optional[ByteRange] = None
        self._discontinuity: bool = False
        self._map: Optional[Map] = None
        self._key: Optional[Key] = None
        self._date: Optional[datetime] = None

    @classmethod
    def create_stream_info(cls, streaminf: Mapping[str, Optional[str]], streaminfoclass=None):
        if False:
            for i in range(10):
                print('nop')
        program_id = streaminf.get('PROGRAM-ID')
        try:
            bandwidth = int(streaminf.get('BANDWIDTH') or 0)
            bandwidth = round(bandwidth, 1 - int(math.log10(bandwidth)))
        except ValueError:
            bandwidth = 0
        _resolution = streaminf.get('RESOLUTION')
        resolution = None if not _resolution else cls.parse_resolution(_resolution)
        codecs = (streaminf.get('CODECS') or '').split(',')
        if streaminfoclass is IFrameStreamInfo:
            return IFrameStreamInfo(bandwidth=bandwidth, program_id=program_id, codecs=codecs, resolution=resolution, video=streaminf.get('VIDEO'))
        else:
            return StreamInfo(bandwidth=bandwidth, program_id=program_id, codecs=codecs, resolution=resolution, audio=streaminf.get('AUDIO'), video=streaminf.get('VIDEO'), subtitles=streaminf.get('SUBTITLES'))

    @classmethod
    def split_tag(cls, line: str) -> Union[Tuple[str, str], Tuple[None, None]]:
        if False:
            while True:
                i = 10
        match = cls._tag_re.match(line)
        if match:
            return (match.group('tag'), (match.group('value') or '').strip())
        return (None, None)

    @classmethod
    def parse_attributes(cls, value: str) -> Dict[str, str]:
        if False:
            for i in range(10):
                print('nop')
        pos = 0
        length = len(value)
        res: Dict[str, str] = {}
        while pos < length:
            match = cls._attr_re.match(value, pos)
            if match is None:
                log.warning('Discarded invalid attributes list')
                res.clear()
                break
            pos = match.end()
            res[match['key']] = match['quoted'] if match['quoted'] is not None else match['value']
        return res

    @staticmethod
    def parse_bool(value: str) -> bool:
        if False:
            for i in range(10):
                print('nop')
        return value == 'YES'

    @classmethod
    def parse_byterange(cls, value: str) -> Optional[ByteRange]:
        if False:
            return 10
        match = cls._range_re.match(value)
        if match is None:
            return None
        (_range, offset) = match.groups()
        return ByteRange(range=int(_range), offset=int(offset) if offset is not None else None)

    @classmethod
    def parse_extinf(cls, value: str) -> ExtInf:
        if False:
            for i in range(10):
                print('nop')
        match = cls._extinf_re.match(value)
        if match is None:
            return ExtInf(0, None)
        return ExtInf(duration=float(match.group('duration')), title=match.group('title'))

    @staticmethod
    def parse_hex(value: Optional[str]) -> Optional[bytes]:
        if False:
            while True:
                i = 10
        if value is None:
            return None
        if value[:2] in ('0x', '0X'):
            try:
                return unhexlify(f"{'0' * (len(value) % 2)}{value[2:]}")
            except BinasciiError:
                pass
        log.warning('Discarded invalid hexadecimal-sequence attribute value')
        return None

    @staticmethod
    def parse_iso8601(value: Optional[str]) -> Optional[datetime]:
        if False:
            while True:
                i = 10
        try:
            return None if value is None else parse_datetime(value)
        except (ISO8601Error, ValueError):
            log.warning('Discarded invalid ISO8601 attribute value')
            return None

    @staticmethod
    def parse_timedelta(value: Optional[str]) -> Optional[timedelta]:
        if False:
            for i in range(10):
                print('nop')
        return None if value is None else timedelta(seconds=float(value))

    @classmethod
    def parse_resolution(cls, value: str) -> Resolution:
        if False:
            for i in range(10):
                print('nop')
        match = cls._res_re.match(value)
        if match is None:
            return Resolution(width=0, height=0)
        return Resolution(width=int(match.group(1)), height=int(match.group(2)))

    @parse_tag('EXT-X-VERSION')
    def parse_tag_ext_x_version(self, value: str) -> None:
        if False:
            for i in range(10):
                print('nop')
        '\n        EXT-X-VERSION\n        https://datatracker.ietf.org/doc/html/rfc8216#section-4.3.1.2\n        '
        self.m3u8.version = int(value)

    @parse_tag('EXTINF')
    def parse_tag_extinf(self, value: str) -> None:
        if False:
            return 10
        '\n        EXTINF\n        https://datatracker.ietf.org/doc/html/rfc8216#section-4.3.2.1\n        '
        self._expect_segment = True
        self._extinf = self.parse_extinf(value)

    @parse_tag('EXT-X-BYTERANGE')
    def parse_tag_ext_x_byterange(self, value: str) -> None:
        if False:
            return 10
        '\n        EXT-X-BYTERANGE\n        https://datatracker.ietf.org/doc/html/rfc8216#section-4.3.2.2\n        '
        self._expect_segment = True
        self._byterange = self.parse_byterange(value)

    @parse_tag('EXT-X-DISCONTINUITY')
    def parse_tag_ext_x_discontinuity(self, value: str) -> None:
        if False:
            while True:
                i = 10
        '\n        EXT-X-DISCONTINUITY\n        https://datatracker.ietf.org/doc/html/rfc8216#section-4.3.2.3\n        '
        self._discontinuity = True
        self._map = None

    @parse_tag('EXT-X-KEY')
    def parse_tag_ext_x_key(self, value: str) -> None:
        if False:
            return 10
        '\n        EXT-X-KEY\n        https://datatracker.ietf.org/doc/html/rfc8216#section-4.3.2.4\n        '
        attr = self.parse_attributes(value)
        method = attr.get('METHOD')
        uri = attr.get('URI')
        if not method:
            return
        self._key = Key(method=method, uri=self.uri(uri) if uri else None, iv=self.parse_hex(attr.get('IV')), key_format=attr.get('KEYFORMAT'), key_format_versions=attr.get('KEYFORMATVERSIONS'))

    @parse_tag('EXT-X-MAP')
    def parse_tag_ext_x_map(self, value: str) -> None:
        if False:
            while True:
                i = 10
        '\n        EXT-X-MAP\n        https://datatracker.ietf.org/doc/html/rfc8216#section-4.3.2.5\n        '
        attr = self.parse_attributes(value)
        uri = attr.get('URI')
        if not uri:
            return
        byterange = self.parse_byterange(attr.get('BYTERANGE', ''))
        self._map = Map(uri=self.uri(uri), byterange=byterange)

    @parse_tag('EXT-X-PROGRAM-DATE-TIME')
    def parse_tag_ext_x_program_date_time(self, value: str) -> None:
        if False:
            i = 10
            return i + 15
        '\n        EXT-X-PROGRAM-DATE-TIME\n        https://datatracker.ietf.org/doc/html/rfc8216#section-4.3.2.6\n        '
        self._date = self.parse_iso8601(value)

    @parse_tag('EXT-X-DATERANGE')
    def parse_tag_ext_x_daterange(self, value: str) -> None:
        if False:
            print('Hello World!')
        '\n        EXT-X-DATERANGE\n        https://datatracker.ietf.org/doc/html/rfc8216#section-4.3.2.7\n        '
        attr = self.parse_attributes(value)
        daterange = DateRange(id=attr.pop('ID', None), classname=attr.pop('CLASS', None), start_date=self.parse_iso8601(attr.pop('START-DATE', None)), end_date=self.parse_iso8601(attr.pop('END-DATE', None)), duration=self.parse_timedelta(attr.pop('DURATION', None)), planned_duration=self.parse_timedelta(attr.pop('PLANNED-DURATION', None)), end_on_next=self.parse_bool(attr.pop('END-ON-NEXT', 'NO')), x=attr)
        self.m3u8.dateranges.append(daterange)

    @parse_tag('EXT-X-TARGETDURATION')
    def parse_tag_ext_x_targetduration(self, value: str) -> None:
        if False:
            return 10
        '\n        EXT-X-TARGETDURATION\n        https://datatracker.ietf.org/doc/html/rfc8216#section-4.3.3.1\n        '
        self.m3u8.targetduration = float(value)

    @parse_tag('EXT-X-MEDIA-SEQUENCE')
    def parse_tag_ext_x_media_sequence(self, value: str) -> None:
        if False:
            for i in range(10):
                print('nop')
        '\n        EXT-X-MEDIA-SEQUENCE\n        https://datatracker.ietf.org/doc/html/rfc8216#section-4.3.3.2\n        '
        self.m3u8.media_sequence = int(value)

    @parse_tag('EXT-X-DISCONTINUTY-SEQUENCE')
    def parse_tag_ext_x_discontinuity_sequence(self, value: str) -> None:
        if False:
            return 10
        '\n        EXT-X-DISCONTINUITY-SEQUENCE\n        https://datatracker.ietf.org/doc/html/rfc8216#section-4.3.3.3\n        '
        self.m3u8.discontinuity_sequence = int(value)

    @parse_tag('EXT-X-ENDLIST')
    def parse_tag_ext_x_endlist(self, value: str) -> None:
        if False:
            return 10
        '\n        EXT-X-ENDLIST\n        https://datatracker.ietf.org/doc/html/rfc8216#section-4.3.3.4\n        '
        self.m3u8.is_endlist = True

    @parse_tag('EXT-X-PLAYLIST-TYPE')
    def parse_tag_ext_x_playlist_type(self, value: str) -> None:
        if False:
            return 10
        '\n        EXT-X-PLAYLISTTYPE\n        https://datatracker.ietf.org/doc/html/rfc8216#section-4.3.3.5\n        '
        self.m3u8.playlist_type = value

    @parse_tag('EXT-X-I-FRAMES-ONLY')
    def parse_tag_ext_x_i_frames_only(self, value: str) -> None:
        if False:
            return 10
        '\n        EXT-X-I-FRAMES-ONLY\n        https://datatracker.ietf.org/doc/html/rfc8216#section-4.3.3.6\n        '
        self.m3u8.iframes_only = True

    @parse_tag('EXT-X-MEDIA')
    def parse_tag_ext_x_media(self, value: str) -> None:
        if False:
            i = 10
            return i + 15
        '\n        EXT-X-MEDIA\n        https://datatracker.ietf.org/doc/html/rfc8216#section-4.3.4.1\n        '
        attr = self.parse_attributes(value)
        _type = attr.get('TYPE')
        uri = attr.get('URI')
        group_id = attr.get('GROUP-ID')
        name = attr.get('NAME')
        if not _type or not group_id or (not name):
            return
        media = Media(type=_type, uri=self.uri(uri) if uri else None, group_id=group_id, language=attr.get('LANGUAGE'), name=name, default=self.parse_bool(attr.get('DEFAULT', 'NO')), autoselect=self.parse_bool(attr.get('AUTOSELECT', 'NO')), forced=self.parse_bool(attr.get('FORCED', 'NO')), characteristics=attr.get('CHARACTERISTICS'))
        self.m3u8.media.append(media)

    @parse_tag('EXT-X-STREAM-INF')
    def parse_tag_ext_x_stream_inf(self, value: str) -> None:
        if False:
            for i in range(10):
                print('nop')
        '\n        EXT-X-STREAM-INF\n        https://datatracker.ietf.org/doc/html/rfc8216#section-4.3.4.2\n        '
        self._expect_playlist = True
        self._streaminf = self.parse_attributes(value)

    @parse_tag('EXT-X-I-FRAME-STREAM-INF')
    def parse_tag_ext_x_i_frame_stream_inf(self, value: str) -> None:
        if False:
            for i in range(10):
                print('nop')
        '\n        EXT-X-I-FRAME-STREAM-INF\n        https://datatracker.ietf.org/doc/html/rfc8216#section-4.3.4.3\n        '
        attr = self.parse_attributes(value)
        uri = attr.get('URI')
        streaminf = self._streaminf or attr
        self._streaminf = None
        if not uri:
            return
        stream_info = self.create_stream_info(streaminf, IFrameStreamInfo)
        playlist = HLSPlaylist(uri=self.uri(uri), stream_info=stream_info, media=[], is_iframe=True)
        self.m3u8.playlists.append(playlist)

    @parse_tag('EXT-X-SESSION-DATA')
    def parse_tag_ext_x_session_data(self, value: str) -> None:
        if False:
            i = 10
            return i + 15
        '\n        EXT-X-SESSION-DATA\n        https://datatracker.ietf.org/doc/html/rfc8216#section-4.3.4.4\n        '

    @parse_tag('EXT-X-SESSION-KEY')
    def parse_tag_ext_x_session_key(self, value: str) -> None:
        if False:
            i = 10
            return i + 15
        '\n        EXT-X-SESSION-KEY\n        https://datatracker.ietf.org/doc/html/rfc8216#section-4.3.4.5\n        '

    @parse_tag('EXT-X-INDEPENDENT-SEGMENTS')
    def parse_tag_ext_x_independent_segments(self, value: str) -> None:
        if False:
            i = 10
            return i + 15
        '\n        EXT-X-INDEPENDENT-SEGMENTS\n        https://datatracker.ietf.org/doc/html/rfc8216#section-4.3.5.1\n        '

    @parse_tag('EXT-X-START')
    def parse_tag_ext_x_start(self, value: str) -> None:
        if False:
            for i in range(10):
                print('nop')
        '\n        EXT-X-START\n        https://datatracker.ietf.org/doc/html/rfc8216#section-4.3.5.2\n        '
        attr = self.parse_attributes(value)
        self.m3u8.start = Start(time_offset=float(attr.get('TIME-OFFSET', 0)), precise=self.parse_bool(attr.get('PRECISE', 'NO')))

    @parse_tag('EXT-X-ALLOW-CACHE')
    def parse_tag_ext_x_allow_cache(self, value: str) -> None:
        if False:
            return 10
        self.m3u8.allow_cache = self.parse_bool(value)

    def parse_line(self, line: str) -> None:
        if False:
            while True:
                i = 10
        if line.startswith('#'):
            (tag, value) = self.split_tag(line)
            if not tag or value is None or tag not in self._TAGS:
                return
            self._TAGS[tag](self, value)
        elif self._expect_segment:
            self._expect_segment = False
            segment = self.get_segment(self.uri(line))
            self.m3u8.segments.append(segment)
        elif self._expect_playlist:
            self._expect_playlist = False
            playlist = self.get_playlist(self.uri(line))
            self.m3u8.playlists.append(playlist)

    def parse(self, data: Union[str, Response]) -> TM3U8_co:
        if False:
            i = 10
            return i + 15
        lines: Iterator[str]
        if isinstance(data, str):
            lines = iter(filter(bool, data.splitlines()))
        else:
            lines = iter(filter(bool, data.iter_lines(decode_unicode=True)))
        try:
            line = next(lines)
        except StopIteration:
            return self.m3u8
        else:
            if not line.startswith('#EXTM3U'):
                log.warning(f'Malformed HLS Playlist. Expected #EXTM3U, but got {line[:250]}')
                raise ValueError('Missing #EXTM3U header')
        lines = log.iter(ALL, lines)
        parse_line = self.parse_line
        for line in lines:
            parse_line(line)
        for playlist in self.m3u8.playlists:
            for media_type in ('audio', 'video', 'subtitles'):
                group_id = getattr(playlist.stream_info, media_type, None)
                if group_id:
                    for media in filter(lambda m: m.group_id == group_id, self.m3u8.media):
                        playlist.media.append(media)
        self.m3u8.is_master = not not self.m3u8.playlists
        media_sequence = self.m3u8.media_sequence or 0
        for (i, segment) in enumerate(self.m3u8.segments):
            segment.num = media_sequence + i
        return self.m3u8

    def uri(self, uri: str) -> str:
        if False:
            print('Hello World!')
        if uri and urlparse(uri).scheme:
            return uri
        elif uri and self.m3u8.uri:
            return urljoin(self.m3u8.uri, uri)
        else:
            return uri

    def get_segment(self, uri: str, **data) -> HLSSegment:
        if False:
            print('Hello World!')
        extinf: ExtInf = self._extinf or ExtInf(0, None)
        self._extinf = None
        discontinuity = self._discontinuity
        self._discontinuity = False
        byterange = self._byterange
        self._byterange = None
        date = self._date
        self._date = None
        return self.__segment__(uri=uri, num=-1, duration=extinf.duration, title=extinf.title, key=self._key, discontinuity=discontinuity, byterange=byterange, date=date, map=self._map, **data)

    def get_playlist(self, uri: str, **data) -> HLSPlaylist:
        if False:
            i = 10
            return i + 15
        streaminf = self._streaminf or {}
        self._streaminf = None
        stream_info = self.create_stream_info(streaminf)
        return self.__playlist__(uri=uri, stream_info=stream_info, media=[], is_iframe=False, **data)

def parse_m3u8(data: Union[str, Response], base_uri: Optional[str]=None, parser: Type[M3U8Parser[TM3U8_co, THLSSegment_co, THLSPlaylist_co]]=M3U8Parser) -> TM3U8_co:
    if False:
        i = 10
        return i + 15
    '\n    Parse an M3U8 playlist from a string of data or an HTTP response.\n\n    If specified, *base_uri* is the base URI that relative URIs will\n    be joined together with, otherwise relative URIs will be as is.\n\n    If specified, *parser* can be an M3U8Parser subclass to be used\n    to parse the data.\n    '
    if base_uri is None and isinstance(data, Response):
        base_uri = data.url
    return parser(base_uri).parse(data)