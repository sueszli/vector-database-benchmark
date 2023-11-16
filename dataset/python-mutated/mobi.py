import io
import numbers
import os
from struct import pack, unpack
from calibre.ebooks import normalize
from calibre.ebooks.mobi import MAX_THUMB_DIMEN, MobiError
from calibre.ebooks.mobi.langcodes import iana2mobi
from calibre.ebooks.mobi.utils import rescale_image
from calibre.utils.date import now as nowf
from calibre.utils.imghdr import what
from calibre.utils.localization import canonicalize_lang, lang_as_iso639_1
from polyglot.builtins import codepoint_to_chr
'\nRetrieve and modify in-place Mobipocket book metadata.\n'

def is_image(ss):
    if False:
        return 10
    if ss is None:
        return False
    return what(None, ss[:200]) is not None

class StreamSlicer:

    def __init__(self, stream, start=0, stop=None):
        if False:
            while True:
                i = 10
        self._stream = stream
        self.start = start
        if stop is None:
            stream.seek(0, 2)
            stop = stream.tell()
        self.stop = stop
        self._len = stop - start

    def __len__(self):
        if False:
            while True:
                i = 10
        return self._len

    def __getitem__(self, key):
        if False:
            i = 10
            return i + 15
        stream = self._stream
        base = self.start
        if isinstance(key, numbers.Integral):
            stream.seek(base + key)
            return stream.read(1)
        if isinstance(key, slice):
            (start, stop, stride) = key.indices(self._len)
            if stride < 0:
                (start, stop) = (stop, start)
            size = stop - start
            if size <= 0:
                return b''
            stream.seek(base + start)
            data = stream.read(size)
            if stride != 1:
                data = data[::stride]
            return data
        raise TypeError('stream indices must be integers')

    def __setitem__(self, key, value):
        if False:
            i = 10
            return i + 15
        stream = self._stream
        base = self.start
        if isinstance(key, numbers.Integral):
            if len(value) != 1:
                raise ValueError('key and value lengths must match')
            stream.seek(base + key)
            return stream.write(value)
        if isinstance(key, slice):
            (start, stop, stride) = key.indices(self._len)
            if stride < 0:
                (start, stop) = (stop, start)
            size = stop - start
            if stride != 1:
                value = value[::stride]
            if len(value) != size:
                raise ValueError('key and value lengths must match')
            stream.seek(base + start)
            return stream.write(value)
        raise TypeError('stream indices must be integers')

    def update(self, data_blocks):
        if False:
            return 10
        stream = self._stream
        base = self.start
        stream.seek(base)
        stream.truncate()
        for block in data_blocks:
            stream.write(block)

    def truncate(self, value):
        if False:
            for i in range(10):
                print('nop')
        self._stream.truncate(value)

class MetadataUpdater:
    DRM_KEY_SIZE = 48

    def __init__(self, stream):
        if False:
            return 10
        self.stream = stream
        data = self.data = StreamSlicer(stream)
        self.type = data[60:68]
        if self.type != b'BOOKMOBI':
            return
        (self.nrecs,) = unpack('>H', data[76:78])
        record0 = self.record0 = self.record(0)
        (mobi_header_length,) = unpack('>I', record0[20:24])
        if not mobi_header_length:
            raise MobiError("Non-standard file format.  Try 'Convert E-Books' with MOBI as Input and Output formats.")
        (self.encryption_type,) = unpack('>H', record0[12:14])
        (codepage,) = unpack('>I', record0[28:32])
        self.codec = 'utf-8' if codepage == 65001 else 'cp1252'
        (image_base,) = unpack('>I', record0[108:112])
        (flags,) = (self.flags,) = unpack('>I', record0[128:132])
        have_exth = self.have_exth = flags & 64 != 0
        self.cover_record = self.thumbnail_record = None
        self.timestamp = None
        self.pdbrecords = self.get_pdbrecords()
        self.drm_block = None
        if self.encryption_type != 0:
            if self.have_exth:
                self.drm_block = self.fetchDRMdata()
            else:
                raise MobiError('Unable to set metadata on DRM file without EXTH header')
        self.original_exth_records = {}
        if not have_exth:
            self.create_exth()
            self.have_exth = True
        self.fetchEXTHFields()

    def fetchDRMdata(self):
        if False:
            for i in range(10):
                print('nop')
        ' Fetch the DRM keys '
        drm_offset = int(unpack('>I', self.record0[168:172])[0])
        self.drm_key_count = int(unpack('>I', self.record0[172:176])[0])
        drm_keys = b''
        for x in range(self.drm_key_count):
            base_addr = drm_offset + x * self.DRM_KEY_SIZE
            drm_keys += self.record0[base_addr:base_addr + self.DRM_KEY_SIZE]
        return drm_keys

    def fetchEXTHFields(self):
        if False:
            while True:
                i = 10
        stream = self.stream
        record0 = self.record0
        exth_off = unpack('>I', record0[20:24])[0] + 16 + record0.start
        (image_base,) = unpack('>I', record0[108:112])
        exth = self.exth = StreamSlicer(stream, exth_off, record0.stop)
        (nitems,) = unpack('>I', exth[8:12])
        pos = 12
        for i in range(nitems):
            (id, size) = unpack('>II', exth[pos:pos + 8])
            content = exth[pos + 8:pos + size]
            pos += size
            self.original_exth_records[id] = content
            if id == 106:
                self.timestamp = content
            elif id == 201:
                (rindex,) = (self.cover_rindex,) = unpack('>I', content)
                if rindex != 4294967295:
                    self.cover_record = self.record(rindex + image_base)
            elif id == 202:
                (rindex,) = (self.thumbnail_rindex,) = unpack('>I', content)
                if rindex > 0 and rindex != 4294967295:
                    self.thumbnail_record = self.record(rindex + image_base)

    def patch(self, off, new_record0):
        if False:
            while True:
                i = 10
        record_sizes = [len(new_record0)]
        for i in range(1, self.nrecs - 1):
            record_sizes.append(self.pdbrecords[i + 1][0] - self.pdbrecords[i][0])
        record_sizes.append(self.data.stop - self.pdbrecords[self.nrecs - 1][0])
        updated_pdbrecords = [self.pdbrecords[0][0]]
        record0_offset = self.pdbrecords[0][0]
        updated_offset = record0_offset + len(new_record0)
        for i in range(1, self.nrecs - 1):
            updated_pdbrecords.append(updated_offset)
            updated_offset += record_sizes[i]
        updated_pdbrecords.append(updated_offset)
        data_blocks = [new_record0]
        for i in range(1, self.nrecs):
            data_blocks.append(self.data[self.pdbrecords[i][0]:self.pdbrecords[i][0] + record_sizes[i]])
        self.record0.update(data_blocks)
        self.update_pdbrecords(updated_pdbrecords)
        if updated_pdbrecords[-1] + record_sizes[-1] < self.data.stop:
            self.data.truncate(updated_pdbrecords[-1] + record_sizes[-1])
        else:
            self.data.stop = updated_pdbrecords[-1] + record_sizes[-1]

    def patchSection(self, section, new):
        if False:
            i = 10
            return i + 15
        off = self.pdbrecords[section][0]
        self.patch(off, new)

    def create_exth(self, new_title=None, exth=None):
        if False:
            while True:
                i = 10
        if isinstance(new_title, str):
            new_title = new_title.encode(self.codec, 'replace')
        (title_offset,) = unpack('>L', self.record0[84:88])
        (title_length,) = unpack('>L', self.record0[88:92])
        (title_in_file,) = unpack('%ds' % title_length, self.record0[title_offset:title_offset + title_length])
        (mobi_header_length,) = unpack('>L', self.record0[20:24])
        if mobi_header_length == 228:
            self.record0[23] = b'\xe8'
            self.record0[244:248] = pack('>L', 4294967295)
            mobi_header_length = 232
        self.record0[128:132] = pack('>L', self.flags | 64)
        if not exth:
            pad = b'\x00' * 4
            exth = [b'EXTH', pack('>II', 12, 0), pad]
            exth = b''.join(exth)
        if self.encryption_type != 0:
            self.record0[168:172] = pack('>L', 16 + mobi_header_length + len(exth))
            self.record0[176:180] = pack('>L', len(self.drm_block))
            self.record0[84:88] = pack('>L', 16 + mobi_header_length + len(exth) + len(self.drm_block))
        else:
            self.record0[84:88] = pack('>L', 16 + mobi_header_length + len(exth))
        if new_title:
            self.record0[88:92] = pack('>L', len(new_title))
        new_record0 = io.BytesIO()
        new_record0.write(self.record0[:16 + mobi_header_length])
        new_record0.write(exth)
        if self.encryption_type != 0:
            new_record0.write(self.drm_block)
        new_record0.write(new_title if new_title else title_in_file)
        trail = len(new_record0.getvalue()) % 4
        pad = b'\x00' * (4 - trail)
        new_record0.write(pad)
        new_record0.write(b'\x00' * (1024 * 8))
        self.patchSection(0, new_record0.getvalue())
        self.record0 = self.record(0)

    def hexdump(self, src, length=16):
        if False:
            return 10
        FILTER = ''.join([len(repr(codepoint_to_chr(x))) == 3 and codepoint_to_chr(x) or '.' for x in range(256)])
        N = 0
        result = ''
        while src:
            (s, src) = (src[:length], src[length:])
            hexa = ' '.join(['%02X' % ord(x) for x in s])
            s = s.translate(FILTER)
            result += '%04X   %-*s   %s\n' % (N, length * 3, hexa, s)
            N += length
        print(result)

    def get_pdbrecords(self):
        if False:
            return 10
        pdbrecords = []
        for i in range(self.nrecs):
            (offset, a1, a2, a3, a4) = unpack('>LBBBB', self.data[78 + i * 8:78 + i * 8 + 8])
            (flags, val) = (a1, a2 << 16 | a3 << 8 | a4)
            pdbrecords.append([offset, flags, val])
        return pdbrecords

    def update_pdbrecords(self, updated_pdbrecords):
        if False:
            print('Hello World!')
        for (i, pdbrecord) in enumerate(updated_pdbrecords):
            self.data[78 + i * 8:78 + i * 8 + 4] = pack('>L', pdbrecord)
        self.pdbrecords = self.get_pdbrecords()

    def dump_pdbrecords(self):
        if False:
            while True:
                i = 10
        print('MetadataUpdater.dump_pdbrecords()')
        print('%10s %10s %10s' % ('offset', 'flags', 'val'))
        for i in range(len(self.pdbrecords)):
            pdbrecord = self.pdbrecords[i]
            print(f'{pdbrecord[0]:10X} {pdbrecord[1]:10X} {pdbrecord[2]:10X}')

    def record(self, n):
        if False:
            for i in range(10):
                print('nop')
        if n >= self.nrecs:
            raise ValueError('non-existent record %r' % n)
        offoff = 78 + 8 * n
        (start,) = unpack('>I', self.data[offoff + 0:offoff + 4])
        stop = None
        if n < self.nrecs - 1:
            (stop,) = unpack('>I', self.data[offoff + 8:offoff + 12])
        return StreamSlicer(self.stream, start, stop)

    def update(self, mi, asin=None):
        if False:
            for i in range(10):
                print('nop')
        mi.title = normalize(mi.title)

        def update_exth_record(rec):
            if False:
                for i in range(10):
                    print('nop')
            recs.append(rec)
            if rec[0] in self.original_exth_records:
                self.original_exth_records.pop(rec[0])
        if self.type != b'BOOKMOBI':
            raise MobiError("Setting metadata only supported for MOBI files of type 'BOOK'.\n\tThis is a %r file of type %r" % (self.type[0:4], self.type[4:8]))
        recs = []
        added_501 = False
        try:
            from calibre.ebooks.conversion.config import load_defaults
            prefs = load_defaults('mobi_output')
            pas = prefs.get('prefer_author_sort', False)
            kindle_pdoc = prefs.get('personal_doc', None)
            share_not_sync = prefs.get('share_not_sync', False)
        except:
            pas = False
            kindle_pdoc = None
            share_not_sync = False
        if mi.author_sort and pas:
            authors = mi.author_sort.split(' & ')
            for author in authors:
                update_exth_record((100, normalize(author).encode(self.codec, 'replace')))
        elif mi.authors:
            authors = mi.authors
            for author in authors:
                update_exth_record((100, normalize(author).encode(self.codec, 'replace')))
        if mi.publisher:
            update_exth_record((101, normalize(mi.publisher).encode(self.codec, 'replace')))
        if mi.comments:
            a_offset = mi.comments.find('<div class="user_annotations">')
            ad_offset = mi.comments.find('<hr class="annotations_divider" />')
            if a_offset >= 0:
                mi.comments = mi.comments[:a_offset]
            if ad_offset >= 0:
                mi.comments = mi.comments[:ad_offset]
            update_exth_record((103, normalize(mi.comments).encode(self.codec, 'replace')))
        if mi.isbn:
            update_exth_record((104, mi.isbn.encode(self.codec, 'replace')))
        if mi.tags:
            subjects = '; '.join(mi.tags)
            update_exth_record((105, normalize(subjects).encode(self.codec, 'replace')))
            if kindle_pdoc and kindle_pdoc in mi.tags:
                added_501 = True
                update_exth_record((501, b'PDOC'))
        if mi.pubdate:
            update_exth_record((106, str(mi.pubdate).encode(self.codec, 'replace')))
        elif mi.timestamp:
            update_exth_record((106, str(mi.timestamp).encode(self.codec, 'replace')))
        elif self.timestamp:
            update_exth_record((106, self.timestamp))
        else:
            update_exth_record((106, nowf().isoformat().encode(self.codec, 'replace')))
        if self.cover_record is not None:
            update_exth_record((201, pack('>I', self.cover_rindex)))
            update_exth_record((203, pack('>I', 0)))
        if self.thumbnail_record is not None:
            update_exth_record((202, pack('>I', self.thumbnail_rindex)))
        if 113 not in self.original_exth_records and self.original_exth_records.get(501, None) == b'EBOK' and (not added_501) and (not share_not_sync):
            from uuid import uuid4
            update_exth_record((113, str(uuid4()).encode(self.codec)))
        if asin is not None:
            update_exth_record((113, asin.encode(self.codec)))
            update_exth_record((501, b'EBOK'))
            update_exth_record((504, asin.encode(self.codec)))
        if getattr(mi, 'uuid', None):
            update_exth_record((112, ('calibre:%s' % mi.uuid).encode(self.codec, 'replace')))
        if 503 in self.original_exth_records:
            update_exth_record((503, mi.title.encode(self.codec, 'replace')))
        if getattr(mi, 'book_producer', False):
            update_exth_record((108, mi.book_producer.encode(self.codec, 'replace')))
        if not mi.is_null('language'):
            lang = canonicalize_lang(mi.language)
            lang = lang_as_iso639_1(lang) or lang
            if lang:
                update_exth_record((524, lang.encode(self.codec, 'replace')))
        for id in sorted(self.original_exth_records):
            recs.append((id, self.original_exth_records[id]))
        recs = sorted(recs, key=lambda x: (x[0], x[0]))
        exth = io.BytesIO()
        for (code, data) in recs:
            exth.write(pack('>II', code, len(data) + 8))
            exth.write(data)
        exth = exth.getvalue()
        trail = len(exth) % 4
        pad = b'\x00' * (4 - trail)
        exth = [b'EXTH', pack('>II', len(exth) + 12, len(recs)), exth, pad]
        exth = b''.join(exth)
        if getattr(self, 'exth', None) is None:
            raise MobiError('No existing EXTH record. Cannot update metadata.')
        if not mi.is_null('language'):
            self.record0[92:96] = iana2mobi(mi.language)
        self.create_exth(exth=exth, new_title=mi.title)
        self.fetchEXTHFields()
        if mi.cover_data[1] or mi.cover:
            try:
                data = mi.cover_data[1]
                if not data:
                    with open(mi.cover, 'rb') as f:
                        data = f.read()
            except:
                pass
            else:
                if is_image(self.cover_record):
                    size = len(self.cover_record)
                    cover = rescale_image(data, size)
                    if len(cover) <= size:
                        cover += b'\x00' * (size - len(cover))
                        self.cover_record[:] = cover
                if is_image(self.thumbnail_record):
                    size = len(self.thumbnail_record)
                    thumbnail = rescale_image(data, size, dimen=MAX_THUMB_DIMEN)
                    if len(thumbnail) <= size:
                        thumbnail += b'\x00' * (size - len(thumbnail))
                        self.thumbnail_record[:] = thumbnail
                return

def set_metadata(stream, mi):
    if False:
        for i in range(10):
            print('nop')
    mu = MetadataUpdater(stream)
    mu.update(mi)
    return

def get_metadata(stream):
    if False:
        print('Hello World!')
    from calibre.ebooks.metadata import MetaInformation
    from calibre.ptempfile import TemporaryDirectory
    from calibre.ebooks.mobi.reader.headers import MetadataHeader
    from calibre.ebooks.mobi.reader.mobi6 import MobiReader
    from calibre.utils.img import save_cover_data_to
    from calibre import CurrentDir
    stream.seek(0)
    try:
        raw = stream.read(3)
    except Exception:
        raw = b''
    stream.seek(0)
    if raw == b'TPZ':
        from calibre.ebooks.metadata.topaz import get_metadata
        return get_metadata(stream)
    from calibre.utils.logging import Log
    log = Log()
    try:
        mi = MetaInformation(os.path.basename(stream.name), [_('Unknown')])
    except:
        mi = MetaInformation(_('Unknown'), [_('Unknown')])
    mh = MetadataHeader(stream, log)
    if mh.title and mh.title != _('Unknown'):
        mi.title = mh.title
    if mh.exth is not None:
        if mh.exth.mi is not None:
            mi = mh.exth.mi
    else:
        size = 1024 ** 3
        if hasattr(stream, 'seek') and hasattr(stream, 'tell'):
            pos = stream.tell()
            stream.seek(0, 2)
            size = stream.tell()
            stream.seek(pos)
        if size < 4 * 1024 * 1024:
            with TemporaryDirectory('_mobi_meta_reader') as tdir:
                with CurrentDir(tdir):
                    mr = MobiReader(stream, log)
                    parse_cache = {}
                    mr.extract_content(tdir, parse_cache)
                    if mr.embedded_mi is not None:
                        mi = mr.embedded_mi
    if hasattr(mh.exth, 'cover_offset'):
        cover_index = mh.first_image_index + mh.exth.cover_offset
        data = mh.section_data(int(cover_index))
    else:
        try:
            data = mh.section_data(mh.first_image_index)
        except Exception:
            data = b''
    if data and what(None, data) in {'jpg', 'jpeg', 'gif', 'png', 'bmp', 'webp'}:
        try:
            mi.cover_data = ('jpg', save_cover_data_to(data))
        except Exception:
            log.exception('Failed to read MOBI cover')
    return mi