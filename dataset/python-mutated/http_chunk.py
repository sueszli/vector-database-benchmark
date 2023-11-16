import codecs
import os
import re
import time
import urllib.parse
from cgi import parse_header as parse_header_line
from email.header import decode_header as parse_mime_header
from ntpath import basename as ntpath_basename
from posixpath import basename as posixpath_basename
import pycurl
from pyload.core.utils import purge, parse
from .http_request import HTTPRequest

class WrongFormat(Exception):
    pass

class ChunkInfo:

    def __init__(self, name):
        if False:
            return 10
        self.name = os.fsdecode(name)
        self.size = 0
        self.resume = False
        self.chunks = []

    def __repr__(self):
        if False:
            i = 10
            return i + 15
        ret = f'ChunkInfo: {self.name}, {self.size}\n'
        for (i, c) in enumerate(self.chunks):
            ret += f'{i}# {c[1]}\n'
        return ret

    def set_size(self, size):
        if False:
            for i in range(10):
                print('nop')
        self.size = int(size)

    def add_chunk(self, name, range):
        if False:
            for i in range(10):
                print('nop')
        self.chunks.append((name, range))

    def clear(self):
        if False:
            i = 10
            return i + 15
        self.chunks = []

    def create_chunks(self, chunks):
        if False:
            while True:
                i = 10
        self.clear()
        chunk_size = self.size // chunks
        current = 0
        for i in range(chunks):
            end = self.size - 1 if i == chunks - 1 else current + chunk_size
            self.add_chunk(f'{self.name}.chunk{i}', (current, end))
            current += chunk_size + 1

    def save(self):
        if False:
            print('Hello World!')
        fs_name = f'{self.name}.chunks'
        with open(fs_name, mode='w', encoding='utf-8', newline='\n') as fh:
            fh.write(f'name:{self.name}\n')
            fh.write(f'size:{self.size}\n')
            for (i, c) in enumerate(self.chunks):
                fh.write(f'#{i}:\n')
                fh.write(f'\tname:{c[0]}\n')
                fh.write(f'\trange:{c[1][0]}-{c[1][1]}\n')

    @staticmethod
    def load(name):
        if False:
            for i in range(10):
                print('nop')
        fs_name = f'{name}.chunks'
        if not os.path.exists(fs_name):
            raise IOError
        with open(fs_name, encoding='utf-8') as fh:
            name = fh.readline()[:-1]
            size = fh.readline()[:-1]
            if name.startswith('name:') and size.startswith('size:'):
                name = name[5:]
                size = size[5:]
            else:
                fh.close()
                raise WrongFormat
            ci = ChunkInfo(name)
            ci.loaded = True
            ci.set_size(size)
            while True:
                if not fh.readline():
                    break
                name = fh.readline()[1:-1]
                range = fh.readline()[1:-1]
                if name.startswith('name:') and range.startswith('range:'):
                    name = name[5:]
                    range = range[6:].split('-')
                else:
                    raise WrongFormat
                ci.add_chunk(name, (int(range[0]), int(range[1])))
        return ci

    def remove(self):
        if False:
            i = 10
            return i + 15
        fs_name = f'{self.name}.chunks'
        if os.path.exists(fs_name):
            os.remove(fs_name)

    def get_count(self):
        if False:
            for i in range(10):
                print('nop')
        return len(self.chunks)

    def get_chunk_name(self, index):
        if False:
            i = 10
            return i + 15
        return self.chunks[index][0]

    def get_chunk_range(self, index):
        if False:
            print('Hello World!')
        return self.chunks[index][1]

class HTTPChunk(HTTPRequest):

    def __init__(self, id, parent, range=None, resume=False):
        if False:
            for i in range(10):
                print('nop')
        self.id = id
        self.p = parent
        self.range = range
        self.resume = resume
        self.log = parent.log
        self.size = range[1] - range[0] if range else -1
        self.arrived = 0
        self.last_url = self.p.referer
        self.aborted = False
        self.c = pycurl.Curl()
        self.response_header = b''
        self.header_parsed = False
        self.fp = None
        self.init_handle()
        self.c.setopt(pycurl.ENCODING, None)
        self.set_interface(self.p.options)
        self.BOMChecked = False
        self.rep = None
        self.sleep = 0.0
        self.last_size = 0

    def __repr__(self):
        if False:
            while True:
                i = 10
        return f'<HTTPChunk id={self.id}, size={self.size}, arrived={self.arrived}>'

    @property
    def cj(self):
        if False:
            while True:
                i = 10
        return self.p.cj

    def format_range(self):
        if False:
            while True:
                i = 10
        if self.id == len(self.p.info.chunks) - 1:
            end = ''
            if self.resume:
                start = self.arrived + self.range[0]
            else:
                start = self.range[0]
        else:
            end = min(self.range[1] + 1, self.p.size - 1)
            if self.id == 0 and (not self.resume):
                start = 0
            else:
                start = self.arrived + self.range[0]
        return f'{start}-{end}'

    def get_handle(self):
        if False:
            while True:
                i = 10
        '\n        returns a Curl handle ready to use for perform/multiperform.\n        '
        self.set_request_context(self.p.url, self.p.get, self.p.post, self.p.referer, self.p.cj)
        self.c.setopt(pycurl.WRITEFUNCTION, self.write_body)
        self.c.setopt(pycurl.HEADERFUNCTION, self.write_header)
        fs_name = self.p.info.get_chunk_name(self.id)
        if self.resume:
            self.fp = open(fs_name, mode='ab')
            self.arrived = self.fp.tell()
            if not self.arrived:
                self.arrived = os.stat(fs_name).st_size
            if self.range:
                if self.arrived + self.range[0] >= self.range[1]:
                    return None
                range = self.format_range()
                self.log.debug(f'Chunk {self.id + 1} chunked with range {range}')
                self.c.setopt(pycurl.RANGE, range)
            else:
                self.log.debug(f'Resume File from {self.arrived}')
                self.c.setopt(pycurl.RESUME_FROM, self.arrived)
        else:
            if self.range:
                range = self.format_range()
                self.log.debug(f'Chunk {self.id + 1} chunked with range {range}')
                self.c.setopt(pycurl.RANGE, range)
            self.fp = open(fs_name, mode='wb')
        return self.c

    def write_header(self, buf):
        if False:
            i = 10
            return i + 15
        self.response_header += buf
        if not self.range and self.response_header.endswith(b'\r\n\r\n'):
            self.parse_header()
        elif not self.range and buf.startswith(b'150') and (b'data connection' in buf):
            size = re.search(b'(\\d+) bytes', buf)
            if size:
                self.p.size = int(size.group(1))
                self.p.chunk_support = True
            self.header_parsed = True

    def write_body(self, buf):
        if False:
            print('Hello World!')
        if not self.BOMChecked:
            if buf[:3] == codecs.BOM_UTF8:
                buf = buf[3:]
            self.BOMChecked = True
        size = len(buf)
        self.arrived += size
        self.fp.write(buf)
        if self.p.bucket:
            time.sleep(self.p.bucket.consumed(size))
        else:
            if size < self.last_size:
                self.sleep += 0.002
            else:
                self.sleep *= 0.7
            self.last_size = size
            time.sleep(self.sleep)
        if self.range and self.arrived > self.size:
            self.aborted = True
            return 0

    def parse_header(self):
        if False:
            return 10
        '\n        parse data from received header.\n        '
        location = None
        for orgline in self.response_header.splitlines():
            try:
                orgline = orgline.decode('utf-8')
            except UnicodeDecodeError:
                try:
                    orgline = orgline.decode('iso-8859-1')
                except UnicodeDecodeError:
                    continue
            line = orgline.strip().lower()
            if line.startswith('accept-ranges') and 'bytes' in line:
                self.p.chunk_support = True
            elif line.startswith('location'):
                location = orgline.split(':', 1)[1].strip()
            elif line.startswith('content-disposition'):
                disposition_value = orgline.split(':', 1)[1].strip()
                (disposition_type, disposition_params) = parse_header_line(disposition_value)
                fname = None
                if 'filename*' in disposition_params:
                    fname = disposition_params['filename*']
                    m = re.search('=\\?([^?]+)\\?([QB])\\?([^?]*)\\?=', fname, re.I)
                    if m is not None:
                        (data, encoding) = parse_mime_header(fname)[0]
                        try:
                            fname = data.decode(encoding)
                        except LookupError:
                            self.log.warning(f'Content-Disposition: | error: No decoder found for {encoding}')
                            fname = None
                        except UnicodeEncodeError:
                            self.log.warning(f'Content-Disposition: | error: Error when decoding string from {encoding}')
                            fname = None
                    else:
                        m = re.search("(.+?)\\'(.*)\\'(.+)", fname)
                        if m is not None:
                            (encoding, lang, data) = m.groups()
                            try:
                                fname = urllib.parse.unquote(data, encoding=encoding, errors='strict')
                            except LookupError:
                                self.log.warning(f'Content-Disposition: | error: No decoder found for {encoding}')
                                fname = None
                            except UnicodeDecodeError:
                                self.log.warning(f'Content-Disposition: | error: Error when decoding string from {encoding}')
                                fname = None
                        else:
                            fname = None
                if fname is None:
                    if 'filename' in disposition_params:
                        fname = disposition_params['filename']
                        m = re.search('=\\?([^?]+)\\?([QB])\\?([^?]*)\\?=', fname, re.I)
                        if m is not None:
                            (data, encoding) = parse_mime_header(m.group(0))[0]
                            try:
                                fname = data.decode(encoding)
                            except LookupError:
                                self.log.warning(f'Content-Disposition: | error: No decoder found for {encoding}')
                                continue
                            except UnicodeEncodeError:
                                self.log.warning(f'Content-Disposition: | error: Error when decoding string from {encoding}')
                                continue
                        else:
                            try:
                                fname = urllib.parse.unquote(fname, encoding='iso-8859-1', errors='strict')
                            except UnicodeDecodeError:
                                self.log.warning('Content-Disposition: | error: Error when decoding string from iso-8859-1.')
                                continue
                    elif disposition_type.lower() == 'attachment':
                        if location is not None:
                            fname = parse.name(location)
                        else:
                            fname = parse.name(self.p.url)
                    else:
                        continue
                fname = posixpath_basename(fname)
                fname = ntpath_basename(fname)
                fname = purge.name(fname, sep='')
                fname = fname.lstrip('.')
                self.log.debug(f'Content-Disposition: {fname}')
                self.p.update_disposition(fname)
            if not self.resume and line.startswith('content-length'):
                self.p.size = int(line.split(':', 1)[1])
        self.header_parsed = True

    def stop(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        The download will not proceed after next call of write_body.\n        '
        self.range = [0, 0]
        self.size = 0

    def reset_range(self):
        if False:
            return 10
        '\n        Reset the range, so the download will load all data available.\n        '
        self.range = None

    def set_range(self, range):
        if False:
            print('Hello World!')
        self.range = range
        self.size = range[1] - range[0]
        self.log.debug('Chunk {id} chunked with range {range}'.format(id=self.id + 1, range=self.format_range()))

    def flush_file(self):
        if False:
            i = 10
            return i + 15
        '\n        flush and close file.\n        '
        self.fp.flush()
        os.fsync(self.fp.fileno())
        self.fp.close()

    def close(self):
        if False:
            i = 10
            return i + 15
        '\n        closes everything, unusable after this.\n        '
        if self.fp:
            self.fp.close()
        self.c.close()
        if hasattr(self, 'p'):
            del self.p