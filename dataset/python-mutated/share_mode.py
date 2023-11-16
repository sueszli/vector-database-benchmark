"""
OnionShare | https://onionshare.org/

Copyright (C) 2014-2022 Micah Lee, et al. <micah@micahflee.com>

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""
import binascii
import hashlib
import os
import sys
import tempfile
import zipfile
import mimetypes
from datetime import datetime, timezone
from flask import Response, request, render_template, make_response, abort
from unidecode import unidecode
from werkzeug.http import parse_date, http_date
from urllib.parse import quote
from .send_base_mode import SendBaseModeWeb

def make_etag(data):
    if False:
        i = 10
        return i + 15
    hasher = hashlib.sha256()
    while True:
        read_bytes = data.read(4096)
        if read_bytes:
            hasher.update(read_bytes)
        else:
            break
    hash_value = binascii.hexlify(hasher.digest()).decode('utf-8')
    return '"sha256:{}"'.format(hash_value)

def parse_range_header(range_header: str, target_size: int) -> list:
    if False:
        return 10
    end_index = target_size - 1
    if range_header is None:
        return [(0, end_index)]
    bytes_ = 'bytes='
    if not range_header.startswith(bytes_):
        abort(416)
    ranges = []
    for range_ in range_header[len(bytes_):].split(','):
        split = range_.split('-')
        if len(split) == 1:
            try:
                start = int(split[0])
                end = end_index
            except ValueError:
                abort(416)
        elif len(split) == 2:
            (start, end) = (split[0], split[1])
            if not start:
                end = end_index
                try:
                    start = end - int(split[1]) + 1
                except ValueError:
                    abort(416)
            else:
                try:
                    start = int(start)
                    if not end:
                        end = target_size
                    else:
                        end = int(end)
                except ValueError:
                    abort(416)
                if end < start:
                    abort(416)
                end = min(end, end_index)
        else:
            abort(416)
        ranges.append((start, end))
    merged = []
    ranges = sorted(ranges, key=lambda x: x[0])
    for range_ in ranges:
        if not merged:
            merged.append(range_)
        elif range_[0] <= merged[-1][1] + 1:
            merged[-1] = (merged[-1][0], max(range_[1], merged[-1][1]))
        else:
            merged.append(range_)
    return merged

class ShareModeWeb(SendBaseModeWeb):
    """
    All of the web logic for share mode
    """

    def init(self):
        if False:
            for i in range(10):
                print('nop')
        self.common.log('ShareModeWeb', 'init')
        self.download_individual_files = not self.web.settings.get('share', 'autostop_sharing')
        self.download_etag = None
        self.gzip_etag = None
        self.last_modified = datetime.now(tz=timezone.utc)

    def define_routes(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        The web app routes for sharing files\n        '

        @self.web.app.route('/', defaults={'path': ''}, methods=['GET'], provide_automatic_options=False)
        @self.web.app.route('/<path:path>', methods=['GET'], provide_automatic_options=False)
        def index(path):
            if False:
                print('Hello World!')
            '\n            Render the template for the onionshare landing page.\n            '
            self.web.add_request(self.web.REQUEST_LOAD, request.path)
            deny_download = self.web.settings.get('share', 'autostop_sharing') and self.download_in_progress
            if deny_download:
                return render_template('denied.html')
            if self.should_use_gzip():
                self.filesize = self.gzip_filesize
            else:
                self.filesize = self.download_filesize
            return self.render_logic(path)

        @self.web.app.route('/download', methods=['GET'], provide_automatic_options=False)
        def download():
            if False:
                print('Hello World!')
            '\n            Download the zip file.\n            '
            deny_download = self.web.settings.get('share', 'autostop_sharing') and self.download_in_progress
            if deny_download:
                return render_template('denied.html')
            request_path = request.path
            use_gzip = self.should_use_gzip()
            if use_gzip:
                file_to_download = self.gzip_filename
                self.filesize = self.gzip_filesize
                etag = self.gzip_etag
            else:
                file_to_download = self.download_filename
                self.filesize = self.download_filesize
                etag = self.download_etag
            (range_, status_code) = self.get_range_and_status_code(self.filesize, etag, self.last_modified)
            history_id = self.cur_history_id
            self.cur_history_id += 1
            self.web.add_request(self.web.REQUEST_STARTED, request_path, {'id': history_id, 'use_gzip': use_gzip})
            basename = os.path.basename(self.download_filename)
            if status_code == 304:
                r = Response()
            else:
                r = Response(self.generate(range_, file_to_download, request_path, history_id, self.filesize))
            if use_gzip:
                r.headers.set('Content-Encoding', 'gzip')
            r.headers.set('Content-Length', range_[1] - range_[0] + 1)
            filename_dict = {'filename': unidecode(basename), 'filename*': "UTF-8''%s" % quote(basename)}
            r.headers.set('Content-Disposition', 'attachment', **filename_dict)
            (content_type, _) = mimetypes.guess_type(basename, strict=False)
            if content_type is not None:
                r.headers.set('Content-Type', content_type)
            r.headers.set('Accept-Ranges', 'bytes')
            r.headers.set('ETag', etag)
            r.headers.set('Last-Modified', http_date(self.last_modified))
            r.headers.set('Vary', 'Accept-Encoding')
            if status_code == 206:
                r.headers.set('Content-Range', 'bytes {}-{}/{}'.format(range_[0], range_[1], self.filesize))
            r.status_code = status_code
            return r

    @classmethod
    def get_range_and_status_code(cls, dl_size, etag, last_modified):
        if False:
            print('Hello World!')
        use_default_range = True
        status_code = 200
        range_header = request.headers.get('Range')
        if request.method == 'GET':
            ranges = parse_range_header(range_header, dl_size)
            if not (len(ranges) == 1 and ranges[0][0] == 0 and (ranges[0][1] == dl_size - 1)):
                use_default_range = False
                status_code = 206
            if range_header:
                if_range = request.headers.get('If-Range')
                if if_range and if_range != etag:
                    use_default_range = True
                    status_code = 200
        if use_default_range:
            ranges = [(0, dl_size - 1)]
        if len(ranges) > 1:
            abort(416)
        range_ = ranges[0]
        etag_header = request.headers.get('ETag')
        if etag_header is not None and etag_header != etag:
            abort(412)
        if_unmod = request.headers.get('If-Unmodified-Since')
        if if_unmod:
            if_date = parse_date(if_unmod)
            if if_date and (not if_date.tzinfo):
                if_date = if_date.replace(tzinfo=timezone.utc)
            if if_date and if_date > last_modified:
                abort(412)
            elif range_header is None:
                status_code = 304
        return (range_, status_code)

    def generate(self, range_, file_to_download, path, history_id, filesize):
        if False:
            i = 10
            return i + 15
        self.client_cancel = False
        if self.web.settings.get('share', 'autostop_sharing'):
            self.download_in_progress = True
        (start, end) = range_
        chunk_size = 102400
        fp = open(file_to_download, 'rb')
        fp.seek(start)
        self.web.done = False
        canceled = False
        bytes_left = end - start + 1
        while not self.web.done:
            if not self.web.stop_q.empty():
                self.web.add_request(self.web.REQUEST_CANCELED, path, {'id': history_id})
                break
            read_size = min(chunk_size, bytes_left)
            chunk = fp.read(read_size)
            if chunk == b'':
                self.web.done = True
            else:
                try:
                    yield chunk
                    downloaded_bytes = fp.tell()
                    percent = 1.0 * downloaded_bytes / filesize * 100
                    bytes_left -= read_size
                    if not self.web.is_gui or self.common.platform == 'Linux' or self.common.platform == 'BSD':
                        sys.stdout.write('\r{0:s}, {1:.2f}%          '.format(self.common.human_readable_filesize(downloaded_bytes), percent))
                        sys.stdout.flush()
                    self.web.add_request(self.web.REQUEST_PROGRESS, path, {'id': history_id, 'bytes': downloaded_bytes, 'total_bytes': filesize})
                    self.web.done = False
                except Exception:
                    self.web.done = True
                    canceled = True
                    self.web.add_request(self.web.REQUEST_CANCELED, path, {'id': history_id})
        fp.close()
        if self.common.platform != 'Darwin':
            sys.stdout.write('\n')
        if self.web.settings.get('share', 'autostop_sharing'):
            self.download_in_progress = False
        if self.web.settings.get('share', 'autostop_sharing') and (not canceled):
            print('Stopped because transfer is complete')
            self.web.running = False
            try:
                self.web.stop()
            except Exception:
                pass

    def directory_listing_template(self, path, files, dirs, breadcrumbs, breadcrumbs_leaf):
        if False:
            i = 10
            return i + 15
        return make_response(render_template('send.html', file_info=self.file_info, files=files, dirs=dirs, breadcrumbs=breadcrumbs, breadcrumbs_leaf=breadcrumbs_leaf, filename=os.path.basename(self.download_filename), filesize=self.filesize, filesize_human=self.common.human_readable_filesize(self.download_filesize), is_zipped=self.is_zipped, static_url_path=self.web.static_url_path, download_individual_files=self.download_individual_files, title=self.web.settings.get('general', 'title')))

    def set_file_info_custom(self, filenames, processed_size_callback):
        if False:
            while True:
                i = 10
        self.common.log('ShareModeWeb', 'set_file_info_custom')
        self.web.cancel_compression = False
        self.build_zipfile_list(filenames, processed_size_callback)

    def render_logic(self, path=''):
        if False:
            print('Hello World!')
        if path in self.files:
            filesystem_path = self.files[path]
            if os.path.isdir(filesystem_path):
                filenames = []
                for filename in os.listdir(filesystem_path):
                    filenames.append(filename)
                filenames.sort()
                return self.directory_listing(filenames, path, filesystem_path)
            elif os.path.isfile(filesystem_path):
                if self.download_individual_files:
                    return self.stream_individual_file(filesystem_path)
                else:
                    history_id = self.cur_history_id
                    self.cur_history_id += 1
                    return self.web.error404(history_id)
            else:
                history_id = self.cur_history_id
                self.cur_history_id += 1
                return self.web.error404(history_id)
        elif path == '':
            filenames = list(self.root_files)
            filenames.sort()
            return self.directory_listing(filenames, path)
        else:
            history_id = self.cur_history_id
            self.cur_history_id += 1
            return self.web.error404(history_id)

    def build_zipfile_list(self, filenames, processed_size_callback=None):
        if False:
            return 10
        self.common.log('ShareModeWeb', 'build_zipfile_list', f'filenames={filenames}')
        for filename in filenames:
            info = {'filename': filename, 'basename': os.path.basename(filename.rstrip('/'))}
            if os.path.isfile(filename):
                info['size'] = os.path.getsize(filename)
                info['size_human'] = self.common.human_readable_filesize(info['size'])
                self.file_info['files'].append(info)
            if os.path.isdir(filename):
                info['size'] = self.common.dir_size(filename)
                info['size_human'] = self.common.human_readable_filesize(info['size'])
                self.file_info['dirs'].append(info)
        self.file_info['files'].sort(key=lambda k: k['basename'])
        self.file_info['dirs'].sort(key=lambda k: k['basename'])
        if len(self.file_info['files']) == 1 and len(self.file_info['dirs']) == 0:
            self.download_filename = self.file_info['files'][0]['filename']
            self.download_filesize = self.file_info['files'][0]['size']
            with open(self.download_filename, 'rb') as f:
                self.download_etag = make_etag(f)
            self.gzip_tmp_dir = tempfile.TemporaryDirectory(dir=self.common.build_tmp_dir())
            self.gzip_filename = os.path.join(self.gzip_tmp_dir.name, 'file.gz')
            self._gzip_compress(self.download_filename, self.gzip_filename, 6, processed_size_callback)
            self.gzip_filesize = os.path.getsize(self.gzip_filename)
            with open(self.gzip_filename, 'rb') as f:
                self.gzip_etag = make_etag(f)
            self.is_zipped = False
            self.web.cleanup_tempdirs.append(self.gzip_tmp_dir)
        else:
            self.zip_writer = ZipWriter(self.common, self.web, processed_size_callback=processed_size_callback)
            self.download_filename = self.zip_writer.zip_filename
            for info in self.file_info['files']:
                self.zip_writer.add_file(info['filename'])
                if self.web.cancel_compression:
                    self.zip_writer.close()
                    return False
            for info in self.file_info['dirs']:
                if not self.zip_writer.add_dir(info['filename']):
                    return False
            self.zip_writer.close()
            self.download_filesize = os.path.getsize(self.download_filename)
            with open(self.download_filename, 'rb') as f:
                self.download_etag = make_etag(f)
            self.is_zipped = True
        return True

class ZipWriter(object):
    """
    ZipWriter accepts files and directories and compresses them into a zip file
    with. If a zip_filename is not passed in, it will use the default onionshare
    filename.
    """

    def __init__(self, common, web=None, zip_filename=None, processed_size_callback=None):
        if False:
            i = 10
            return i + 15
        self.common = common
        self.web = web
        self.cancel_compression = False
        if zip_filename:
            self.zip_filename = zip_filename
        else:
            self.zip_temp_dir = tempfile.TemporaryDirectory(dir=self.common.build_tmp_dir())
            self.zip_filename = f'{self.zip_temp_dir.name}/onionshare_{self.common.random_string(4, 6)}.zip'
            if self.web:
                self.web.cleanup_tempdirs.append(self.zip_temp_dir)
        self.z = zipfile.ZipFile(self.zip_filename, 'w', allowZip64=True)
        self.processed_size_callback = processed_size_callback
        if self.processed_size_callback is None:
            self.processed_size_callback = lambda _: None
        self._size = 0
        self.processed_size_callback(self._size)

    def add_file(self, filename):
        if False:
            i = 10
            return i + 15
        '\n        Add a file to the zip archive.\n        '
        self.z.write(filename, os.path.basename(filename), zipfile.ZIP_DEFLATED)
        self._size += os.path.getsize(filename)
        self.processed_size_callback(self._size)

    def add_dir(self, filename):
        if False:
            print('Hello World!')
        '\n        Add a directory, and all of its children, to the zip archive.\n        '
        dir_to_strip = os.path.dirname(filename.rstrip('/')) + '/'
        for (dirpath, dirnames, filenames) in os.walk(filename):
            for f in filenames:
                if self.cancel_compression:
                    return False
                full_filename = os.path.join(dirpath, f)
                if not os.path.islink(full_filename):
                    arc_filename = full_filename[len(dir_to_strip):]
                    self.z.write(full_filename, arc_filename, zipfile.ZIP_DEFLATED)
                    self._size += os.path.getsize(full_filename)
                    self.processed_size_callback(self._size)
        return True

    def close(self):
        if False:
            while True:
                i = 10
        '\n        Close the zip archive.\n        '
        self.z.close()