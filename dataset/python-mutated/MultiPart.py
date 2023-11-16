from __future__ import absolute_import
import sys
from logging import debug, info, warning, error
from .Crypto import calculateChecksum
from .Exceptions import ParameterError
from .S3Uri import S3UriS3
from .BaseUtils import getTextFromXml, getTreeFromXml, s3_quote, parseNodes
from .Utils import formatSize
SIZE_1MB = 1024 * 1024

class MultiPartUpload(object):
    """Supports MultiPartUpload and MultiPartUpload(Copy) operation"""
    MIN_CHUNK_SIZE_MB = 5
    MAX_CHUNK_SIZE_MB = 5 * 1024
    MAX_FILE_SIZE = 5 * 1024 * 1024

    def __init__(self, s3, src, dst_uri, headers_baseline=None, src_size=None):
        if False:
            while True:
                i = 10
        self.s3 = s3
        self.file_stream = None
        self.src_uri = None
        self.src_size = src_size
        self.dst_uri = dst_uri
        self.parts = {}
        self.headers_baseline = headers_baseline or {}
        if isinstance(src, S3UriS3):
            self.src_uri = src
            if not src_size:
                raise ParameterError('Source size is missing for MultipartUploadCopy operation')
            c_size = self.s3.config.multipart_copy_chunk_size_mb * SIZE_1MB
        else:
            self.file_stream = src
            c_size = self.s3.config.multipart_chunk_size_mb * SIZE_1MB
        self.chunk_size = c_size
        self.upload_id = self.initiate_multipart_upload()

    def get_parts_information(self, uri, upload_id):
        if False:
            for i in range(10):
                print('nop')
        part_list = self.s3.list_multipart(uri, upload_id)
        parts = dict()
        for elem in part_list:
            try:
                parts[int(elem['PartNumber'])] = {'checksum': elem['ETag'], 'size': elem['Size']}
            except KeyError:
                pass
        return parts

    def get_unique_upload_id(self, uri):
        if False:
            while True:
                i = 10
        upload_id = ''
        multipart_list = self.s3.get_multipart(uri)
        for mpupload in multipart_list:
            try:
                mp_upload_id = mpupload['UploadId']
                mp_path = mpupload['Key']
                info('mp_path: %s, object: %s' % (mp_path, uri.object()))
                if mp_path == uri.object():
                    if upload_id:
                        raise ValueError('More than one UploadId for URI %s.  Disable multipart upload, or use\n %s multipart %s\nto list the Ids, then pass a unique --upload-id into the put command.' % (uri, sys.argv[0], uri))
                    upload_id = mp_upload_id
            except KeyError:
                pass
        return upload_id

    def initiate_multipart_upload(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Begin a multipart upload\n        http://docs.amazonwebservices.com/AmazonS3/latest/API/index.html?mpUploadInitiate.html\n        '
        if self.s3.config.upload_id:
            self.upload_id = self.s3.config.upload_id
        elif self.s3.config.put_continue:
            self.upload_id = self.get_unique_upload_id(self.dst_uri)
        else:
            self.upload_id = ''
        if not self.upload_id:
            request = self.s3.create_request('OBJECT_POST', uri=self.dst_uri, headers=self.headers_baseline, uri_params={'uploads': None})
            response = self.s3.send_request(request)
            data = response['data']
            self.upload_id = getTextFromXml(data, 'UploadId')
        return self.upload_id

    def upload_all_parts(self, extra_label=''):
        if False:
            print('Hello World!')
        '\n        Execute a full multipart upload on a file\n        Returns the seq/etag dict\n        TODO use num_processes to thread it\n        '
        if not self.upload_id:
            raise ParameterError('Attempting to use a multipart upload that has not been initiated.')
        remote_statuses = {}
        if self.src_uri:
            filename = self.src_uri.uri()
        else:
            filename = self.file_stream.stream_name
        if self.s3.config.put_continue:
            remote_statuses = self.get_parts_information(self.dst_uri, self.upload_id)
        if extra_label:
            extra_label = u' ' + extra_label
        labels = {'source': filename, 'destination': self.dst_uri.uri()}
        seq = 1
        if self.src_size:
            size_left = self.src_size
            nr_parts = self.src_size // self.chunk_size + (self.src_size % self.chunk_size and 1)
            debug('MultiPart: Uploading %s in %d parts' % (filename, nr_parts))
            while size_left > 0:
                offset = self.chunk_size * (seq - 1)
                current_chunk_size = min(self.src_size - offset, self.chunk_size)
                size_left -= current_chunk_size
                labels['extra'] = '[part %d of %d, %s]%s' % (seq, nr_parts, '%d%sB' % formatSize(current_chunk_size, human_readable=True), extra_label)
                try:
                    if self.file_stream:
                        self.upload_part(seq, offset, current_chunk_size, labels, remote_status=remote_statuses.get(seq))
                    else:
                        self.copy_part(seq, offset, current_chunk_size, labels, remote_status=remote_statuses.get(seq))
                except:
                    error(u"\nUpload of '%s' part %d failed. Use\n  %s abortmp %s %s\nto abort the upload, or\n  %s --upload-id %s put ...\nto continue the upload." % (filename, seq, sys.argv[0], self.dst_uri, self.upload_id, sys.argv[0], self.upload_id))
                    raise
                seq += 1
            debug('MultiPart: Upload finished: %d parts', seq - 1)
            return
        debug('MultiPart: Uploading from %s' % filename)
        while True:
            buffer = self.file_stream.read(self.chunk_size)
            offset = 0
            current_chunk_size = len(buffer)
            labels['extra'] = '[part %d of -, %s]%s' % (seq, '%d%sB' % formatSize(current_chunk_size, human_readable=True), extra_label)
            if not buffer:
                break
            try:
                self.upload_part(seq, offset, current_chunk_size, labels, buffer, remote_status=remote_statuses.get(seq))
            except:
                error(u"\nUpload of '%s' part %d failed. Use\n  %s abortmp %s %s\nto abort, or\n  %s --upload-id %s put ...\nto continue the upload." % (filename, seq, sys.argv[0], self.dst_uri, self.upload_id, sys.argv[0], self.upload_id))
                raise
            seq += 1
        debug('MultiPart: Upload finished: %d parts', seq - 1)

    def upload_part(self, seq, offset, chunk_size, labels, buffer='', remote_status=None):
        if False:
            i = 10
            return i + 15
        '\n        Upload a file chunk\n        http://docs.amazonwebservices.com/AmazonS3/latest/API/index.html?mpUploadUploadPart.html\n        '
        debug('Uploading part %i of %r (%s bytes)' % (seq, self.upload_id, chunk_size))
        if remote_status is not None:
            if int(remote_status['size']) == chunk_size:
                checksum = calculateChecksum(buffer, self.file_stream, offset, chunk_size, self.s3.config.send_chunk)
                remote_checksum = remote_status['checksum'].strip('"\'')
                if remote_checksum == checksum:
                    warning('MultiPart: size and md5sum match for %s part %d, skipping.' % (self.dst_uri, seq))
                    self.parts[seq] = remote_status['checksum']
                    return None
                else:
                    warning('MultiPart: checksum (%s vs %s) does not match for %s part %d, reuploading.' % (remote_checksum, checksum, self.dst_uri, seq))
            else:
                warning('MultiPart: size (%d vs %d) does not match for %s part %d, reuploading.' % (int(remote_status['size']), chunk_size, self.dst_uri, seq))
        headers = {'content-length': str(chunk_size)}
        query_string_params = {'partNumber': '%s' % seq, 'uploadId': self.upload_id}
        request = self.s3.create_request('OBJECT_PUT', uri=self.dst_uri, headers=headers, uri_params=query_string_params)
        response = self.s3.send_file(request, self.file_stream, labels, buffer, offset=offset, chunk_size=chunk_size)
        self.parts[seq] = response['headers'].get('etag', '').strip('"\'')
        return response

    def copy_part(self, seq, offset, chunk_size, labels, remote_status=None):
        if False:
            return 10
        '\n        Copy a remote file chunk\n        http://docs.amazonwebservices.com/AmazonS3/latest/API/index.html?mpUploadUploadPart.html\n        http://docs.amazonwebservices.com/AmazonS3/latest/API/mpUploadUploadPartCopy.html\n        '
        debug('Copying part %i of %r (%s bytes)' % (seq, self.upload_id, chunk_size))
        headers = {'x-amz-copy-source': s3_quote('/%s/%s' % (self.src_uri.bucket(), self.src_uri.object()), quote_backslashes=False, unicode_output=True)}
        headers['x-amz-copy-source-range'] = 'bytes=%d-%d' % (offset, offset + chunk_size - 1)
        query_string_params = {'partNumber': '%s' % seq, 'uploadId': self.upload_id}
        request = self.s3.create_request('OBJECT_PUT', uri=self.dst_uri, headers=headers, uri_params=query_string_params)
        labels[u'action'] = u'remote copy'
        response = self.s3.send_request_with_progress(request, labels, chunk_size)
        self.parts[seq] = getTextFromXml(response['data'], 'ETag') or ''
        return response

    def complete_multipart_upload(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Finish a multipart upload\n        http://docs.amazonwebservices.com/AmazonS3/latest/API/index.html?mpUploadComplete.html\n        '
        debug('MultiPart: Completing upload: %s' % self.upload_id)
        parts_xml = []
        part_xml = '<Part><PartNumber>%i</PartNumber><ETag>%s</ETag></Part>'
        for (seq, etag) in self.parts.items():
            parts_xml.append(part_xml % (seq, etag))
        body = '<CompleteMultipartUpload>%s</CompleteMultipartUpload>' % ''.join(parts_xml)
        headers = {'content-length': str(len(body))}
        request = self.s3.create_request('OBJECT_POST', uri=self.dst_uri, headers=headers, body=body, uri_params={'uploadId': self.upload_id})
        response = self.s3.send_request(request)
        return response

    def abort_upload(self):
        if False:
            return 10
        '\n        Abort multipart upload\n        http://docs.amazonwebservices.com/AmazonS3/latest/API/index.html?mpUploadAbort.html\n        '
        debug('MultiPart: Aborting upload: %s' % self.upload_id)
        response = None
        return response