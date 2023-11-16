from __future__ import absolute_import, print_function
import os
import sys
import http.client as httplib
from http.client import _CS_REQ_SENT, _CS_REQ_STARTED, CONTINUE, UnknownProtocol, CannotSendHeader, NO_CONTENT, NOT_MODIFIED, EXPECTATION_FAILED, HTTPMessage, HTTPException
from io import StringIO
from .BaseUtils import encode_to_s3
_METHODS_EXPECTING_BODY = ['PATCH', 'POST', 'PUT']

def _encode(data, name='data'):
    if False:
        for i in range(10):
            print('nop')
    'Call data.encode("latin-1") but show a better error message.'
    try:
        return data.encode('latin-1')
    except UnicodeEncodeError as err:
        exc = UnicodeEncodeError(err.encoding, err.object, err.start, err.end, "%s (%.20r) is not valid Latin-1. Use %s.encode('utf-8') if you want to send it encoded in UTF-8." % (name.title(), data[err.start:err.end], name))
        exc.__cause__ = None
        raise exc

def httpresponse_patched_begin(self):
    if False:
        for i in range(10):
            print('nop')
    ' Re-implemented httplib begin function\n    to not loop over "100 CONTINUE" status replies\n    but to report it to higher level so it can be processed.\n    '
    if self.headers is not None:
        return
    (version, status, reason) = self._read_status()
    self.code = self.status = status
    self.reason = reason.strip()
    if version in ('HTTP/1.0', 'HTTP/0.9'):
        self.version = 10
    elif version.startswith('HTTP/1.'):
        self.version = 11
    else:
        raise UnknownProtocol(version)
    self.headers = self.msg = httplib.parse_headers(self.fp)
    if self.debuglevel > 0:
        for hdr in self.headers:
            print('header:', hdr, end=' ')
    tr_enc = self.headers.get('transfer-encoding')
    if tr_enc and tr_enc.lower() == 'chunked':
        self.chunked = True
        self.chunk_left = None
    else:
        self.chunked = False
    self.will_close = self._check_close()
    self.length = None
    length = self.headers.get('content-length')
    if length and (not self.chunked):
        try:
            self.length = int(length)
        except ValueError:
            self.length = None
        else:
            if self.length < 0:
                self.length = None
    else:
        self.length = None
    if status == NO_CONTENT or status == NOT_MODIFIED or 100 <= status < 200 or (self._method == 'HEAD'):
        self.length = 0
    if not self.will_close and (not self.chunked) and (self.length is None):
        self.will_close = True

def httpconnection_patched_get_content_length(body, method):
    if False:
        while True:
            i = 10
    '## REIMPLEMENTED because new in last httplib but needed by send_request'
    'Get the content-length based on the body.\n\n    If the body is None, we set Content-Length: 0 for methods that expect\n    a body (RFC 7230, Section 3.3.2). We also set the Content-Length for\n    any method if the body is a str or bytes-like object and not a file.\n    '
    if body is None:
        if method.upper() in _METHODS_EXPECTING_BODY:
            return 0
        else:
            return None
    if hasattr(body, 'read'):
        return None
    try:
        mv = memoryview(body)
        return mv.nbytes
    except TypeError:
        pass
    if isinstance(body, str):
        return len(body)
    return None

def httpconnection_patched_send_request(self, method, url, body, headers, encode_chunked=False):
    if False:
        for i in range(10):
            print('nop')
    header_names = dict.fromkeys([k.lower() for k in headers])
    skips = {}
    if 'host' in header_names:
        skips['skip_host'] = 1
    if 'accept-encoding' in header_names:
        skips['skip_accept_encoding'] = 1
    expect_continue = False
    for (hdr, value) in headers.items():
        if 'expect' == hdr.lower() and '100-continue' in value.lower():
            expect_continue = True
    self.putrequest(method, url, **skips)
    if 'content-length' not in header_names:
        if 'transfer-encoding' not in header_names:
            encode_chunked = False
            content_length = httpconnection_patched_get_content_length(body, method)
            if content_length is None:
                if body is not None:
                    if self.debuglevel > 0:
                        print('Unable to determine size of %r' % body)
                    encode_chunked = True
                    self.putheader('Transfer-Encoding', 'chunked')
            else:
                self.putheader('Content-Length', str(content_length))
    else:
        encode_chunked = False
    for (hdr, value) in headers.items():
        self.putheader(encode_to_s3(hdr), encode_to_s3(value))
    if isinstance(body, str):
        body = _encode(body, 'body')
    if not expect_continue:
        self.endheaders(body, encode_chunked=encode_chunked)
    else:
        if not body:
            raise HTTPException('A body is required when expecting 100-continue')
        self.endheaders()
        resp = self.getresponse()
        resp.read()
        self._HTTPConnection__state = _CS_REQ_SENT
        if resp.status == EXPECTATION_FAILED:
            raise ExpectationFailed()
        elif resp.status == CONTINUE:
            self.wrapper_send_body(body, encode_chunked)

def httpconnection_patched_endheaders(self, message_body=None, encode_chunked=False):
    if False:
        for i in range(10):
            print('nop')
    'REIMPLEMENTED because new argument encode_chunked added after py 3.4'
    'Indicate that the last header line has been sent to the server.\n\n    This method sends the request to the server.  The optional message_body\n    argument can be used to pass a message body associated with the\n    request.\n    '
    if self._HTTPConnection__state == _CS_REQ_STARTED:
        self._HTTPConnection__state = _CS_REQ_SENT
    else:
        raise CannotSendHeader()
    self._send_output(message_body, encode_chunked=encode_chunked)

def httpconnection_patched_read_readable(self, readable):
    if False:
        for i in range(10):
            print('nop')
    'REIMPLEMENTED because needed by send_output and added after py 3.4\n    '
    blocksize = 8192
    if self.debuglevel > 0:
        print('sendIng a read()able')
    encode = self._is_textIO(readable)
    if encode and self.debuglevel > 0:
        print('encoding file using iso-8859-1')
    while True:
        datablock = readable.read(blocksize)
        if not datablock:
            break
        if encode:
            datablock = datablock.encode('iso-8859-1')
        yield datablock

def httpconnection_patched_send_output(self, message_body=None, encode_chunked=False):
    if False:
        i = 10
        return i + 15
    'REIMPLEMENTED because needed by endheaders and parameter\n    encode_chunked was added'
    'Send the currently buffered request and clear the buffer.\n\n    Appends an extra \\r\\n to the buffer.\n    A message_body may be specified, to be appended to the request.\n    '
    self._buffer.extend((b'', b''))
    msg = b'\r\n'.join(self._buffer)
    del self._buffer[:]
    self.send(msg)
    if message_body is not None:
        self.wrapper_send_body(message_body, encode_chunked)

class ExpectationFailed(HTTPException):
    pass

def httpconnection_patched_wrapper_send_body(self, message_body, encode_chunked=False):
    if False:
        for i in range(10):
            print('nop')
    if hasattr(message_body, 'read'):
        chunks = self._read_readable(message_body)
    else:
        try:
            memoryview(message_body)
        except TypeError:
            try:
                chunks = iter(message_body)
            except TypeError:
                raise TypeError('message_body should be a bytes-like object or an iterable, got %r' % type(message_body))
        else:
            chunks = (message_body,)
    for chunk in chunks:
        if not chunk:
            if self.debuglevel > 0:
                print('Zero length chunk ignored')
            continue
        if encode_chunked and self._http_vsn == 11:
            chunk = '{:X}\r\n'.format(len(chunk)).encode('ascii') + chunk + b'\r\n'
        self.send(chunk)
    if encode_chunked and self._http_vsn == 11:
        self.send(b'0\r\n\r\n')
httplib.HTTPResponse.begin = httpresponse_patched_begin
httplib.HTTPConnection.endheaders = httpconnection_patched_endheaders
httplib.HTTPConnection._send_readable = httpconnection_patched_read_readable
httplib.HTTPConnection._send_output = httpconnection_patched_send_output
httplib.HTTPConnection._send_request = httpconnection_patched_send_request
httplib.HTTPConnection.wrapper_send_body = httpconnection_patched_wrapper_send_body