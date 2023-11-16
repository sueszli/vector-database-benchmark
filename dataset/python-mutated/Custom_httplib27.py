from __future__ import absolute_import, print_function
import os
import httplib
from httplib import _CS_REQ_SENT, _CS_REQ_STARTED, CONTINUE, UnknownProtocol, CannotSendHeader, NO_CONTENT, NOT_MODIFIED, EXPECTATION_FAILED, HTTPMessage, HTTPException
try:
    from cStringIO import StringIO
except ImportError:
    from StringIO import StringIO
from .BaseUtils import encode_to_s3
_METHODS_EXPECTING_BODY = ['PATCH', 'POST', 'PUT']

def httpresponse_patched_begin(self):
    if False:
        i = 10
        return i + 15
    ' Re-implemented httplib begin function\n    to not loop over "100 CONTINUE" status replies\n    but to report it to higher level so it can be processed.\n    '
    if self.msg is not None:
        return
    (version, status, reason) = self._read_status()
    self.status = status
    self.reason = reason.strip()
    if version == 'HTTP/1.0':
        self.version = 10
    elif version.startswith('HTTP/1.'):
        self.version = 11
    elif version == 'HTTP/0.9':
        self.version = 9
    else:
        raise UnknownProtocol(version)
    if self.version == 9:
        self.length = None
        self.chunked = 0
        self.will_close = 1
        self.msg = HTTPMessage(StringIO())
        return
    self.msg = HTTPMessage(self.fp, 0)
    if self.debuglevel > 0:
        for hdr in self.msg.headers:
            print('header:', hdr, end=' ')
    self.msg.fp = None
    tr_enc = self.msg.getheader('transfer-encoding')
    if tr_enc and tr_enc.lower() == 'chunked':
        self.chunked = 1
        self.chunk_left = None
    else:
        self.chunked = 0
    self.will_close = self._check_close()
    length = self.msg.getheader('content-length')
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
        self.will_close = 1

def httpconnection_patched_set_content_length(self, body, method):
    if False:
        return 10
    thelen = None
    if body is None and method.upper() in _METHODS_EXPECTING_BODY:
        thelen = '0'
    elif body is not None:
        try:
            thelen = str(len(body))
        except (TypeError, AttributeError):
            try:
                thelen = str(os.fstat(body.fileno()).st_size)
            except (AttributeError, OSError):
                if self.debuglevel > 0:
                    print('Cannot stat!!')
    if thelen is not None:
        self.putheader('Content-Length', thelen)

def httpconnection_patched_send_request(self, method, url, body, headers):
    if False:
        print('Hello World!')
    header_names = dict.fromkeys([k.lower() for k in headers])
    skips = {}
    if 'host' in header_names:
        skips['skip_host'] = 1
    if 'accept-encoding' in header_names:
        skips['skip_accept_encoding'] = 1
    expect_continue = False
    for (hdr, value) in headers.iteritems():
        if 'expect' == hdr.lower() and '100-continue' in value.lower():
            expect_continue = True
    url = encode_to_s3(url)
    self.putrequest(method, url, **skips)
    if 'content-length' not in header_names:
        self._set_content_length(body, method)
    for (hdr, value) in headers.iteritems():
        self.putheader(encode_to_s3(hdr), encode_to_s3(value))
    if not expect_continue:
        self.endheaders(body)
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
            self.send(body)

def httpconnection_patched_endheaders(self, message_body=None):
    if False:
        for i in range(10):
            print('nop')
    'Indicate that the last header line has been sent to the server.\n\n    This method sends the request to the server.  The optional\n    message_body argument can be used to pass a message body\n    associated with the request.  The message body will be sent in\n    the same packet as the message headers if it is string, otherwise it is\n    sent as a separate packet.\n    '
    if self._HTTPConnection__state == _CS_REQ_STARTED:
        self._HTTPConnection__state = _CS_REQ_SENT
    else:
        raise CannotSendHeader()
    self._send_output(message_body)
mss = 16384

def httpconnection_patched_send_output(self, message_body=None):
    if False:
        while True:
            i = 10
    'Send the currently buffered request and clear the buffer.\n\n    Appends an extra \\r\\n to the buffer.\n    A message_body may be specified, to be appended to the request.\n    '
    self._buffer.extend((b'', b''))
    msg = b'\r\n'.join(self._buffer)
    del self._buffer[:]
    msg = encode_to_s3(msg)
    if isinstance(message_body, str) and len(message_body) < mss:
        msg += message_body
        message_body = None
    self.send(msg)
    if message_body is not None:
        self.send(message_body)

class ExpectationFailed(HTTPException):
    pass

def httpconnection_patched_wrapper_send_body(self, message_body):
    if False:
        for i in range(10):
            print('nop')
    self.send(message_body)
httplib.HTTPResponse.begin = httpresponse_patched_begin
httplib.HTTPConnection.endheaders = httpconnection_patched_endheaders
httplib.HTTPConnection._send_output = httpconnection_patched_send_output
httplib.HTTPConnection._set_content_length = httpconnection_patched_set_content_length
httplib.HTTPConnection._send_request = httpconnection_patched_send_request
httplib.HTTPConnection.wrapper_send_body = httpconnection_patched_wrapper_send_body