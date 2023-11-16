import os.path
from mimetypes import guess_type
VERSION = (0, 3, 6)
__version__ = '.'.join(map(str, VERSION))

def _lazy_load(fn):
    if False:
        i = 10
        return i + 15
    _cached = []

    def _decorated():
        if False:
            for i in range(10):
                print('nop')
        if not _cached:
            _cached.append(fn())
        return _cached[0]

    def clear():
        if False:
            i = 10
            return i + 15
        while _cached:
            _cached.pop()
    _decorated.clear = clear
    return _decorated

@_lazy_load
def _get_sendfile():
    if False:
        for i in range(10):
            print('nop')
    from importlib import import_module
    from django.conf import settings
    from django.core.exceptions import ImproperlyConfigured
    backend = getattr(settings, 'SENDFILE_BACKEND', None)
    if not backend:
        raise ImproperlyConfigured('You must specify a value for SENDFILE_BACKEND')
    module = import_module(backend)
    return module.sendfile

def sendfile(request, filename, attachment=False, attachment_filename=None, mimetype=None, encoding=None, backend=None):
    if False:
        print('Hello World!')
    '\n    create a response to send file using backend configured in SENDFILE_BACKEND\n\n    If attachment is True the content-disposition header will be set.\n    This will typically prompt the user to download the file, rather\n    than view it.  The content-disposition filename depends on the\n    value of attachment_filename:\n\n        None (default): Same as filename\n        False: No content-disposition filename\n        String: Value used as filename\n\n    If no mimetype or encoding are specified, then they will be guessed via the\n    filename (using the standard python mimetypes module)\n    '
    _sendfile = backend or _get_sendfile()
    if not os.path.exists(filename):
        from django.http import Http404
        raise Http404('"%s" does not exist' % filename)
    (guessed_mimetype, guessed_encoding) = guess_type(filename)
    if mimetype is None:
        if guessed_mimetype:
            mimetype = guessed_mimetype
        else:
            mimetype = 'application/octet-stream'
    response = _sendfile(request, filename, mimetype=mimetype)
    if attachment:
        parts = ['attachment']
    else:
        parts = ['inline']
    if attachment_filename is None:
        attachment_filename = os.path.basename(filename)
    if attachment_filename:
        from django.utils.encoding import force_str
        from wagtail.coreutils import string_to_ascii
        attachment_filename = force_str(attachment_filename)
        ascii_filename = string_to_ascii(attachment_filename)
        parts.append('filename="%s"' % ascii_filename)
        if ascii_filename != attachment_filename:
            from urllib.parse import quote
            quoted_filename = quote(attachment_filename)
            parts.append("filename*=UTF-8''%s" % quoted_filename)
    response['Content-Disposition'] = '; '.join(parts)
    response['Content-length'] = os.path.getsize(filename)
    response['Content-Type'] = mimetype
    response['Content-Encoding'] = encoding or guessed_encoding
    return response