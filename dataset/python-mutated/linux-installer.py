from __future__ import absolute_import, division, print_function, unicode_literals
import errno
import hashlib
import os
import platform
import re
import shutil
import signal
import socket
import ssl
import stat
import subprocess
import sys
import tempfile
from contextlib import closing
is64bit = platform.architecture()[0] == '64bit'
py3 = sys.version_info[0] > 2
enc = getattr(sys.stdout, 'encoding', 'utf-8') or 'utf-8'
if enc.lower() == 'ascii':
    enc = 'utf-8'
dl_url = calibre_version = signature = None
has_ssl_verify = hasattr(ssl, 'create_default_context')
is_linux_arm = is_linux_arm64 = False
machine = (os.uname()[4] or '').lower()
arch = 'x86_64'
if machine.startswith('arm') or machine.startswith('aarch64'):
    is_linux_arm = True
    is_linux_arm64 = machine.startswith('arm64') or machine.startswith('aarch64')
    arch = 'arm64'
if py3:
    unicode = str
    raw_input = input
    from urllib.parse import urlparse
    from urllib.request import BaseHandler, build_opener, Request, urlopen, getproxies, addinfourl
    import http.client as httplib

    def encode_for_subprocess(x):
        if False:
            for i in range(10):
                print('nop')
        return x
else:
    from future_builtins import map
    from urlparse import urlparse
    from urllib import urlopen, getproxies, addinfourl
    from urllib2 import BaseHandler, build_opener, Request
    import httplib

    def encode_for_subprocess(x):
        if False:
            print('Hello World!')
        if isinstance(x, unicode):
            x = x.encode(enc)
        return x

class TerminalController:
    BOL = ''
    UP = ''
    DOWN = ''
    LEFT = ''
    RIGHT = ''
    CLEAR_SCREEN = ''
    CLEAR_EOL = ''
    CLEAR_BOL = ''
    CLEAR_EOS = ''
    BOLD = ''
    BLINK = ''
    DIM = ''
    REVERSE = ''
    NORMAL = ''
    HIDE_CURSOR = ''
    SHOW_CURSOR = ''
    COLS = None
    LINES = None
    BLACK = BLUE = GREEN = CYAN = RED = MAGENTA = YELLOW = WHITE = ''
    BG_BLACK = BG_BLUE = BG_GREEN = BG_CYAN = ''
    BG_RED = BG_MAGENTA = BG_YELLOW = BG_WHITE = ''
    _STRING_CAPABILITIES = '\n    BOL=cr UP=cuu1 DOWN=cud1 LEFT=cub1 RIGHT=cuf1\n    CLEAR_SCREEN=clear CLEAR_EOL=el CLEAR_BOL=el1 CLEAR_EOS=ed BOLD=bold\n    BLINK=blink DIM=dim REVERSE=rev UNDERLINE=smul NORMAL=sgr0\n    HIDE_CURSOR=cinvis SHOW_CURSOR=cnorm'.split()
    _COLORS = 'BLACK BLUE GREEN CYAN RED MAGENTA YELLOW WHITE'.split()
    _ANSICOLORS = 'BLACK RED GREEN YELLOW BLUE MAGENTA CYAN WHITE'.split()

    def __init__(self, term_stream=sys.stdout):
        if False:
            while True:
                i = 10
        try:
            import curses
        except:
            return
        if not hasattr(term_stream, 'isatty') or not term_stream.isatty():
            return
        try:
            curses.setupterm()
        except:
            return
        self.COLS = curses.tigetnum('cols')
        self.LINES = curses.tigetnum('lines')
        for capability in self._STRING_CAPABILITIES:
            (attrib, cap_name) = capability.split('=')
            setattr(self, attrib, self._escape_code(self._tigetstr(cap_name)))
        set_fg = self._tigetstr('setf')
        if set_fg:
            if not isinstance(set_fg, bytes):
                set_fg = set_fg.encode('utf-8')
            for (i, color) in zip(range(len(self._COLORS)), self._COLORS):
                setattr(self, color, self._escape_code(curses.tparm(set_fg, i)))
        set_fg_ansi = self._tigetstr('setaf')
        if set_fg_ansi:
            if not isinstance(set_fg_ansi, bytes):
                set_fg_ansi = set_fg_ansi.encode('utf-8')
            for (i, color) in zip(range(len(self._ANSICOLORS)), self._ANSICOLORS):
                setattr(self, color, self._escape_code(curses.tparm(set_fg_ansi, i)))
        set_bg = self._tigetstr('setb')
        if set_bg:
            if not isinstance(set_bg, bytes):
                set_bg = set_bg.encode('utf-8')
            for (i, color) in zip(range(len(self._COLORS)), self._COLORS):
                setattr(self, 'BG_' + color, self._escape_code(curses.tparm(set_bg, i)))
        set_bg_ansi = self._tigetstr('setab')
        if set_bg_ansi:
            if not isinstance(set_bg_ansi, bytes):
                set_bg_ansi = set_bg_ansi.encode('utf-8')
            for (i, color) in zip(range(len(self._ANSICOLORS)), self._ANSICOLORS):
                setattr(self, 'BG_' + color, self._escape_code(curses.tparm(set_bg_ansi, i)))

    def _escape_code(self, raw):
        if False:
            print('Hello World!')
        if not raw:
            raw = ''
        if not isinstance(raw, unicode):
            raw = raw.decode('ascii')
        return raw

    def _tigetstr(self, cap_name):
        if False:
            while True:
                i = 10
        import curses
        if isinstance(cap_name, bytes):
            cap_name = cap_name.decode('utf-8')
        cap = self._escape_code(curses.tigetstr(cap_name))
        return re.sub('\\$<\\d+>[/*]?', '', cap)

    def render(self, template):
        if False:
            print('Hello World!')
        return re.sub('\\$\\$|\\${\\w+}', self._render_sub, template)

    def _render_sub(self, match):
        if False:
            print('Hello World!')
        s = match.group()
        if s == '$$':
            return s
        else:
            return getattr(self, s[2:-1])

class ProgressBar:
    BAR = '%3d%% ${GREEN}[${BOLD}%s%s${NORMAL}${GREEN}]${NORMAL}\n'
    HEADER = '${BOLD}${CYAN}%s${NORMAL}\n\n'

    def __init__(self, term, header):
        if False:
            for i in range(10):
                print('nop')
        self.term = term
        if not (self.term.CLEAR_EOL and self.term.UP and self.term.BOL):
            raise ValueError("Terminal isn't capable enough -- you should use a simpler progress display.")
        self.width = self.term.COLS or 75
        self.bar = term.render(self.BAR)
        self.header = self.term.render(self.HEADER % header.center(self.width))
        self.cleared = 1

    def update(self, percent, message=''):
        if False:
            i = 10
            return i + 15
        out = sys.stdout.buffer if py3 else sys.stdout
        if self.cleared:
            out.write(self.header.encode(enc))
            self.cleared = 0
        n = int((self.width - 10) * percent)
        msg = message.center(self.width)
        msg = (self.term.BOL + self.term.UP + self.term.CLEAR_EOL + self.bar % (100 * percent, '=' * n, '-' * (self.width - 10 - n)) + self.term.CLEAR_EOL + msg).encode(enc)
        out.write(msg)
        out.flush()

    def clear(self):
        if False:
            return 10
        out = sys.stdout.buffer if py3 else sys.stdout
        if not self.cleared:
            out.write((self.term.BOL + self.term.CLEAR_EOL + self.term.UP + self.term.CLEAR_EOL + self.term.UP + self.term.CLEAR_EOL).encode(enc))
            self.cleared = 1
            out.flush()

def prints(*args, **kwargs):
    if False:
        return 10
    f = kwargs.get('file', sys.stdout.buffer if py3 else sys.stdout)
    end = kwargs.get('end', b'\n')
    enc = getattr(f, 'encoding', 'utf-8') or 'utf-8'
    if isinstance(end, unicode):
        end = end.encode(enc)
    for x in args:
        if isinstance(x, unicode):
            x = x.encode(enc)
        f.write(x)
        f.write(b' ')
    f.write(end)
    if py3 and f is sys.stdout.buffer:
        f.flush()

class Reporter:

    def __init__(self, fname):
        if False:
            print('Hello World!')
        try:
            self.pb = ProgressBar(TerminalController(), 'Downloading ' + fname)
        except ValueError:
            prints('Downloading', fname)
            self.pb = None
        self.last_percent = 0

    def __call__(self, blocks, block_size, total_size):
        if False:
            while True:
                i = 10
        percent = blocks * block_size / float(total_size)
        if self.pb is None:
            if percent - self.last_percent > 0.05:
                self.last_percent = percent
                prints('Downloaded {0:%}'.format(percent))
        else:
            try:
                self.pb.update(percent)
            except:
                import traceback
                traceback.print_exc()

def clean_cache(cache, fname):
    if False:
        while True:
            i = 10
    for x in os.listdir(cache):
        if fname not in x:
            os.remove(os.path.join(cache, x))

def check_signature(dest, signature):
    if False:
        print('Hello World!')
    if not os.path.exists(dest):
        return None
    m = hashlib.sha512()
    with open(dest, 'rb') as f:
        raw = f.read()
    m.update(raw)
    if m.hexdigest().encode('ascii') == signature:
        return raw

class RangeHandler(BaseHandler):

    def http_error_206(self, req, fp, code, msg, hdrs):
        if False:
            return 10
        r = addinfourl(fp, hdrs, req.get_full_url())
        r.code = code
        r.msg = msg
        return r
    https_error_206 = http_error_206

def do_download(dest):
    if False:
        return 10
    prints('Will download and install', os.path.basename(dest))
    reporter = Reporter(os.path.basename(dest))
    offset = 0
    if os.path.exists(dest):
        offset = os.path.getsize(dest)
    rq = urlopen(dl_url)
    headers = rq.info()
    size = int(headers['content-length'])
    accepts_ranges = headers.get('accept-ranges', None) == 'bytes'
    mode = 'wb'
    if accepts_ranges and offset > 0:
        req = Request(rq.geturl())
        req.add_header('Range', 'bytes=%s-' % offset)
        mode = 'ab'
        rq.close()
        handler = RangeHandler()
        opener = build_opener(handler)
        rq = opener.open(req)
    with open(dest, mode) as f:
        while f.tell() < size:
            raw = rq.read(8192)
            if not raw:
                break
            f.write(raw)
            reporter(f.tell(), 1, size)
    rq.close()
    if os.path.getsize(dest) < size:
        print('Download failed, try again later')
        raise SystemExit(1)
    prints('Downloaded %s bytes' % os.path.getsize(dest))

def download_tarball():
    if False:
        i = 10
        return i + 15
    fname = 'calibre-%s-%s.%s' % (calibre_version, arch, 'txz')
    tdir = tempfile.gettempdir()
    cache = os.path.join(tdir, 'calibre-installer-cache')
    if not os.path.exists(cache):
        os.makedirs(cache)
    clean_cache(cache, fname)
    dest = os.path.join(cache, fname)
    raw = check_signature(dest, signature)
    if raw is not None:
        print('Using previously downloaded', fname)
        return raw
    cached_sigf = dest + '.signature'
    cached_sig = None
    if os.path.exists(cached_sigf):
        with open(cached_sigf, 'rb') as sigf:
            cached_sig = sigf.read()
    if cached_sig != signature and os.path.exists(dest):
        os.remove(dest)
    try:
        with open(cached_sigf, 'wb') as f:
            f.write(signature)
    except IOError as e:
        if e.errno != errno.EACCES:
            raise
        print('The installer cache directory has incorrect permissions. Delete %s and try again.' % cache)
        raise SystemExit(1)
    do_download(dest)
    prints('Checking downloaded file integrity...')
    raw = check_signature(dest, signature)
    if raw is None:
        os.remove(dest)
        print("The downloaded files' signature does not match. Try the download again later.")
        raise SystemExit(1)
    return raw

def get_proxies(debug=True):
    if False:
        while True:
            i = 10
    proxies = getproxies()
    for (key, proxy) in list(proxies.items()):
        if not proxy or '..' in proxy:
            del proxies[key]
            continue
        if proxy.startswith(key + '://'):
            proxy = proxy[len(key) + 3:]
        if key == 'https' and proxy.startswith('http://'):
            proxy = proxy[7:]
        if proxy.endswith('/'):
            proxy = proxy[:-1]
        if len(proxy) > 4:
            proxies[key] = proxy
        else:
            prints('Removing invalid', key, 'proxy:', proxy)
            del proxies[key]
    if proxies and debug:
        prints('Using proxies:', repr(proxies))
    return proxies

class HTTPError(ValueError):

    def __init__(self, url, code):
        if False:
            return 10
        msg = '%s returned an unsupported http response code: %d (%s)' % (url, code, httplib.responses.get(code, None))
        ValueError.__init__(self, msg)
        self.code = code
        self.url = url

class CertificateError(ValueError):
    pass

def _dnsname_match(dn, hostname, max_wildcards=1):
    if False:
        i = 10
        return i + 15
    'Matching according to RFC 6125, section 6.4.3\n\n    http://tools.ietf.org/html/rfc6125#section-6.4.3\n    '
    pats = []
    if not dn:
        return False
    parts = dn.split('.')
    (leftmost, remainder) = (parts[0], parts[1:])
    wildcards = leftmost.count('*')
    if wildcards > max_wildcards:
        raise CertificateError('too many wildcards in certificate DNS name: ' + repr(dn))
    if not wildcards:
        return dn.lower() == hostname.lower()
    if leftmost == '*':
        pats.append('[^.]+')
    elif leftmost.startswith('xn--') or hostname.startswith('xn--'):
        pats.append(re.escape(leftmost))
    else:
        pats.append(re.escape(leftmost).replace('\\*', '[^.]*'))
    for frag in remainder:
        pats.append(re.escape(frag))
    pat = re.compile('\\A' + '\\.'.join(pats) + '\\Z', re.IGNORECASE)
    return pat.match(hostname)

def match_hostname(cert, hostname):
    if False:
        print('Hello World!')
    'Verify that *cert* (in decoded format as returned by\n    SSLSocket.getpeercert()) matches the *hostname*.  RFC 2818 and RFC 6125\n    rules are followed, but IP addresses are not accepted for *hostname*.\n\n    CertificateError is raised on failure. On success, the function\n    returns nothing.\n    '
    if not cert:
        raise ValueError('empty or no certificate')
    dnsnames = []
    san = cert.get('subjectAltName', ())
    for (key, value) in san:
        if key == 'DNS':
            if _dnsname_match(value, hostname):
                return
            dnsnames.append(value)
    if not dnsnames:
        for sub in cert.get('subject', ()):
            for (key, value) in sub:
                if key == 'commonName':
                    if _dnsname_match(value, hostname):
                        return
                    dnsnames.append(value)
    if len(dnsnames) > 1:
        raise CertificateError("hostname %r doesn't match either of %s" % (hostname, ', '.join(map(repr, dnsnames))))
    elif len(dnsnames) == 1:
        if sys.version_info[:3] < (2, 7, 3) and dnsnames[0] == 'calibre-ebook.com':
            return
        raise CertificateError("hostname %r doesn't match %r" % (hostname, dnsnames[0]))
    else:
        raise CertificateError('no appropriate commonName or subjectAltName fields were found')
if has_ssl_verify:

    class HTTPSConnection(httplib.HTTPSConnection):

        def __init__(self, ssl_version, *args, **kwargs):
            if False:
                i = 10
                return i + 15
            kwargs['context'] = ssl.create_default_context(cafile=kwargs.pop('cert_file'))
            httplib.HTTPSConnection.__init__(self, *args, **kwargs)
else:

    class HTTPSConnection(httplib.HTTPSConnection):

        def __init__(self, ssl_version, *args, **kwargs):
            if False:
                for i in range(10):
                    print('nop')
            httplib.HTTPSConnection.__init__(self, *args, **kwargs)
            self.calibre_ssl_version = ssl_version

        def connect(self):
            if False:
                i = 10
                return i + 15
            'Connect to a host on a given (SSL) port, properly verifying the SSL\n            certificate, both that it is valid and that its declared hostnames\n            match the hostname we are connecting to.'
            if hasattr(self, 'source_address'):
                sock = socket.create_connection((self.host, self.port), self.timeout, self.source_address)
            else:
                sock = socket.create_connection((self.host, self.port), self.timeout)
            if self._tunnel_host:
                self.sock = sock
                self._tunnel()
            self.sock = ssl.wrap_socket(sock, cert_reqs=ssl.CERT_REQUIRED, ca_certs=self.cert_file, ssl_version=self.calibre_ssl_version)
            getattr(ssl, 'match_hostname', match_hostname)(self.sock.getpeercert(), self.host)
CACERT = b'-----BEGIN CERTIFICATE-----\n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\nnR0=\n-----END CERTIFICATE-----\n'

def get_https_resource_securely(url, timeout=60, max_redirects=5, ssl_version=None):
    if False:
        print('Hello World!')
    "\n    Download the resource pointed to by url using https securely (verify server\n    certificate).  Ensures that redirects, if any, are also downloaded\n    securely. Needs a CA certificates bundle (in PEM format) to verify the\n    server's certificates.\n    "
    if ssl_version is None:
        try:
            ssl_version = ssl.PROTOCOL_TLSv1_2
        except AttributeError:
            ssl_version = ssl.PROTOCOL_TLSv1
    with tempfile.NamedTemporaryFile(prefix='calibre-ca-cert-') as f:
        f.write(CACERT)
        f.flush()
        p = urlparse(url)
        if p.scheme != 'https':
            raise ValueError('URL %s scheme must be https, not %r' % (url, p.scheme))
        (hostname, port) = (p.hostname, p.port)
        proxies = get_proxies()
        has_proxy = False
        for q in ('https', 'http'):
            if q in proxies:
                try:
                    (h, po) = proxies[q].rpartition(':')[::2]
                    po = int(po)
                    if h:
                        (hostname, port, has_proxy) = (h, po, True)
                        break
                except Exception:
                    pass
        c = HTTPSConnection(ssl_version, hostname, port, cert_file=f.name, timeout=timeout)
        if has_proxy:
            c.set_tunnel(p.hostname, p.port)
        with closing(c):
            c.connect()
            path = p.path or '/'
            if p.query:
                path += '?' + p.query
            c.request('GET', path)
            response = c.getresponse()
            if response.status in (httplib.MOVED_PERMANENTLY, httplib.FOUND, httplib.SEE_OTHER):
                if max_redirects <= 0:
                    raise ValueError('Too many redirects, giving up')
                newurl = response.getheader('Location', None)
                if newurl is None:
                    raise ValueError('%s returned a redirect response with no Location header' % url)
                return get_https_resource_securely(newurl, timeout=timeout, max_redirects=max_redirects - 1, ssl_version=ssl_version)
            if response.status != httplib.OK:
                raise HTTPError(url, response.status)
            return response.read()

def extract_tarball(raw, destdir):
    if False:
        i = 10
        return i + 15
    prints('Extracting application files...')
    with open('/dev/null', 'w') as null:
        p = subprocess.Popen(list(map(encode_for_subprocess, ['tar', 'xJof', '-', '-C', destdir])), stdout=null, stdin=subprocess.PIPE, close_fds=True, preexec_fn=lambda : signal.signal(signal.SIGPIPE, signal.SIG_DFL))
        p.stdin.write(raw)
        p.stdin.close()
        if p.wait() != 0:
            prints('Extracting of application files failed with error code: %s' % p.returncode)
            raise SystemExit(1)

def get_tarball_info(version):
    if False:
        i = 10
        return i + 15
    global dl_url, signature, calibre_version
    print('Downloading tarball signature securely...')
    if version:
        sigfname = 'calibre-' + version + '-' + arch + '.txz.sha512'
        try:
            signature = get_https_resource_securely('https://code.calibre-ebook.com/signatures/' + sigfname)
        except HTTPError as err:
            if err.code != 404:
                raise
            signature = get_https_resource_securely('https://code.calibre-ebook.com/signatures/old/' + sigfname)
        calibre_version = version
        dl_url = 'https://download.calibre-ebook.com/' + version + '/calibre-' + version + '-' + arch + '.txz'
    else:
        raw = get_https_resource_securely('https://code.calibre-ebook.com/tarball-info/' + arch)
        (signature, calibre_version) = raw.rpartition(b'@')[::2]
        dl_url = 'https://calibre-ebook.com/dist/linux-' + arch
    if not signature or not calibre_version:
        raise ValueError('Failed to get install file signature, invalid signature returned')
    dl_url = os.environ.get('CALIBRE_INSTALLER_LOCAL_URL', dl_url)
    if isinstance(calibre_version, bytes):
        calibre_version = calibre_version.decode('utf-8')

def download_and_extract(destdir, version):
    if False:
        i = 10
        return i + 15
    get_tarball_info(version)
    raw = download_tarball()
    if os.path.exists(destdir):
        shutil.rmtree(destdir)
    os.makedirs(destdir)
    print('Extracting files to %s ...' % destdir)
    extract_tarball(raw, destdir)

def run_installer(install_dir, isolated, bin_dir, share_dir, version):
    if False:
        print('Hello World!')
    destdir = os.path.abspath(os.path.expanduser(install_dir or '/opt'))
    if destdir == '/usr/bin':
        prints(destdir, 'is not a valid install location. Choose', end='')
        prints('a location like /opt or /usr/local')
        return 1
    destdir = os.path.realpath(os.path.join(destdir, 'calibre'))
    if os.path.exists(destdir):
        if not os.path.isdir(destdir):
            prints(destdir, 'exists and is not a directory. Choose a location like /opt or /usr/local')
            return 1
    print('Installing to', destdir)
    download_and_extract(destdir, version)
    if not isolated:
        pi = [os.path.join(destdir, 'calibre_postinstall')]
        if bin_dir is not None:
            pi.extend(['--bindir', bin_dir])
        if share_dir is not None:
            pi.extend(['--sharedir', share_dir])
        subprocess.call(pi)
        prints('Run "calibre" to start calibre')
    else:
        prints('Run "%s/calibre" to start calibre' % destdir)
    return 0

def check_umask():
    if False:
        while True:
            i = 10
    mask = os.umask(18)
    os.umask(mask)
    forbid_user_read = mask & stat.S_IRUSR
    forbid_user_exec = mask & stat.S_IXUSR
    forbid_group_read = mask & stat.S_IRGRP
    forbid_group_exec = mask & stat.S_IXGRP
    forbid_other_read = mask & stat.S_IROTH
    forbid_other_exec = mask & stat.S_IXOTH
    if forbid_user_read or forbid_user_exec or forbid_group_read or forbid_group_exec or forbid_other_read or forbid_other_exec:
        prints('WARNING: Your current umask disallows reading of files by some users, this can cause system breakage when running the installer because of bugs in common system utilities.')
        sys.stdin = open('/dev/tty')
        while True:
            q = raw_input('Should the installer (f)ix the umask, (i)gnore it or (a)bort [f/i/a Default is abort]: ') or 'a'
            if q in 'f i a'.split():
                break
            prints('Response', q, 'not understood')
        if q == 'f':
            mask = mask & ~stat.S_IRUSR & ~stat.S_IXUSR & ~stat.S_IRGRP & ~stat.S_IXGRP & ~stat.S_IROTH & ~stat.S_IXOTH
            os.umask(mask)
            prints('umask changed to: {:03o}'.format(mask))
        elif q == 'i':
            prints('Ignoring bad umask and proceeding anyway, you have been warned!')
        else:
            raise SystemExit('The system umask is unsuitable, aborting')

def check_for_libEGL():
    if False:
        while True:
            i = 10
    import ctypes
    try:
        ctypes.CDLL('libEGL.so.1')
        return
    except Exception:
        pass
    raise SystemExit('You are missing the system library libEGL.so.1. Try installing packages such as libegl1 and libopengl0')

def check_for_libOpenGl():
    if False:
        i = 10
        return i + 15
    import ctypes
    try:
        ctypes.CDLL('libOpenGL.so.0')
        return
    except Exception:
        pass
    raise SystemExit('You are missing the system library libOpenGL.so.0. Try installing packages such as libopengl0')

def check_glibc_version(min_required=(2, 31), release_date='2020-02-01'):
    if False:
        for i in range(10):
            print('nop')
    import ctypes
    libc = ctypes.CDLL(None)
    try:
        f = libc.gnu_get_libc_version
    except AttributeError:
        raise SystemExit('Your system is not based on GNU libc. The calibre binaries require GNU libc')
    f.restype = ctypes.c_char_p
    ver = f().decode('ascii')
    q = tuple(map(int, ver.split('.')))
    if q < min_required:
        raise SystemExit('Your system has GNU libc version {}. The calibre binaries require at least version: {} (released on {}). Update your system.'.format(ver, '.'.join(map(str, min_required)), release_date))

def main(install_dir=None, isolated=False, bin_dir=None, share_dir=None, ignore_umask=False, version=None):
    if False:
        for i in range(10):
            print('nop')
    if not ignore_umask and (not isolated):
        check_umask()
    if is_linux_arm and (not is_linux_arm64) or not is64bit:
        raise SystemExit('You are running on a 32-bit system. The calibre binaries are only available for 64-bit systems. You will have to compile from source.')
    glibc_versions = {(6, 0, 0): {'min_required': (2, 31), 'release_date': '2020-02-01'}}
    if is_linux_arm64:
        glibc_versions.update({(6, 8, 0): {'min_required': (2, 34), 'release_date': '2022-02-03'}})
    q = tuple(map(int, version.split('.'))) if version else (sys.maxsize, 999, 999)
    for key in sorted(glibc_versions, reverse=True):
        if q >= key:
            check_glibc_version(**glibc_versions[key])
            break
    if q[0] >= 6:
        check_for_libEGL()
        check_for_libOpenGl()
    run_installer(install_dir, isolated, bin_dir, share_dir, version)
try:
    __file__
    from_file = True
except NameError:
    from_file = False

def update_intaller_wrapper():
    if False:
        return 10
    with open(__file__, 'rb') as f:
        src = f.read().decode('utf-8')
    wrapper = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'linux-installer.sh')
    with open(wrapper, 'r+b') as f:
        raw = f.read().decode('utf-8')
        nraw = re.sub('^# HEREDOC_START.+^# HEREDOC_END', lambda m: '# HEREDOC_START\n{}\n# HEREDOC_END'.format(src), raw, flags=re.MULTILINE | re.DOTALL)
        if 'update_intaller_wrapper()' not in nraw:
            raise SystemExit('regex substitute of HEREDOC failed')
        (f.seek(0), f.truncate())
        f.write(nraw.encode('utf-8'))

def script_launch():
    if False:
        print('Hello World!')

    def path(x):
        if False:
            for i in range(10):
                print('nop')
        return os.path.expanduser(x)

    def to_bool(x):
        if False:
            while True:
                i = 10
        return x.lower() in ('y', 'yes', '1', 'true')
    type_map = {x: path for x in 'install_dir isolated bin_dir share_dir ignore_umask version'.split()}
    type_map['isolated'] = type_map['ignore_umask'] = to_bool
    kwargs = {}
    for arg in sys.argv[1:]:
        if arg:
            m = re.match('([a-z_]+)=(.+)', arg)
            if m is None:
                raise SystemExit('Unrecognized command line argument: ' + arg)
            k = m.group(1)
            if k not in type_map:
                raise SystemExit('Unrecognized command line argument: ' + arg)
            kwargs[k] = type_map[k](m.group(2))
    main(**kwargs)
if __name__ == '__main__' and from_file:
    main()
elif __name__ == 'update_wrapper':
    update_intaller_wrapper()