import os
import re
import subprocess
from pyload import PKGDIR
from pyload.core.utils.convert import to_str
from pyload.plugins.base.extractor import ArchiveError, BaseExtractor, CRCError, PasswordError
from pyload.plugins.helpers import renice

class SevenZip(BaseExtractor):
    __name__ = 'SevenZip'
    __type__ = 'extractor'
    __version__ = '0.39'
    __status__ = 'testing'
    __description__ = '7-Zip extractor plugin'
    __license__ = 'GPLv3'
    __authors__ = [('Walter Purcaro', 'vuolter@gmail.com'), ('Michael Nowak', None), ('GammaC0de', 'nitzo2001[AT]yahoo[DOT]com')]
    CMD = '7z'
    EXTENSIONS = [('7z', '7z(?:\\.\\d{3})?'), 'xz', 'gz', 'gzip', 'tgz', 'bz2', 'bzip2', 'tbz2', 'tbz', 'tar', 'wim', 'swm', 'lzma', 'rar', 'cab', 'arj', 'z', 'taz', 'cpio', 'rpm', 'deb', 'lzh', 'lha', 'chm', 'chw', 'hxs', 'iso', 'msi', 'doc', 'xls', 'ppt', 'dmg', 'xar', 'hfs', 'exe', 'ntfs', 'fat', 'vhd', 'mbr', 'squashfs', 'cramfs', 'scap']
    _RE_PART = re.compile('\\.7z\\.\\d{3}|\\.(part|r)\\d+(\\.rar|\\.rev)?(\\.bad)?|\\.rar$', re.I)
    _RE_FILES = re.compile('([\\d\\-]+)\\s+([\\d:]+)\\s+([RHSA.]+)\\s+(\\d+)\\s+(?:(\\d+)\\s+)?(.+)')
    _RE_ENCRYPTED_HEADER = re.compile('encrypted archive')
    _RE_ENCRYPTED_FILES = re.compile('Encrypted\\s+=\\s+\\+')
    _RE_BADPWD = re.compile('Wrong password', re.I)
    _RE_BADCRC = re.compile('CRC Failed|Can not open file', re.I)
    _RE_VERSION = re.compile('7-Zip\\s(?:\\(\\w+\\)\\s)?(?:\\[(?:32|64)\\]\\s)?(\\d+\\.\\d+)', re.I)

    @classmethod
    def find(cls):
        if False:
            i = 10
            return i + 15
        try:
            if os.name == 'nt':
                cls.CMD = os.path.join(PKGDIR, 'lib', '7z.exe')
            p = subprocess.Popen([cls.CMD], stdout=subprocess.PIPE, stderr=subprocess.PIPE, encoding='utf-8')
            (out, err) = (r.strip() if r else '' for r in p.communicate())
        except OSError:
            return False
        else:
            m = cls._RE_VERSION.search(out)
            if m is not None:
                cls.VERSION = m.group(1)
            return True

    @classmethod
    def ismultipart(cls, filename):
        if False:
            print('Hello World!')
        return cls._RE_PART.search(filename) is not None

    def init(self):
        if False:
            for i in range(10):
                print('nop')
        self.smallest = None
        self.archive_encryption = None

    def verify(self, password=None):
        if False:
            for i in range(10):
                print('nop')
        (encrypted_header, encrypted_files) = self._check_archive_encryption()
        if encrypted_header:
            p = self.call_cmd('l', '-slt', self.filename, password=password)
            (out, err) = (r.strip() if r else '' for r in p.communicate())
            if err:
                if self._RE_ENCRYPTED_HEADER.search(err):
                    raise PasswordError
                else:
                    raise ArchiveError(err)
        elif encrypted_files:
            smallest = self._find_smallest_file(password=password)[0]
            if smallest is None:
                raise ArchiveError('Cannot find smallest file')
            try:
                extracted = os.path.join(self.dest, smallest if self.fullpath else os.path.basename(smallest))
                try:
                    os.remove(extracted)
                except OSError as exc:
                    pass
                self.extract(password=password, file=smallest)
                if smallest not in self.excludefiles:
                    self.excludefiles.append(smallest)
            except (PasswordError, CRCError, ArchiveError) as exc:
                try:
                    os.remove(extracted)
                except OSError as exc:
                    pass
                raise exc

    def progress(self, process):
        if False:
            while True:
                i = 10
        s = ''
        while True:
            c = process.stdout.read(1)
            if not c:
                break
            if c == '%' and s:
                self.pyfile.set_progress(int(s))
                s = ''
            elif not c.isdigit():
                s = ''
            else:
                s += c

    def extract(self, password=None, file=None):
        if False:
            i = 10
            return i + 15
        command = 'x' if self.fullpath else 'e'
        p = self.call_cmd(command, '-o' + self.dest, self.filename, file, password=password)
        self.progress(p)
        (out, err) = (r.strip() if r else '' for r in p.communicate())
        if err:
            if self._RE_BADPWD.search(err):
                raise PasswordError
            elif self._RE_BADCRC.search(err):
                raise CRCError(err)
            else:
                raise ArchiveError(err)
        if p.returncode > 1:
            raise ArchiveError(self._('Process return code: {}').format(p.returncode))

    def chunks(self):
        if False:
            print('Hello World!')
        files = []
        (dir, name) = os.path.split(self.filename)
        files.extend((os.path.join(dir, os.path.basename(_f)) for _f in filter(self.ismultipart, os.listdir(dir)) if self._RE_PART.sub('', name) == self._RE_PART.sub('', _f)))
        if self.filename not in files:
            files.append(self.filename)
        return files

    def list(self, password=None):
        if False:
            return 10
        if not self.files:
            self._find_smallest_file(password=password)
        return self.files

    def call_cmd(self, command, *xargs, **kwargs):
        if False:
            while True:
                i = 10
        args = []
        args.append('-scsUTF-8')
        args.append('-sccUTF-8')
        if self.VERSION and float(self.VERSION) >= 15.08:
            args.append('-bso0')
            args.append('-bsp1')
        if self.overwrite:
            if self.VERSION and float(self.VERSION) >= 15.08:
                args.append('-aoa')
            else:
                args.append('-y')
        elif self.VERSION and float(self.VERSION) >= 15.08:
            args.append('-aos')
        for word in self.excludefiles:
            args.append('-xr!{}'.format(word.strip()))
        password = kwargs.get('password')
        if password:
            args.append('-p{}'.format(password))
        else:
            args.append('-p-')
        call = [self.CMD, command] + args + [arg for arg in xargs if arg]
        self.log_debug('EXECUTE ' + ' '.join(call))
        call = [to_str(cmd) for cmd in call]
        p = subprocess.Popen(call, stdout=subprocess.PIPE, stderr=subprocess.PIPE, encoding='utf-8')
        renice(p.pid, self.priority)
        return p

    def _check_archive_encryption(self):
        if False:
            for i in range(10):
                print('nop')
        if self.archive_encryption is None:
            p = self.call_cmd('l', '-slt', self.filename)
            (out, err) = (r.strip() if r else '' for r in p.communicate())
            encrypted_header = self._RE_ENCRYPTED_HEADER.search(err) is not None
            encrypted_files = self._RE_ENCRYPTED_FILES.search(out) is not None
            self.archive_encryption = (encrypted_header, encrypted_files)
        return self.archive_encryption

    def _find_smallest_file(self, password=None):
        if False:
            for i in range(10):
                print('nop')
        if not self.smallest:
            p = self.call_cmd('l', self.filename, password=password)
            (out, err) = (r.strip() if r else '' for r in p.communicate())
            if any((e in err for e in ('Can not open', 'cannot find the file'))):
                raise ArchiveError(self._('Cannot open file'))
            if p.returncode > 1:
                raise ArchiveError(self._('Process return code: {}').format(p.returncode))
            smallest = (None, 0)
            files = set()
            for groups in self._RE_FILES.findall(out):
                s = int(groups[3])
                f = groups[-1].strip()
                if smallest[1] == 0 or smallest[1] > s > 0:
                    smallest = (f, s)
                if not self.fullpath:
                    f = os.path.basename(f)
                f = os.path.join(self.dest, f)
                files.add(f)
            self.smallest = smallest
            self.files = list(files)
        return self.smallest