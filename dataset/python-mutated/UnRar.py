import os
import re
import subprocess
from pyload import PKGDIR
from pyload.core.utils.convert import to_str
from pyload.plugins.base.extractor import ArchiveError, BaseExtractor, CRCError, PasswordError
from pyload.plugins.helpers import renice

class UnRar(BaseExtractor):
    __name__ = 'UnRar'
    __type__ = 'extractor'
    __version__ = '1.47'
    __status__ = 'testing'
    __config__ = [('ignore_warnings', 'bool', 'Ignore unrar warnings', False)]
    __description__ = 'RAR extractor plugin'
    __license__ = 'GPLv3'
    __authors__ = [('RaNaN', 'RaNaN@pyload.net'), ('Walter Purcaro', 'vuolter@gmail.com'), ('Immenz', 'immenz@gmx.net'), ('GammaCode', 'nitzo2001[AT]yahoo[DOT]com')]
    CMD = 'unrar'
    EXTENSIONS = ['rar', 'cab', 'arj', 'lzh', 'tar', 'gz', 'ace', 'uue', 'bz2', 'jar', 'iso', 'xz', 'z']
    _RE_PART = re.compile('\\.(part|r)\\d+(\\.rar|\\.rev)?(\\.bad)?|\\.rar$', re.I)
    _RE_FIXNAME = re.compile('Building (.+)')
    _RE_FILES_V4 = re.compile('^([* ])(.+?)\\s+(\\d+)\\s+(\\d+)\\s+(\\d+%|-->|<--)\\s+([\\d-]+)\\s+([\\d:]+)\\s*([ACHIRS.rw\\-]+)\\s+([0-9A-F]{8})\\s+(\\w+)\\s+([\\d.]+)', re.M)
    _RE_FILES_V5 = re.compile('^([* ])\\s*([ACHIRS.rw\\-]+)\\s+(\\d+)(?:\\s+\\d+)?(?:\\s+(?:\\d+%|-->|<--))?\\s+([\\d-]+)\\s+([\\d:]+)(?:\\s+[0-9A-F]{8})?\\s+(.+)', re.M)
    _RE_ENCRYPTED_HEADER = re.compile('\\s0 files')
    _RE_BADPWD = re.compile('password', re.I)
    _RE_BADCRC = re.compile('encrypted|damaged|CRC failed|checksum error|corrupt', re.I)
    _RE_VERSION = re.compile('(?:UN)?RAR\\s(\\d+\\.\\d+)', re.I)

    @classmethod
    def find(cls):
        if False:
            print('Hello World!')
        try:
            if os.name == 'nt':
                cls.CMD = os.path.join(PKGDIR, 'lib', 'RAR.exe')
            else:
                cls.CMD = 'rar'
            p = subprocess.Popen([cls.CMD], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            (out, err) = (to_str(r).strip() if r else '' for r in p.communicate())
            cls.REPAIR = True
        except OSError:
            try:
                if os.name == 'nt':
                    cls.CMD = os.path.join(PKGDIR, 'lib', 'UnRAR.exe')
                else:
                    cls.CMD = 'unrar'
                p = subprocess.Popen([cls.CMD], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                (out, err) = (to_str(r).strip() if r else '' for r in p.communicate())
            except OSError:
                return False
        m = cls._RE_VERSION.search(out)
        if m is not None:
            cls.VERSION = m.group(1)
            cls._RE_FILES = cls._RE_FILES_V4 if float(cls.VERSION) < 5 else cls._RE_FILES_V5
            return True
        else:
            return False

    @classmethod
    def ismultipart(cls, filename):
        if False:
            print('Hello World!')
        return cls._RE_PART.search(filename) is not None

    def init(self):
        if False:
            while True:
                i = 10
        self.smallest = None
        self.archive_encryption = None

    def verify(self, password=None):
        if False:
            i = 10
            return i + 15
        (encrypted_header, encrypted_files) = self._check_archive_encryption()
        if encrypted_header:
            p = self.call_cmd('l', '-v', self.filename, password=password)
            (out, err) = (_r.strip() if _r else '' for _r in p.communicate())
            if self._RE_ENCRYPTED_HEADER.search(out):
                raise PasswordError
        elif encrypted_files:
            smallest = self._find_smallest_file(password=password)[0]
            if smallest is None:
                raise ArchiveError('Cannot find smallest file')
            try:
                extracted = os.path.join(self.dest, smallest if self.fullpath else os.path.basename(smallest))
                try:
                    os.remove(extracted)
                except OSError:
                    pass
                self.extract(password=password, file=smallest)
                if smallest not in self.excludefiles:
                    self.excludefiles.append(smallest)
            except (PasswordError, CRCError, ArchiveError) as ex:
                try:
                    os.remove(extracted)
                except OSError:
                    pass
                raise ex

    def repair(self):
        if False:
            print('Hello World!')
        p = self.call_cmd('rc', self.filename)
        self.progress(p)
        (out, err) = (to_str(r).strip() if r else '' for r in p.communicate())
        if err or p.returncode:
            p = self.call_cmd('r', self.filename)
            self.progress(p)
            (out, err) = (to_str(r).strip() if r else '' for r in p.communicate())
            if err or p.returncode:
                return False
            else:
                dir = os.path.dirname(self.filename)
                name = self._RE_FIXNAME.search(out).group(1)
                self.filename = os.path.join(dir, name)
        return True

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
        p = self.call_cmd(command, self.filename, file, self.dest, password=password)
        self.progress(p)
        (out, err) = (to_str(r).strip() if r else '' for r in p.communicate())
        if err:
            if self._RE_BADPWD.search(err):
                raise PasswordError
            elif self._RE_BADCRC.search(err):
                raise CRCError(err)
            elif self.config.get('ignore_warnings', False) and err.startswith('WARNING:'):
                pass
            else:
                raise ArchiveError(err)
        if p.returncode and p.returncode != 10:
            raise ArchiveError(self._('Process return code: {}').format(p.returncode))
        return self.list(password)

    def chunks(self):
        if False:
            for i in range(10):
                print('nop')
        files = []
        (dir, name) = os.path.split(self.filename)
        files.extend((os.path.join(dir, os.path.basename(_f)) for _f in filter(self.ismultipart, os.listdir(dir)) if self._RE_PART.sub('', name) == self._RE_PART.sub('', _f)))
        if self.filename not in files:
            files.append(self.filename)
        return files

    def list(self, password=None):
        if False:
            print('Hello World!')
        if not self.files:
            self._find_smallest_file(password=password)
        return self.files

    def call_cmd(self, command, *xargs, **kwargs):
        if False:
            print('Hello World!')
        args = []
        if float(self.VERSION) >= 5.5:
            args.append('-scf')
        if self.overwrite:
            args.append('-o+')
        else:
            args.append('-o-')
            args.append('-or')
        for word in self.excludefiles:
            args.append('-x{}'.format(word.strip()))
        args.append('-y')
        args.append('-c-')
        password = kwargs.get('password')
        if password:
            args.append('-p{}'.format(password))
        else:
            args.append('-p-')
        if self.keepbroken:
            args.append('-kb')
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
            p = self.call_cmd('l', '-v', self.filename)
            (out, err) = (_r.strip() if _r else '' for _r in p.communicate())
            encrypted_header = self._RE_ENCRYPTED_HEADER.search(out) is not None
            encrypted_files = any((m.group(1) == '*' for m in self._RE_FILES.finditer(out)))
            self.archive_encryption = (encrypted_header, encrypted_files)
        return self.archive_encryption

    def _find_smallest_file(self, password=None):
        if False:
            return 10
        if not self.smallest:
            command = 'v' if self.fullpath else 'l'
            p = self.call_cmd(command, '-v', self.filename, password=password)
            (out, err) = (_r.strip() if _r else '' for _r in p.communicate())
            if 'Cannot open' in err:
                raise ArchiveError(_('Cannot open file'))
            if err:
                self.log_error(err)
            smallest = (None, 0)
            files = set()
            f_grp = 5 if float(self.VERSION) >= 5 else 1
            for groups in self._RE_FILES.findall(out):
                s = int(groups[2])
                f = groups[f_grp].strip()
                if smallest[1] == 0 or smallest[1] > s > 0:
                    smallest = (f, s)
                if not self.fullpath:
                    f = os.path.basename(f)
                f = os.path.join(self.dest, f)
                files.add(f)
            self.smallest = smallest
            self.files = list(files)
        return self.smallest