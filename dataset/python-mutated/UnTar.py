import os
import sys
import tarfile
from pyload.plugins.base.extractor import ArchiveError, BaseExtractor, CRCError

def _safe_extractall(tar, path='.', members=None, *, numeric_owner=False):
    if False:
        return 10

    def _is_within_directory(directory, target):
        if False:
            for i in range(10):
                print('nop')
        abs_directory = os.path.abspath(directory)
        abs_target = os.path.abspath(target)
        prefix = os.path.commonprefix([abs_directory, abs_target])
        return prefix == abs_directory
    for member in tar.getmembers():
        member_path = os.path.join(path, member.name)
        if not _is_within_directory(path, member_path):
            raise ArchiveError('Attempted Path Traversal in Tar File (CVE-2007-4559)')
    tar.extractall(path, members, numeric_owner=numeric_owner)

class UnTar(BaseExtractor):
    __name__ = 'UnTar'
    __type__ = 'extractor'
    __version__ = '0.06'
    __status__ = 'stable'
    __description__ = 'TAR extractor plugin'
    __license__ = 'GPLv3'
    __authors__ = [('Walter Purcaro', 'vuolter@gmail.com')]
    VERSION = '{}.{}.{}'.format(sys.version_info[0], sys.version_info[1], sys.version_info[2])

    @classmethod
    def archivetype(cls, filename):
        if False:
            i = 10
            return i + 15
        return 'tar' if cls.isarchive(filename) else None

    @classmethod
    def isarchive(cls, filename):
        if False:
            while True:
                i = 10
        try:
            return tarfile.is_tarfile(os.fsdecode(filename))
        except IOError:
            return False

    @classmethod
    def find(cls):
        if False:
            print('Hello World!')
        return sys.version_info[:2] >= (2, 5)

    def list(self, password=None):
        if False:
            print('Hello World!')
        with tarfile.open(self.filename) as t:
            self.files = [os.path.join(self.dest, _f) for _f in t.getnames()]
        return self.files

    def verify(self, password=None):
        if False:
            print('Hello World!')
        try:
            t = tarfile.open(self.filename, errorlevel=1)
        except tarfile.CompressionError as exc:
            raise CRCError(exc)
        except (OSError, tarfile.TarError) as exc:
            raise ArchiveError(exc)
        else:
            t.close()

    def extract(self, password=None):
        if False:
            print('Hello World!')
        self.verify(password)
        try:
            with tarfile.open(self.filename, errorlevel=2) as t:
                _safe_extractall(t, self.dest)
                self.files = t.getnames()
            return self.files
        except tarfile.ExtractError as exc:
            self.log_warning(exc)
        except tarfile.CompressionError as exc:
            raise CRCError(exc)
        except (OSError, tarfile.TarError) as exc:
            raise ArchiveError(exc)