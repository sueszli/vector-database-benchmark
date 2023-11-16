""" Search modes for Nuitka's test runner.

The test runner can handle found errors, skip tests, etc. with search
modes, which are implemented here.
"""
import os
import sys
from nuitka.__past__ import md5
from nuitka.utils.FileOperations import areSamePaths, getFileContents, putTextFileContents

class SearchModeBase(object):

    def __init__(self):
        if False:
            while True:
                i = 10
        self.may_fail = []

    def consider(self, dirname, filename):
        if False:
            while True:
                i = 10
        return True

    def finish(self):
        if False:
            i = 10
            return i + 15
        pass

    def abortOnFinding(self, dirname, filename):
        if False:
            while True:
                i = 10
        for candidate in self.may_fail:
            if self._match(dirname, filename, candidate):
                return False
        return True

    def getExtraFlags(self, dirname, filename):
        if False:
            return 10
        return []

    def mayFailFor(self, *names):
        if False:
            i = 10
            return i + 15
        self.may_fail += names

    @classmethod
    def _match(cls, dirname, filename, candidate):
        if False:
            return 10
        from .Common import getStartDir
        parts = [dirname, filename]
        while None in parts:
            parts.remove(None)
        assert parts
        path = os.path.join(*parts)
        candidates = (dirname, filename, filename.rsplit('.', 1)[0], filename.rsplit('.', 1)[0].replace('Test', ''), path, path.rsplit('.', 1)[0], path.rsplit('.', 1)[0].replace('Test', ''))
        return candidate.rstrip('/') in candidates or areSamePaths(os.path.join(getStartDir(), candidate), filename)

    def exit(self, message):
        if False:
            print('Hello World!')
        sys.exit(message)

    def isCoverage(self):
        if False:
            i = 10
            return i + 15
        return False

    def onErrorDetected(self, message):
        if False:
            while True:
                i = 10
        self.exit(message)

class SearchModeImmediate(SearchModeBase):
    pass

class SearchModeByPattern(SearchModeBase):

    def __init__(self, start_at):
        if False:
            i = 10
            return i + 15
        SearchModeBase.__init__(self)
        self.active = False
        self.start_at = start_at

    def consider(self, dirname, filename):
        if False:
            return 10
        if self.start_at is None:
            self.active = True
        if self.active:
            return True
        self.active = self._match(dirname, filename, self.start_at)
        return self.active

    def finish(self):
        if False:
            return 10
        if not self.active:
            sys.exit('Error, became never active.')

class SearchModeResume(SearchModeBase):

    def __init__(self, tests_path):
        if False:
            for i in range(10):
                print('nop')
        SearchModeBase.__init__(self)
        tests_path = os.path.normcase(os.path.abspath(tests_path))
        version = sys.version
        if str is not bytes:
            tests_path = tests_path.encode('utf8')
            version = version.encode('utf8')
        case_hash = md5(tests_path)
        case_hash.update(version)
        from .Common import getTestingCacheDir
        cache_filename = os.path.join(getTestingCacheDir(), case_hash.hexdigest())
        self.cache_filename = cache_filename
        if os.path.exists(cache_filename):
            self.resume_from = getFileContents(cache_filename) or None
        else:
            self.resume_from = None
        self.active = not self.resume_from

    def consider(self, dirname, filename):
        if False:
            return 10
        parts = [dirname, filename]
        while None in parts:
            parts.remove(None)
        assert parts
        path = os.path.join(*parts)
        if self.active:
            putTextFileContents(self.cache_filename, contents=path)
            return True
        if areSamePaths(path, self.resume_from):
            self.active = True
        return self.active

    def finish(self):
        if False:
            print('Hello World!')
        os.unlink(self.cache_filename)
        if not self.active:
            sys.exit('Error, became never active, restarting next time.')

class SearchModeCoverage(SearchModeByPattern):

    def getExtraFlags(self, dirname, filename):
        if False:
            i = 10
            return i + 15
        return ['coverage']

    def isCoverage(self):
        if False:
            while True:
                i = 10
        return True

class SearchModeOnly(SearchModeByPattern):

    def __init__(self, start_at):
        if False:
            return 10
        SearchModeByPattern.__init__(self, start_at=start_at)
        self.done = False

    def consider(self, dirname, filename):
        if False:
            i = 10
            return i + 15
        if self.done:
            return False
        else:
            active = SearchModeByPattern.consider(self, dirname=dirname, filename=filename)
            if active:
                self.done = True
            return active