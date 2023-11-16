"""Branch opening with URL-based restrictions."""
from __future__ import absolute_import
import threading
from bzrlib import errors, urlutils
from bzrlib.branch import Branch
from bzrlib.controldir import ControlDir

class BadUrl(errors.BzrError):
    _fmt = 'Tried to access a branch from bad URL %(url)s.'

class BranchReferenceForbidden(errors.BzrError):
    _fmt = 'Trying to mirror a branch reference and the branch type does not allow references.'

class BranchLoopError(errors.BzrError):
    """Encountered a branch cycle.

    A URL may point to a branch reference or it may point to a stacked branch.
    In either case, it's possible for there to be a cycle in these references,
    and this exception is raised when we detect such a cycle.
    """
    _fmt = 'Encountered a branch cycle'

class BranchOpenPolicy(object):
    """Policy on how to open branches.

    In particular, a policy determines which branches are okay to open by
    checking their URLs and deciding whether or not to follow branch
    references.
    """

    def should_follow_references(self):
        if False:
            i = 10
            return i + 15
        'Whether we traverse references when mirroring.\n\n        Subclasses must override this method.\n\n        If we encounter a branch reference and this returns false, an error is\n        raised.\n\n        :returns: A boolean to indicate whether to follow a branch reference.\n        '
        raise NotImplementedError(self.should_follow_references)

    def transform_fallback_location(self, branch, url):
        if False:
            for i in range(10):
                print('nop')
        "Validate, maybe modify, 'url' to be used as a stacked-on location.\n\n        :param branch:  The branch that is being opened.\n        :param url: The URL that the branch provides for its stacked-on\n            location.\n        :return: (new_url, check) where 'new_url' is the URL of the branch to\n            actually open and 'check' is true if 'new_url' needs to be\n            validated by check_and_follow_branch_reference.\n        "
        raise NotImplementedError(self.transform_fallback_location)

    def check_one_url(self, url):
        if False:
            return 10
        'Check a URL.\n\n        Subclasses must override this method.\n\n        :param url: The source URL to check.\n        :raise BadUrl: subclasses are expected to raise this or a subclass\n            when it finds a URL it deems to be unacceptable.\n        '
        raise NotImplementedError(self.check_one_url)

class _BlacklistPolicy(BranchOpenPolicy):
    """Branch policy that forbids certain URLs.

    This doesn't cope with various alternative spellings of URLs,
    with e.g. url encoding. It's mostly useful for tests.
    """

    def __init__(self, should_follow_references, bad_urls=None):
        if False:
            return 10
        if bad_urls is None:
            bad_urls = set()
        self._bad_urls = bad_urls
        self._should_follow_references = should_follow_references

    def should_follow_references(self):
        if False:
            i = 10
            return i + 15
        return self._should_follow_references

    def check_one_url(self, url):
        if False:
            print('Hello World!')
        if url in self._bad_urls:
            raise BadUrl(url)

    def transform_fallback_location(self, branch, url):
        if False:
            while True:
                i = 10
        "See `BranchOpenPolicy.transform_fallback_location`.\n\n        This class is not used for testing our smarter stacking features so we\n        just do the simplest thing: return the URL that would be used anyway\n        and don't check it.\n        "
        return (urlutils.join(branch.base, url), False)

class AcceptAnythingPolicy(_BlacklistPolicy):
    """Accept anything, to make testing easier."""

    def __init__(self):
        if False:
            return 10
        super(AcceptAnythingPolicy, self).__init__(True, set())

class WhitelistPolicy(BranchOpenPolicy):
    """Branch policy that only allows certain URLs."""

    def __init__(self, should_follow_references, allowed_urls=None, check=False):
        if False:
            return 10
        if allowed_urls is None:
            allowed_urls = []
        self.allowed_urls = set((url.rstrip('/') for url in allowed_urls))
        self.check = check

    def should_follow_references(self):
        if False:
            while True:
                i = 10
        return self._should_follow_references

    def check_one_url(self, url):
        if False:
            return 10
        if url.rstrip('/') not in self.allowed_urls:
            raise BadUrl(url)

    def transform_fallback_location(self, branch, url):
        if False:
            while True:
                i = 10
        'See `BranchOpenPolicy.transform_fallback_location`.\n\n        Here we return the URL that would be used anyway and optionally check\n        it.\n        '
        return (urlutils.join(branch.base, url), self.check)

class SingleSchemePolicy(BranchOpenPolicy):
    """Branch open policy that rejects URLs not on the given scheme."""

    def __init__(self, allowed_scheme):
        if False:
            while True:
                i = 10
        self.allowed_scheme = allowed_scheme

    def should_follow_references(self):
        if False:
            print('Hello World!')
        return True

    def transform_fallback_location(self, branch, url):
        if False:
            while True:
                i = 10
        return (urlutils.join(branch.base, url), True)

    def check_one_url(self, url):
        if False:
            for i in range(10):
                print('nop')
        'Check that `url` is okay to open.'
        if urlutils.URL.from_string(str(url)).scheme != self.allowed_scheme:
            raise BadUrl(url)

class BranchOpener(object):
    """Branch opener which uses a URL policy.

    All locations that are opened (stacked-on branches, references) are
    checked against a policy object.

    The policy object is expected to have the following methods:
    * check_one_url 
    * should_follow_references
    * transform_fallback_location
    """
    _threading_data = threading.local()

    def __init__(self, policy, probers=None):
        if False:
            print('Hello World!')
        'Create a new BranchOpener.\n\n        :param policy: The opener policy to use.\n        :param probers: Optional list of probers to allow.\n            Defaults to local and remote bzr probers.\n        '
        self.policy = policy
        self._seen_urls = set()
        self.probers = probers

    @classmethod
    def install_hook(cls):
        if False:
            for i in range(10):
                print('nop')
        "Install the ``transform_fallback_location`` hook.\n\n        This is done at module import time, but transform_fallback_locationHook\n        doesn't do anything unless the `_active_openers` threading.Local\n        object has a 'opener' attribute in this thread.\n\n        This is in a module-level function rather than performed at module\n        level so that it can be called in setUp for testing `BranchOpener`\n        as bzrlib.tests.TestCase.setUp clears hooks.\n        "
        Branch.hooks.install_named_hook('transform_fallback_location', cls.transform_fallback_locationHook, 'BranchOpener.transform_fallback_locationHook')

    def check_and_follow_branch_reference(self, url):
        if False:
            return 10
        "Check URL (and possibly the referenced URL).\n\n        This method checks that `url` passes the policy's `check_one_url`\n        method, and if `url` refers to a branch reference, it checks whether\n        references are allowed and whether the reference's URL passes muster\n        also -- recursively, until a real branch is found.\n\n        :param url: URL to check\n        :raise BranchLoopError: If the branch references form a loop.\n        :raise BranchReferenceForbidden: If this opener forbids branch\n            references.\n        "
        while True:
            if url in self._seen_urls:
                raise BranchLoopError()
            self._seen_urls.add(url)
            self.policy.check_one_url(url)
            next_url = self.follow_reference(url)
            if next_url is None:
                return url
            url = next_url
            if not self.policy.should_follow_references():
                raise BranchReferenceForbidden(url)

    @classmethod
    def transform_fallback_locationHook(cls, branch, url):
        if False:
            print('Hello World!')
        "Installed as the 'transform_fallback_location' Branch hook.\n\n        This method calls `transform_fallback_location` on the policy object and\n        either returns the url it provides or passes it back to\n        check_and_follow_branch_reference.\n        "
        try:
            opener = getattr(cls._threading_data, 'opener')
        except AttributeError:
            return url
        (new_url, check) = opener.policy.transform_fallback_location(branch, url)
        if check:
            return opener.check_and_follow_branch_reference(new_url)
        else:
            return new_url

    def run_with_transform_fallback_location_hook_installed(self, callable, *args, **kw):
        if False:
            print('Hello World!')
        if self.transform_fallback_locationHook not in Branch.hooks['transform_fallback_location']:
            raise AssertionError('hook not installed')
        self._threading_data.opener = self
        try:
            return callable(*args, **kw)
        finally:
            del self._threading_data.opener
            self._seen_urls = set()

    def follow_reference(self, url):
        if False:
            while True:
                i = 10
        'Get the branch-reference value at the specified url.\n\n        This exists as a separate method only to be overriden in unit tests.\n        '
        bzrdir = ControlDir.open(url, probers=self.probers)
        return bzrdir.get_branch_reference()

    def open(self, url):
        if False:
            print('Hello World!')
        "Open the Bazaar branch at url, first checking it.\n\n        What is acceptable means is defined by the policy's `follow_reference` and\n        `check_one_url` methods.\n        "
        if type(url) != str:
            raise TypeError
        url = self.check_and_follow_branch_reference(url)

        def open_branch(url):
            if False:
                while True:
                    i = 10
            dir = ControlDir.open(url, probers=self.probers)
            return dir.open_branch()
        return self.run_with_transform_fallback_location_hook_installed(open_branch, url)

def open_only_scheme(allowed_scheme, url):
    if False:
        for i in range(10):
            print('nop')
    'Open the branch at `url`, only accessing URLs on `allowed_scheme`.\n\n    :raises BadUrl: An attempt was made to open a URL that was not on\n        `allowed_scheme`.\n    '
    return BranchOpener(SingleSchemePolicy(allowed_scheme)).open(url)
BranchOpener.install_hook()