from __future__ import absolute_import
from bzrlib import registry
from bzrlib.lazy_import import lazy_import
lazy_import(globals(), '\nfrom bzrlib import errors, urlutils\n')
"Provides a shorthand for referring to bugs on a variety of bug trackers.\n\n'commit --fixes' stores references to bugs as a <bug_url> -> <bug_status>\nmapping in the properties for that revision.\n\nHowever, it's inconvenient to type out full URLs for bugs on the command line,\nparticularly given that many users will be using only a single bug tracker per\nbranch.\n\nThus, this module provides a registry of types of bug tracker (e.g. Launchpad,\nTrac). Given an abbreviated name (e.g. 'lp', 'twisted') and a branch with\nconfiguration information, these tracker types can return an instance capable\nof converting bug IDs into URLs.\n"
_bugs_help = 'When making a commit, metadata about bugs fixed by that change can be\nrecorded by using the ``--fixes`` option. For each bug marked as fixed, an\nentry is included in the \'bugs\' revision property stating \'<url> <status>\'.\n(The only ``status`` value currently supported is ``fixed.``)\n\nThe ``--fixes`` option allows you to specify a bug tracker and a bug identifier\nrather than a full URL. This looks like::\n\n    bzr commit --fixes <tracker>:<id>\n\nor::\n\n    bzr commit --fixes <id>\n\nwhere "<tracker>" is an identifier for the bug tracker, and "<id>" is the\nidentifier for that bug within the bugtracker, usually the bug number.\nIf "<tracker>" is not specified the ``bugtracker`` set in the branch\nor global configuration is used.\n\nBazaar knows about a few bug trackers that have many users. If\nyou use one of these bug trackers then there is no setup required to\nuse this feature, you just need to know the tracker identifier to use.\nThese are the bugtrackers that are built in:\n\n  ============================ ============ ============\n  URL                          Abbreviation Example\n  ============================ ============ ============\n  https://bugs.launchpad.net/  lp           lp:12345\n  http://bugs.debian.org/      deb          deb:12345\n  http://bugzilla.gnome.org/   gnome        gnome:12345\n  ============================ ============ ============\n\nFor the bug trackers not listed above configuration is required.\nSupport for generating the URLs for any project using Bugzilla or Trac\nis built in, along with a template mechanism for other bugtrackers with\nsimple URL schemes. If your bug tracker can\'t be described by one\nof the schemes described below then you can write a plugin to support\nit.\n\nIf you use Bugzilla or Trac, then you only need to set a configuration\nvariable which contains the base URL of the bug tracker. These options\ncan go into ``bazaar.conf``, ``branch.conf`` or into a branch-specific\nconfiguration section in ``locations.conf``.  You can set up these values\nfor each of the projects you work on.\n\nNote: As you provide a short name for each tracker, you can specify one or\nmore bugs in one or more trackers at commit time if you wish.\n\nLaunchpad\n---------\n\nUse ``bzr commit --fixes lp:2`` to record that this commit fixes bug 2.\n\nbugzilla_<tracker>_url\n----------------------\n\nIf present, the location of the Bugzilla bug tracker referred to by\n<tracker>. This option can then be used together with ``bzr commit\n--fixes`` to mark bugs in that tracker as being fixed by that commit. For\nexample::\n\n    bugzilla_squid_url = http://bugs.squid-cache.org\n\nwould allow ``bzr commit --fixes squid:1234`` to mark Squid\'s bug 1234 as\nfixed.\n\ntrac_<tracker>_url\n------------------\n\nIf present, the location of the Trac instance referred to by\n<tracker>. This option can then be used together with ``bzr commit\n--fixes`` to mark bugs in that tracker as being fixed by that commit. For\nexample::\n\n    trac_twisted_url = http://www.twistedmatrix.com/trac\n\nwould allow ``bzr commit --fixes twisted:1234`` to mark Twisted\'s bug 1234 as\nfixed.\n\nbugtracker_<tracker>_url\n------------------------\n\nIf present, the location of a generic bug tracker instance referred to by\n<tracker>. The location must contain an ``{id}`` placeholder,\nwhich will be replaced by a specific bug ID. This option can then be used\ntogether with ``bzr commit --fixes`` to mark bugs in that tracker as being\nfixed by that commit. For example::\n\n    bugtracker_python_url = http://bugs.python.org/issue{id}\n\nwould allow ``bzr commit --fixes python:1234`` to mark bug 1234 in Python\'s\nRoundup bug tracker as fixed, or::\n\n    bugtracker_cpan_url = http://rt.cpan.org/Public/Bug/Display.html?id={id}\n\nwould allow ``bzr commit --fixes cpan:1234`` to mark bug 1234 in CPAN\'s\nRT bug tracker as fixed, or::\n\n    bugtracker_hudson_url = http://issues.hudson-ci.org/browse/{id}\n\nwould allow ``bzr commit --fixes hudson:HUDSON-1234`` to mark bug HUDSON-1234\nin Hudson\'s JIRA bug tracker as fixed.\n'

def get_bug_url(abbreviated_bugtracker_name, branch, bug_id):
    if False:
        print('Hello World!')
    "Return a URL pointing to the canonical web page of the bug identified by\n    'bug_id'.\n    "
    tracker = tracker_registry.get_tracker(abbreviated_bugtracker_name, branch)
    return tracker.get_bug_url(bug_id)

class TrackerRegistry(registry.Registry):
    """Registry of bug tracker types."""

    def get_tracker(self, abbreviated_bugtracker_name, branch):
        if False:
            return 10
        "Return the first registered tracker that understands\n        'abbreviated_bugtracker_name'.\n\n        If no such tracker is found, raise KeyError.\n        "
        for tracker_name in self.keys():
            tracker_type = self.get(tracker_name)
            tracker = tracker_type.get(abbreviated_bugtracker_name, branch)
            if tracker is not None:
                return tracker
        raise errors.UnknownBugTrackerAbbreviation(abbreviated_bugtracker_name, branch)

    def help_topic(self, topic):
        if False:
            i = 10
            return i + 15
        return _bugs_help
tracker_registry = TrackerRegistry()
'Registry of bug trackers.'

class BugTracker(object):
    """Base class for bug trackers."""

    def check_bug_id(self, bug_id):
        if False:
            for i in range(10):
                print('nop')
        'Check that the bug_id is valid.\n\n        The base implementation assumes that all bug_ids are valid.\n        '

    def get_bug_url(self, bug_id):
        if False:
            while True:
                i = 10
        'Return the URL for bug_id. Raise an error if bug ID is malformed.'
        self.check_bug_id(bug_id)
        return self._get_bug_url(bug_id)

    def _get_bug_url(self, bug_id):
        if False:
            return 10
        "Given a validated bug_id, return the bug's web page's URL."

class IntegerBugTracker(BugTracker):
    """A bug tracker that only allows integer bug IDs."""

    def check_bug_id(self, bug_id):
        if False:
            while True:
                i = 10
        try:
            int(bug_id)
        except ValueError:
            raise errors.MalformedBugIdentifier(bug_id, 'Must be an integer')

class UniqueIntegerBugTracker(IntegerBugTracker):
    """A style of bug tracker that exists in one place only, such as Launchpad.

    If you have one of these trackers then register an instance passing in an
    abbreviated name for the bug tracker and a base URL. The bug ids are
    appended directly to the URL.
    """

    def __init__(self, abbreviated_bugtracker_name, base_url):
        if False:
            while True:
                i = 10
        self.abbreviation = abbreviated_bugtracker_name
        self.base_url = base_url

    def get(self, abbreviated_bugtracker_name, branch):
        if False:
            for i in range(10):
                print('nop')
        'Returns the tracker if the abbreviation matches. Returns None\n        otherwise.'
        if abbreviated_bugtracker_name != self.abbreviation:
            return None
        return self

    def _get_bug_url(self, bug_id):
        if False:
            return 10
        'Return the URL for bug_id.'
        return self.base_url + bug_id
tracker_registry.register('launchpad', UniqueIntegerBugTracker('lp', 'https://launchpad.net/bugs/'))
tracker_registry.register('debian', UniqueIntegerBugTracker('deb', 'http://bugs.debian.org/'))
tracker_registry.register('gnome', UniqueIntegerBugTracker('gnome', 'http://bugzilla.gnome.org/show_bug.cgi?id='))

class URLParametrizedBugTracker(BugTracker):
    """A type of bug tracker that can be found on a variety of different sites,
    and thus needs to have the base URL configured.

    Looks for a config setting in the form '<type_name>_<abbreviation>_url'.
    `type_name` is the name of the type of tracker and `abbreviation`
    is a short name for the particular instance.
    """

    def get(self, abbreviation, branch):
        if False:
            return 10
        config = branch.get_config()
        url = config.get_user_option('%s_%s_url' % (self.type_name, abbreviation), expand=False)
        if url is None:
            return None
        self._base_url = url
        return self

    def __init__(self, type_name, bug_area):
        if False:
            i = 10
            return i + 15
        self.type_name = type_name
        self._bug_area = bug_area

    def _get_bug_url(self, bug_id):
        if False:
            while True:
                i = 10
        'Return a URL for a bug on this Trac instance.'
        return urlutils.join(self._base_url, self._bug_area) + str(bug_id)

class URLParametrizedIntegerBugTracker(IntegerBugTracker, URLParametrizedBugTracker):
    """A type of bug tracker that  only allows integer bug IDs.

    This can be found on a variety of different sites, and thus needs to have
    the base URL configured.

    Looks for a config setting in the form '<type_name>_<abbreviation>_url'.
    `type_name` is the name of the type of tracker (e.g. 'bugzilla' or 'trac')
    and `abbreviation` is a short name for the particular instance (e.g.
    'squid' or 'apache').
    """
tracker_registry.register('trac', URLParametrizedIntegerBugTracker('trac', 'ticket/'))
tracker_registry.register('bugzilla', URLParametrizedIntegerBugTracker('bugzilla', 'show_bug.cgi?id='))

class GenericBugTracker(URLParametrizedBugTracker):
    """Generic bug tracker specified by an URL template."""

    def __init__(self):
        if False:
            print('Hello World!')
        super(GenericBugTracker, self).__init__('bugtracker', None)

    def get(self, abbreviation, branch):
        if False:
            print('Hello World!')
        self._abbreviation = abbreviation
        return super(GenericBugTracker, self).get(abbreviation, branch)

    def _get_bug_url(self, bug_id):
        if False:
            while True:
                i = 10
        "Given a validated bug_id, return the bug's web page's URL."
        if '{id}' not in self._base_url:
            raise errors.InvalidBugTrackerURL(self._abbreviation, self._base_url)
        return self._base_url.replace('{id}', str(bug_id))
tracker_registry.register('generic', GenericBugTracker())
FIXED = 'fixed'
ALLOWED_BUG_STATUSES = set([FIXED])

def encode_fixes_bug_urls(bug_urls):
    if False:
        i = 10
        return i + 15
    "Get the revision property value for a commit that fixes bugs.\n\n    :param bug_urls: An iterable of escaped URLs to bugs. These normally\n        come from `get_bug_url`.\n    :return: A string that will be set as the 'bugs' property of a revision\n        as part of a commit.\n    "
    return '\n'.join(('%s %s' % (url, FIXED) for url in bug_urls))