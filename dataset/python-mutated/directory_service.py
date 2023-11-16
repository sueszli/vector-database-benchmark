"""Directory service registration and usage.

Directory services are utilities that provide a mapping from URL-like strings
to true URLs.  Examples include lp:urls and per-user location aliases.
"""
from __future__ import absolute_import
from bzrlib import errors, registry
from bzrlib.lazy_import import lazy_import
lazy_import(globals(), '\nfrom bzrlib import (\n    branch as _mod_branch,\n    controldir as _mod_controldir,\n    urlutils,\n    )\n')

class DirectoryServiceRegistry(registry.Registry):
    """This object maintains and uses a list of directory services.

    Directory services may be registered via the standard Registry methods.
    They will be invoked if their key is a prefix of the supplied URL.

    Each item registered should be a factory of objects that provide a look_up
    method, as invoked by dereference.  Specifically, look_up should accept a
    name and URL, and return a URL.
    """

    def dereference(self, url):
        if False:
            for i in range(10):
                print('nop')
        'Dereference a supplied URL if possible.\n\n        URLs that match a registered directory service prefix are looked up in\n        it.  Non-matching urls are returned verbatim.\n\n        This is applied only once; the resulting URL must not be one that\n        requires further dereferencing.\n\n        :param url: The URL to dereference\n        :return: The dereferenced URL if applicable, the input URL otherwise.\n        '
        match = self.get_prefix(url)
        if match is None:
            return url
        (service, name) = match
        return service().look_up(name, url)
directories = DirectoryServiceRegistry()

class AliasDirectory(object):
    """Directory lookup for locations associated with a branch.

    :parent, :submit, :public, :push, :this, and :bound are currently
    supported.  On error, a subclass of DirectoryLookupFailure will be raised.
    """
    branch_aliases = registry.Registry()
    branch_aliases.register('parent', lambda b: b.get_parent(), help='The parent of this branch.')
    branch_aliases.register('submit', lambda b: b.get_submit_branch(), help='The submit branch for this branch.')
    branch_aliases.register('public', lambda b: b.get_public_branch(), help='The public location of this branch.')
    branch_aliases.register('bound', lambda b: b.get_bound_location(), help='The branch this branch is bound to, for bound branches.')
    branch_aliases.register('push', lambda b: b.get_push_location(), help='The saved location used for `bzr push` with no arguments.')
    branch_aliases.register('this', lambda b: b.base, help='This branch.')

    def look_up(self, name, url):
        if False:
            while True:
                i = 10
        branch = _mod_branch.Branch.open_containing('.')[0]
        parts = url.split('/', 1)
        if len(parts) == 2:
            (name, extra) = parts
        else:
            (name,) = parts
            extra = None
        try:
            method = self.branch_aliases.get(name[1:])
        except KeyError:
            raise errors.InvalidLocationAlias(url)
        else:
            result = method(branch)
        if result is None:
            raise errors.UnsetLocationAlias(url)
        if extra is not None:
            result = urlutils.join(result, extra)
        return result

    @classmethod
    def help_text(cls, topic):
        if False:
            print('Hello World!')
        alias_lines = []
        for key in cls.branch_aliases.keys():
            help = cls.branch_aliases.get_help(key)
            alias_lines.append('  :%-10s%s\n' % (key, help))
        return 'Location aliases\n================\n\nBazaar defines several aliases for locations associated with a branch.  These\ncan be used with most commands that expect a location, such as `bzr push`.\n\nThe aliases are::\n\n%s\nFor example, to push to the parent location::\n\n    bzr push :parent\n' % ''.join(alias_lines)
directories.register(':', AliasDirectory, 'Easy access to remembered branch locations')

class ColocatedDirectory(object):
    """Directory lookup for colocated branches.

    co:somename will resolve to the colocated branch with "somename" in
    the current directory.
    """

    def look_up(self, name, url):
        if False:
            while True:
                i = 10
        dir = _mod_controldir.ControlDir.open_containing('.')[0]
        return urlutils.join_segment_parameters(dir.user_url, {'branch': urlutils.escape(name)})
directories.register('co:', ColocatedDirectory, 'Easy access to colocated branches')