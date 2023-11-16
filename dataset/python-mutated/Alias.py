"""scons.Node.Alias

Alias nodes.

This creates a hash of global Aliases (dummy targets).

"""
__revision__ = 'src/engine/SCons/Node/Alias.py bee7caf9defd6e108fc2998a2520ddb36a967691 2019-12-17 02:07:09 bdeegan'
import collections
import SCons.Errors
import SCons.Node
import SCons.Util

class AliasNameSpace(collections.UserDict):

    def Alias(self, name, **kw):
        if False:
            i = 10
            return i + 15
        if isinstance(name, SCons.Node.Alias.Alias):
            return name
        try:
            a = self[name]
        except KeyError:
            a = SCons.Node.Alias.Alias(name, **kw)
            self[name] = a
        return a

    def lookup(self, name, **kw):
        if False:
            print('Hello World!')
        try:
            return self[name]
        except KeyError:
            return None

class AliasNodeInfo(SCons.Node.NodeInfoBase):
    __slots__ = ('csig',)
    current_version_id = 2
    field_list = ['csig']

    def str_to_node(self, s):
        if False:
            print('Hello World!')
        return default_ans.Alias(s)

    def __getstate__(self):
        if False:
            while True:
                i = 10
        "\n        Return all fields that shall be pickled. Walk the slots in the class\n        hierarchy and add those to the state dictionary. If a '__dict__' slot is\n        available, copy all entries to the dictionary. Also include the version\n        id, which is fixed for all instances of a class.\n        "
        state = getattr(self, '__dict__', {}).copy()
        for obj in type(self).mro():
            for name in getattr(obj, '__slots__', ()):
                if hasattr(self, name):
                    state[name] = getattr(self, name)
        state['_version_id'] = self.current_version_id
        try:
            del state['__weakref__']
        except KeyError:
            pass
        return state

    def __setstate__(self, state):
        if False:
            i = 10
            return i + 15
        '\n        Restore the attributes from a pickled state.\n        '
        del state['_version_id']
        for (key, value) in state.items():
            if key not in ('__weakref__',):
                setattr(self, key, value)

class AliasBuildInfo(SCons.Node.BuildInfoBase):
    __slots__ = ()
    current_version_id = 2

class Alias(SCons.Node.Node):
    NodeInfo = AliasNodeInfo
    BuildInfo = AliasBuildInfo

    def __init__(self, name):
        if False:
            return 10
        SCons.Node.Node.__init__(self)
        self.name = name
        self.changed_since_last_build = 1
        self.store_info = 0

    def str_for_display(self):
        if False:
            while True:
                i = 10
        return '"' + self.__str__() + '"'

    def __str__(self):
        if False:
            while True:
                i = 10
        return self.name

    def make_ready(self):
        if False:
            print('Hello World!')
        self.get_csig()
    really_build = SCons.Node.Node.build
    is_up_to_date = SCons.Node.Node.children_are_up_to_date

    def is_under(self, dir):
        if False:
            i = 10
            return i + 15
        return 1

    def get_contents(self):
        if False:
            return 10
        'The contents of an alias is the concatenation\n        of the content signatures of all its sources.'
        childsigs = [n.get_csig() for n in self.children()]
        return ''.join(childsigs)

    def sconsign(self):
        if False:
            i = 10
            return i + 15
        'An Alias is not recorded in .sconsign files'
        pass

    def build(self):
        if False:
            while True:
                i = 10
        'A "builder" for aliases.'
        pass

    def convert(self):
        if False:
            i = 10
            return i + 15
        try:
            del self.builder
        except AttributeError:
            pass
        self.reset_executor()
        self.build = self.really_build

    def get_csig(self):
        if False:
            i = 10
            return i + 15
        "\n        Generate a node's content signature, the digested signature\n        of its content.\n\n        node - the node\n        cache - alternate node to use for the signature cache\n        returns - the content signature\n        "
        try:
            return self.ninfo.csig
        except AttributeError:
            pass
        contents = self.get_contents()
        csig = SCons.Util.MD5signature(contents)
        self.get_ninfo().csig = csig
        return csig
default_ans = AliasNameSpace()
SCons.Node.arg2nodes_lookups.append(default_ans.lookup)