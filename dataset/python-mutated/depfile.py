"""
This module contains the traversal logic and models that can be used to generate
depfiles from an environment.
"""
import os
import re
from enum import Enum
from typing import List, Optional
import spack.deptypes as dt
import spack.environment.environment as ev
import spack.spec
import spack.traverse as traverse

class UseBuildCache(Enum):
    ONLY = 1
    NEVER = 2
    AUTO = 3

    @staticmethod
    def from_string(s: str) -> 'UseBuildCache':
        if False:
            return 10
        if s == 'only':
            return UseBuildCache.ONLY
        elif s == 'never':
            return UseBuildCache.NEVER
        elif s == 'auto':
            return UseBuildCache.AUTO
        raise ValueError(f'invalid value for UseBuildCache: {s}')

def _deptypes(use_buildcache: UseBuildCache):
    if False:
        return 10
    "What edges should we follow for a given node? If it's a cache-only\n    node, then we can drop build type deps."
    return dt.LINK | dt.RUN if use_buildcache == UseBuildCache.ONLY else dt.BUILD | dt.LINK | dt.RUN

class DepfileNode:
    """Contains a spec, a subset of its dependencies, and a flag whether it should be
    buildcache only/never/auto."""

    def __init__(self, target: spack.spec.Spec, prereqs: List[spack.spec.Spec], buildcache: UseBuildCache):
        if False:
            while True:
                i = 10
        self.target = MakefileSpec(target)
        self.prereqs = list((MakefileSpec(x) for x in prereqs))
        if buildcache == UseBuildCache.ONLY:
            self.buildcache_flag = '--use-buildcache=only'
        elif buildcache == UseBuildCache.NEVER:
            self.buildcache_flag = '--use-buildcache=never'
        else:
            self.buildcache_flag = ''

class DepfileSpecVisitor:
    """This visitor produces an adjacency list of a (reduced) DAG, which
    is used to generate depfile targets with their prerequisites. Currently
    it only drops build deps when using buildcache only mode.

    Note that the DAG could be reduced even more by dropping build edges of specs
    installed at the moment the depfile is generated, but that would produce
    stateful depfiles that would not fail when the database is wiped later."""

    def __init__(self, pkg_buildcache: UseBuildCache, deps_buildcache: UseBuildCache):
        if False:
            print('Hello World!')
        self.adjacency_list: List[DepfileNode] = []
        self.pkg_buildcache = pkg_buildcache
        self.deps_buildcache = deps_buildcache
        self.depflag_root = _deptypes(pkg_buildcache)
        self.depflag_deps = _deptypes(deps_buildcache)

    def neighbors(self, node):
        if False:
            for i in range(10):
                print('nop')
        'Produce a list of spec to follow from node'
        depflag = self.depflag_root if node.depth == 0 else self.depflag_deps
        return traverse.sort_edges(node.edge.spec.edges_to_dependencies(depflag=depflag))

    def accept(self, node):
        if False:
            return 10
        self.adjacency_list.append(DepfileNode(target=node.edge.spec, prereqs=[edge.spec for edge in self.neighbors(node)], buildcache=self.pkg_buildcache if node.depth == 0 else self.deps_buildcache))
        return True

class MakefileSpec(object):
    """Limited interface to spec to help generate targets etc. without
    introducing unwanted special characters.
    """
    _pattern = None

    def __init__(self, spec):
        if False:
            print('Hello World!')
        self.spec = spec

    def safe_name(self):
        if False:
            print('Hello World!')
        return self.safe_format('{name}-{version}-{hash}')

    def spec_hash(self):
        if False:
            print('Hello World!')
        return self.spec.dag_hash()

    def safe_format(self, format_str):
        if False:
            while True:
                i = 10
        unsafe_result = self.spec.format(format_str)
        if not MakefileSpec._pattern:
            MakefileSpec._pattern = re.compile('[^A-Za-z0-9_.-]')
        return MakefileSpec._pattern.sub('_', unsafe_result)

    def unsafe_format(self, format_str):
        if False:
            for i in range(10):
                print('nop')
        return self.spec.format(format_str)

class MakefileModel:
    """This class produces all data to render a makefile for specs of an environment."""

    def __init__(self, env: ev.Environment, roots: List[spack.spec.Spec], adjacency_list: List[DepfileNode], make_prefix: Optional[str], jobserver: bool):
        if False:
            for i in range(10):
                print('nop')
        '\n        Args:\n            env: environment to generate the makefile for\n            roots: specs that get built in the default target\n            adjacency_list: list of DepfileNode, mapping specs to their dependencies\n            make_prefix: prefix for makefile targets\n            jobserver: when enabled, make will invoke Spack with jobserver support. For\n                dry-run this should be disabled.\n        '
        self.env_path = env.path
        self.roots = list((MakefileSpec(x) for x in roots))
        if make_prefix is None:
            self.make_prefix = os.path.join(env.env_subdir_path, 'makedeps')
            self.pkg_identifier_variable = 'SPACK_PACKAGE_IDS'
        else:
            self.make_prefix = make_prefix
            self.pkg_identifier_variable = os.path.join(make_prefix, 'SPACK_PACKAGE_IDS')
        self.make_adjacency_list = [(item.target.safe_name(), ' '.join((self._install_target(s.safe_name()) for s in item.prereqs)), item.target.spec_hash(), item.target.unsafe_format('{name}{@version}{%compiler}{variants}{arch=architecture}'), item.buildcache_flag) for item in adjacency_list]
        self.root_install_targets = [self._install_target(s.safe_name()) for s in self.roots]
        self.jobserver_support = '+' if jobserver else ''
        self.all_pkg_identifiers: List[str] = []
        self.all_install_related_targets: List[str] = []
        self.phony_convenience_targets: List[str] = []
        for node in adjacency_list:
            tgt = node.target.safe_name()
            self.all_pkg_identifiers.append(tgt)
            self.all_install_related_targets.append(self._install_target(tgt))
            self.all_install_related_targets.append(self._install_deps_target(tgt))
            if make_prefix is None:
                self.phony_convenience_targets.append(os.path.join('install', tgt))
                self.phony_convenience_targets.append(os.path.join('install-deps', tgt))

    def _target(self, name: str) -> str:
        if False:
            return 10
        if name in ('all', 'clean') and os.path.isabs(self.make_prefix):
            return name
        else:
            return os.path.join(self.make_prefix, name)

    def _install_target(self, name: str) -> str:
        if False:
            i = 10
            return i + 15
        return os.path.join(self.make_prefix, 'install', name)

    def _install_deps_target(self, name: str) -> str:
        if False:
            i = 10
            return i + 15
        return os.path.join(self.make_prefix, 'install-deps', name)

    def to_dict(self):
        if False:
            while True:
                i = 10
        return {'all_target': self._target('all'), 'env_target': self._target('env'), 'clean_target': self._target('clean'), 'all_install_related_targets': ' '.join(self.all_install_related_targets), 'root_install_targets': ' '.join(self.root_install_targets), 'dirs_target': self._target('dirs'), 'environment': self.env_path, 'install_target': self._target('install'), 'install_deps_target': self._target('install-deps'), 'any_hash_target': self._target('%'), 'jobserver_support': self.jobserver_support, 'adjacency_list': self.make_adjacency_list, 'phony_convenience_targets': ' '.join(self.phony_convenience_targets), 'pkg_ids_variable': self.pkg_identifier_variable, 'pkg_ids': ' '.join(self.all_pkg_identifiers)}

    @property
    def empty(self):
        if False:
            print('Hello World!')
        return len(self.roots) == 0

    @staticmethod
    def from_env(env: ev.Environment, *, filter_specs: Optional[List[spack.spec.Spec]]=None, pkg_buildcache: UseBuildCache=UseBuildCache.AUTO, dep_buildcache: UseBuildCache=UseBuildCache.AUTO, make_prefix: Optional[str]=None, jobserver: bool=True) -> 'MakefileModel':
        if False:
            while True:
                i = 10
        'Produces a MakefileModel from an environment and a list of specs.\n\n        Args:\n            env: the environment to use\n            filter_specs: if provided, only these specs will be built from the environment,\n                otherwise the environment roots are used.\n            pkg_buildcache: whether to only use the buildcache for top-level specs.\n            dep_buildcache: whether to only use the buildcache for non-top-level specs.\n            make_prefix: the prefix for the makefile targets\n            jobserver: when enabled, make will invoke Spack with jobserver support. For\n                dry-run this should be disabled.\n        '
        roots = env.all_matching_specs(*filter_specs) if filter_specs else env.concrete_roots()
        visitor = DepfileSpecVisitor(pkg_buildcache, dep_buildcache)
        traverse.traverse_breadth_first_with_visitor(roots, traverse.CoverNodesVisitor(visitor, key=lambda s: s.dag_hash()))
        return MakefileModel(env, roots, visitor.adjacency_list, make_prefix, jobserver)