import functools as ft
import itertools
import os
import re
import shutil
import stat
import sys
from typing import Optional
from llnl.util import tty
from llnl.util.filesystem import mkdirp, remove_dead_links, remove_empty_directories, visit_directory_tree
from llnl.util.lang import index_by, match_predicate
from llnl.util.link_tree import ConflictingSpecsError, DestinationMergeVisitor, LinkTree, MergeConflictSummary, SingleMergeConflictError, SourceMergeVisitor
from llnl.util.symlink import symlink
from llnl.util.tty.color import colorize
import spack.config
import spack.projections
import spack.relocate
import spack.schema.projections
import spack.spec
import spack.store
import spack.util.spack_json as s_json
import spack.util.spack_yaml as s_yaml
from spack.error import SpackError
from spack.hooks import sbang
__all__ = ['FilesystemView', 'YamlFilesystemView']
_projections_path = '.spack/projections.yaml'

def view_symlink(src, dst, **kwargs):
    if False:
        i = 10
        return i + 15
    symlink(src, dst)

def view_hardlink(src, dst, **kwargs):
    if False:
        while True:
            i = 10
    os.link(src, dst)

def view_copy(src: str, dst: str, view, spec: Optional[spack.spec.Spec]=None):
    if False:
        while True:
            i = 10
    '\n    Copy a file from src to dst.\n\n    Use spec and view to generate relocations\n    '
    shutil.copy2(src, dst, follow_symlinks=False)
    if not spec or spec.external:
        return
    prefix_to_projection = {s.prefix: view.get_projection_for_spec(s) for s in spec.traverse(root=True, order='breadth') if not s.external}
    src_stat = os.lstat(src)
    if stat.S_ISLNK(src_stat.st_mode):
        spack.relocate.relocate_links(links=[dst], prefix_to_prefix=prefix_to_projection)
    elif spack.relocate.is_binary(dst):
        spack.relocate.relocate_text_bin(binaries=[dst], prefixes=prefix_to_projection)
    else:
        prefix_to_projection[spack.store.STORE.layout.root] = view._root
        prefix_to_projection['#!/bin/bash {0}/bin/sbang'.format(spack.paths.spack_root)] = sbang.sbang_shebang_line()
        spack.relocate.relocate_text(files=[dst], prefixes=prefix_to_projection)
    try:
        os.chown(dst, src_stat.st_uid, src_stat.st_gid)
    except OSError:
        tty.debug("Can't change the permissions for %s" % dst)

def view_func_parser(parsed_name):
    if False:
        for i in range(10):
            print('nop')
    if parsed_name in ('hardlink', 'hard'):
        return view_hardlink
    elif parsed_name in ('copy', 'relocate'):
        return view_copy
    elif parsed_name in ('add', 'symlink', 'soft'):
        return view_symlink
    else:
        raise ValueError("invalid link type for view: '%s'" % parsed_name)

def inverse_view_func_parser(view_type):
    if False:
        return 10
    if view_type is view_hardlink:
        link_name = 'hardlink'
    elif view_type is view_copy:
        link_name = 'copy'
    else:
        link_name = 'symlink'
    return link_name

class FilesystemView:
    """
    Governs a filesystem view that is located at certain root-directory.

    Packages are linked from their install directories into a common file
    hierachy.

    In distributed filesystems, loading each installed package seperately
    can lead to slow-downs due to too many directories being traversed.
    This can be circumvented by loading all needed modules into a common
    directory structure.
    """

    def __init__(self, root, layout, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        '\n        Initialize a filesystem view under the given `root` directory with\n        corresponding directory `layout`.\n\n        Files are linked by method `link` (llnl.util.symlink by default).\n        '
        self._root = root
        self.layout = layout
        self.projections = kwargs.get('projections', {})
        self.ignore_conflicts = kwargs.get('ignore_conflicts', False)
        self.verbose = kwargs.get('verbose', False)
        link_func = kwargs.get('link', view_symlink)
        self.link = ft.partial(link_func, view=self)

    def add_specs(self, *specs, **kwargs):
        if False:
            i = 10
            return i + 15
        '\n        Add given specs to view.\n\n        Should accept `with_dependencies` as keyword argument (default\n        True) to indicate wether or not dependencies should be activated as\n        well.\n\n        Should except an `exclude` keyword argument containing a list of\n        regexps that filter out matching spec names.\n\n        This method should make use of `activate_standalone`.\n        '
        raise NotImplementedError

    def add_standalone(self, spec):
        if False:
            for i in range(10):
                print('nop')
        '\n        Add (link) a standalone package into this view.\n        '
        raise NotImplementedError

    def check_added(self, spec):
        if False:
            i = 10
            return i + 15
        '\n        Check if the given concrete spec is active in this view.\n        '
        raise NotImplementedError

    def remove_specs(self, *specs, **kwargs):
        if False:
            while True:
                i = 10
        '\n        Removes given specs from view.\n\n        Should accept `with_dependencies` as keyword argument (default\n        True) to indicate wether or not dependencies should be deactivated\n        as well.\n\n        Should accept `with_dependents` as keyword argument (default True)\n        to indicate wether or not dependents on the deactivated specs\n        should be removed as well.\n\n        Should except an `exclude` keyword argument containing a list of\n        regexps that filter out matching spec names.\n\n        This method should make use of `deactivate_standalone`.\n        '
        raise NotImplementedError

    def remove_standalone(self, spec):
        if False:
            print('Hello World!')
        '\n        Remove (unlink) a standalone package from this view.\n        '
        raise NotImplementedError

    def get_projection_for_spec(self, spec):
        if False:
            i = 10
            return i + 15
        '\n        Get the projection in this view for a spec.\n        '
        raise NotImplementedError

    def get_all_specs(self):
        if False:
            i = 10
            return i + 15
        '\n        Get all specs currently active in this view.\n        '
        raise NotImplementedError

    def get_spec(self, spec):
        if False:
            while True:
                i = 10
        '\n        Return the actual spec linked in this view (i.e. do not look it up\n        in the database by name).\n\n        `spec` can be a name or a spec from which the name is extracted.\n\n        As there can only be a single version active for any spec the name\n        is enough to identify the spec in the view.\n\n        If no spec is present, returns None.\n        '
        raise NotImplementedError

    def print_status(self, *specs, **kwargs):
        if False:
            return 10
        '\n        Print a short summary about the given specs, detailing whether..\n            * ..they are active in the view.\n            * ..they are active but the activated version differs.\n            * ..they are not activte in the view.\n\n        Takes `with_dependencies` keyword argument so that the status of\n        dependencies is printed as well.\n        '
        raise NotImplementedError

class YamlFilesystemView(FilesystemView):
    """
    Filesystem view to work with a yaml based directory layout.
    """

    def __init__(self, root, layout, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        super().__init__(root, layout, **kwargs)
        self.projections_path = os.path.join(self._root, _projections_path)
        if not self.projections:
            self.projections = self.read_projections()
        elif not os.path.exists(self.projections_path):
            self.write_projections()
        elif self.projections != self.read_projections():
            msg = 'View at %s has projections file' % self._root
            msg += ' which does not match projections passed manually.'
            raise ConflictingProjectionsError(msg)
        self._croot = colorize_root(self._root) + ' '

    def write_projections(self):
        if False:
            print('Hello World!')
        if self.projections:
            mkdirp(os.path.dirname(self.projections_path))
            with open(self.projections_path, 'w') as f:
                f.write(s_yaml.dump_config({'projections': self.projections}))

    def read_projections(self):
        if False:
            i = 10
            return i + 15
        if os.path.exists(self.projections_path):
            with open(self.projections_path, 'r') as f:
                projections_data = s_yaml.load(f)
                spack.config.validate(projections_data, spack.schema.projections.schema)
                return projections_data['projections']
        else:
            return {}

    def add_specs(self, *specs, **kwargs):
        if False:
            print('Hello World!')
        assert all((s.concrete for s in specs))
        specs = set(specs)
        if kwargs.get('with_dependencies', True):
            specs.update(get_dependencies(specs))
        if kwargs.get('exclude', None):
            specs = set(filter_exclude(specs, kwargs['exclude']))
        conflicts = self.get_conflicts(*specs)
        if conflicts:
            for (s, v) in conflicts:
                self.print_conflict(v, s)
            return
        for s in specs:
            self.add_standalone(s)

    def add_standalone(self, spec):
        if False:
            for i in range(10):
                print('nop')
        if spec.external:
            tty.warn(self._croot + 'Skipping external package: %s' % colorize_spec(spec))
            return True
        if self.check_added(spec):
            tty.warn(self._croot + 'Skipping already linked package: %s' % colorize_spec(spec))
            return True
        self.merge(spec)
        self.link_meta_folder(spec)
        if self.verbose:
            tty.info(self._croot + 'Linked package: %s' % colorize_spec(spec))
        return True

    def merge(self, spec, ignore=None):
        if False:
            for i in range(10):
                print('nop')
        pkg = spec.package
        view_source = pkg.view_source()
        view_dst = pkg.view_destination(self)
        tree = LinkTree(view_source)
        ignore = ignore or (lambda f: False)
        ignore_file = match_predicate(self.layout.hidden_file_regexes, ignore)
        conflicts = tree.find_dir_conflicts(view_dst, ignore_file)
        merge_map = tree.get_file_map(view_dst, ignore_file)
        if not self.ignore_conflicts:
            conflicts.extend(pkg.view_file_conflicts(self, merge_map))
        if conflicts:
            raise SingleMergeConflictError(conflicts[0])
        tree.merge_directories(view_dst, ignore_file)
        pkg.add_files_to_view(self, merge_map)

    def unmerge(self, spec, ignore=None):
        if False:
            return 10
        pkg = spec.package
        view_source = pkg.view_source()
        view_dst = pkg.view_destination(self)
        tree = LinkTree(view_source)
        ignore = ignore or (lambda f: False)
        ignore_file = match_predicate(self.layout.hidden_file_regexes, ignore)
        merge_map = tree.get_file_map(view_dst, ignore_file)
        pkg.remove_files_from_view(self, merge_map)
        tree.unmerge_directories(view_dst, ignore_file)

    def remove_files(self, files):
        if False:
            return 10

        def needs_file(spec, file):
            if False:
                return 10
            projection = self.get_projection_for_spec(spec)
            relative_path = os.path.relpath(file, projection)
            test_path = os.path.join(spec.prefix, relative_path)
            manifest_file = os.path.join(self.get_path_meta_folder(spec), spack.store.STORE.layout.manifest_file_name)
            try:
                with open(manifest_file, 'r') as f:
                    manifest = s_json.load(f)
            except (OSError, IOError):
                manifest = {}
            return test_path in manifest
        specs = self.get_all_specs()
        for file in files:
            if not os.path.lexists(file):
                tty.warn('Tried to remove %s which does not exist' % file)
                continue
            if len([s for s in specs if needs_file(s, file)]) <= 1:
                tty.debug('Removing file ' + file)
                os.remove(file)

    def check_added(self, spec):
        if False:
            i = 10
            return i + 15
        assert spec.concrete
        return spec == self.get_spec(spec)

    def remove_specs(self, *specs, **kwargs):
        if False:
            return 10
        assert all((s.concrete for s in specs))
        with_dependents = kwargs.get('with_dependents', True)
        with_dependencies = kwargs.get('with_dependencies', False)
        all_specs = kwargs.get('all_specs', None) or set(self.get_all_specs())
        specs = set(specs)
        if with_dependencies:
            specs = get_dependencies(specs)
        if kwargs.get('exclude', None):
            specs = set(filter_exclude(specs, kwargs['exclude']))
        to_deactivate = specs
        to_keep = all_specs - to_deactivate
        dependents = find_dependents(to_keep, to_deactivate)
        if with_dependents:
            if len(dependents) > 0:
                tty.warn(self._croot + 'The following dependents will be removed: %s' % ', '.join((s.name for s in dependents)))
                to_deactivate.update(dependents)
        elif len(dependents) > 0:
            tty.warn(self._croot + 'The following packages will be unusable: %s' % ', '.join((s.name for s in dependents)))
        to_deactivate_sorted = list()
        depmap = dict()
        for spec in to_deactivate:
            depmap[spec] = set((d for d in spec.traverse(root=False) if d in to_deactivate))
        while depmap:
            for spec in [s for (s, d) in depmap.items() if not d]:
                to_deactivate_sorted.append(spec)
                for s in depmap.keys():
                    depmap[s].discard(spec)
                depmap.pop(spec)
        to_deactivate_sorted.reverse()
        assert set(to_deactivate_sorted) == to_deactivate
        for spec in to_deactivate_sorted:
            self.remove_standalone(spec)
        self._purge_empty_directories()

    def remove_standalone(self, spec):
        if False:
            i = 10
            return i + 15
        '\n        Remove (unlink) a standalone package from this view.\n        '
        if not self.check_added(spec):
            tty.warn(self._croot + 'Skipping package not linked in view: %s' % spec.name)
            return
        self.unmerge(spec)
        self.unlink_meta_folder(spec)
        if self.verbose:
            tty.info(self._croot + 'Removed package: %s' % colorize_spec(spec))

    def get_projection_for_spec(self, spec):
        if False:
            print('Hello World!')
        '\n        Return the projection for a spec in this view.\n\n        Relies on the ordering of projections to avoid ambiguity.\n        '
        spec = spack.spec.Spec(spec)
        locator_spec = spec
        if spec.package.extendee_spec:
            locator_spec = spec.package.extendee_spec
        proj = spack.projections.get_projection(self.projections, locator_spec)
        if proj:
            return os.path.join(self._root, locator_spec.format_path(proj))
        return self._root

    def get_all_specs(self):
        if False:
            for i in range(10):
                print('nop')
        md_dirs = []
        for (root, dirs, files) in os.walk(self._root):
            if spack.store.STORE.layout.metadata_dir in dirs:
                md_dirs.append(os.path.join(root, spack.store.STORE.layout.metadata_dir))
        specs = []
        for md_dir in md_dirs:
            if os.path.exists(md_dir):
                for name_dir in os.listdir(md_dir):
                    filename = os.path.join(md_dir, name_dir, spack.store.STORE.layout.spec_file_name)
                    spec = get_spec_from_file(filename)
                    if spec:
                        specs.append(spec)
        return specs

    def get_conflicts(self, *specs):
        if False:
            return 10
        '\n        Return list of tuples (<spec>, <spec in view>) where the spec\n        active in the view differs from the one to be activated.\n        '
        in_view = map(self.get_spec, specs)
        return [(s, v) for (s, v) in zip(specs, in_view) if v is not None and s != v]

    def get_path_meta_folder(self, spec):
        if False:
            return 10
        'Get path to meta folder for either spec or spec name.'
        return os.path.join(self.get_projection_for_spec(spec), spack.store.STORE.layout.metadata_dir, getattr(spec, 'name', spec))

    def get_spec(self, spec):
        if False:
            while True:
                i = 10
        dotspack = self.get_path_meta_folder(spec)
        filename = os.path.join(dotspack, spack.store.STORE.layout.spec_file_name)
        return get_spec_from_file(filename)

    def link_meta_folder(self, spec):
        if False:
            print('Hello World!')
        src = spack.store.STORE.layout.metadata_path(spec)
        tgt = self.get_path_meta_folder(spec)
        tree = LinkTree(src)
        tree.merge(tgt, link=self.link)

    def print_conflict(self, spec_active, spec_specified, level='error'):
        if False:
            return 10
        'Singular print function for spec conflicts.'
        cprint = getattr(tty, level)
        color = sys.stdout.isatty()
        linked = tty.color.colorize('   (@gLinked@.)', color=color)
        specified = tty.color.colorize('(@rSpecified@.)', color=color)
        cprint(self._croot + 'Package conflict detected:\n%s %s\n' % (linked, colorize_spec(spec_active)) + '%s %s' % (specified, colorize_spec(spec_specified)))

    def print_status(self, *specs, **kwargs):
        if False:
            print('Hello World!')
        if kwargs.get('with_dependencies', False):
            specs = set(get_dependencies(specs))
        specs = sorted(specs, key=lambda s: s.name)
        in_view = list(map(self.get_spec, specs))
        for (s, v) in zip(specs, in_view):
            if not v:
                tty.error(self._croot + 'Package not linked: %s' % s.name)
            elif s != v:
                self.print_conflict(v, s, level='warn')
        in_view = list(filter(None, in_view))
        if len(specs) > 0:
            tty.msg('Packages linked in %s:' % self._croot[:-1])
            index = index_by(specs, ('architecture', 'compiler'))
            for (i, (architecture, compiler)) in enumerate(sorted(index)):
                if i > 0:
                    print()
                header = '%s{%s} / %s{%s}' % (spack.spec.ARCHITECTURE_COLOR, architecture, spack.spec.COMPILER_COLOR, compiler)
                tty.hline(colorize(header), char='-')
                specs = index[architecture, compiler]
                specs.sort()
                format_string = '{name}{@version}'
                format_string += '{%compiler}{compiler_flags}{variants}'
                abbreviated = [s.cformat(format_string) for s in specs]
                width = max((len(s) for s in abbreviated))
                width += 2
                format = '    %%-%ds%%s' % width
                for (abbrv, s) in zip(abbreviated, specs):
                    prefix = ''
                    if self.verbose:
                        prefix = colorize('@K{%s}' % s.dag_hash(7))
                    print(prefix + format % (abbrv, self.get_projection_for_spec(s)))
        else:
            tty.warn(self._croot + 'No packages found.')

    def _purge_empty_directories(self):
        if False:
            i = 10
            return i + 15
        remove_empty_directories(self._root)

    def _purge_broken_links(self):
        if False:
            for i in range(10):
                print('nop')
        remove_dead_links(self._root)

    def clean(self):
        if False:
            for i in range(10):
                print('nop')
        self._purge_broken_links()
        self._purge_empty_directories()

    def unlink_meta_folder(self, spec):
        if False:
            return 10
        path = self.get_path_meta_folder(spec)
        assert os.path.exists(path)
        shutil.rmtree(path)

class SimpleFilesystemView(FilesystemView):
    """A simple and partial implementation of FilesystemView focused on
    performance and immutable views, where specs cannot be removed after they
    were added."""

    def __init__(self, root, layout, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        super().__init__(root, layout, **kwargs)

    def _sanity_check_view_projection(self, specs):
        if False:
            print('Hello World!')
        "A very common issue is that we end up with two specs of the same\n        package, that project to the same prefix. We want to catch that as\n        early as possible and give a sensible error to the user. Here we use\n        the metadata dir (.spack) projection as a quick test to see whether\n        two specs in the view are going to clash. The metadata dir is used\n        because it's always added by Spack with identical files, so a\n        guaranteed clash that's easily verified."
        seen = dict()
        for current_spec in specs:
            metadata_dir = self.relative_metadata_dir_for_spec(current_spec)
            conflicting_spec = seen.get(metadata_dir)
            if conflicting_spec:
                raise ConflictingSpecsError(current_spec, conflicting_spec)
            seen[metadata_dir] = current_spec

    def add_specs(self, *specs, **kwargs):
        if False:
            while True:
                i = 10
        assert all((s.concrete for s in specs))
        if len(specs) == 0:
            return
        for s in specs:
            if s.external:
                tty.warn('Skipping external package: ' + s.short_spec)
        specs = [s for s in specs if not s.external]
        if kwargs.get('exclude', None):
            specs = set(filter_exclude(specs, kwargs['exclude']))
        self._sanity_check_view_projection(specs)

        def skip_list(file):
            if False:
                return 10
            return os.path.basename(file) == spack.store.STORE.layout.metadata_dir
        visitor = SourceMergeVisitor(ignore=skip_list)
        for spec in specs:
            src_prefix = spec.package.view_source()
            visitor.set_projection(self.get_relative_projection_for_spec(spec))
            visit_directory_tree(src_prefix, visitor)
        visit_directory_tree(self._root, DestinationMergeVisitor(visitor))
        if visitor.fatal_conflicts:
            raise MergeConflictSummary(visitor.fatal_conflicts)
        if visitor.file_conflicts:
            if self.ignore_conflicts:
                tty.debug('{0} file conflicts'.format(len(visitor.file_conflicts)))
            else:
                raise MergeConflictSummary(visitor.file_conflicts)
        tty.debug('Creating {0} dirs and {1} links'.format(len(visitor.directories), len(visitor.files)))
        for dst in visitor.directories:
            os.mkdir(os.path.join(self._root, dst))
        merge_map_per_prefix = self._source_merge_visitor_to_merge_map(visitor)
        for spec in specs:
            merge_map = merge_map_per_prefix.get(spec.package.view_source(), None)
            if not merge_map:
                continue
            spec.package.add_files_to_view(self, merge_map, skip_if_exists=False)
        self.link_metadata(specs)

    def _source_merge_visitor_to_merge_map(self, visitor: SourceMergeVisitor):
        if False:
            while True:
                i = 10
        source_root = lambda item: item[1][0]
        per_source = itertools.groupby(visitor.files.items(), key=source_root)
        return {src_root: {os.path.join(src_root, src_rel): os.path.join(self._root, dst_rel) for (dst_rel, (_, src_rel)) in group} for (src_root, group) in per_source}

    def relative_metadata_dir_for_spec(self, spec):
        if False:
            return 10
        return os.path.join(self.get_relative_projection_for_spec(spec), spack.store.STORE.layout.metadata_dir, spec.name)

    def link_metadata(self, specs):
        if False:
            return 10
        metadata_visitor = SourceMergeVisitor()
        for spec in specs:
            src_prefix = os.path.join(spec.package.view_source(), spack.store.STORE.layout.metadata_dir)
            proj = self.relative_metadata_dir_for_spec(spec)
            metadata_visitor.set_projection(proj)
            visit_directory_tree(src_prefix, metadata_visitor)
        visit_directory_tree(self._root, DestinationMergeVisitor(metadata_visitor))
        if metadata_visitor.fatal_conflicts:
            raise MergeConflictSummary(metadata_visitor.fatal_conflicts)
        if metadata_visitor.file_conflicts:
            raise MergeConflictSummary(metadata_visitor.file_conflicts)
        for dst in metadata_visitor.directories:
            os.mkdir(os.path.join(self._root, dst))
        for (dst_relpath, (src_root, src_relpath)) in metadata_visitor.files.items():
            self.link(os.path.join(src_root, src_relpath), os.path.join(self._root, dst_relpath))

    def get_relative_projection_for_spec(self, spec):
        if False:
            i = 10
            return i + 15
        if spec.package.extendee_spec:
            spec = spec.package.extendee_spec
        p = spack.projections.get_projection(self.projections, spec)
        return spec.format_path(p) if p else ''

    def get_projection_for_spec(self, spec):
        if False:
            print('Hello World!')
        '\n        Return the projection for a spec in this view.\n\n        Relies on the ordering of projections to avoid ambiguity.\n        '
        spec = spack.spec.Spec(spec)
        if spec.package.extendee_spec:
            spec = spec.package.extendee_spec
        proj = spack.projections.get_projection(self.projections, spec)
        if proj:
            return os.path.join(self._root, spec.format_path(proj))
        return self._root

def get_spec_from_file(filename):
    if False:
        while True:
            i = 10
    try:
        with open(filename, 'r') as f:
            return spack.spec.Spec.from_yaml(f)
    except IOError:
        return None

def colorize_root(root):
    if False:
        print('Hello World!')
    colorize = ft.partial(tty.color.colorize, color=sys.stdout.isatty())
    (pre, post) = map(colorize, '@M[@. @M]@.'.split())
    return ''.join([pre, root, post])

def colorize_spec(spec):
    if False:
        i = 10
        return i + 15
    'Colorize spec output if in TTY.'
    if sys.stdout.isatty():
        return spec.cshort_spec
    else:
        return spec.short_spec

def find_dependents(all_specs, providers, deptype='run'):
    if False:
        return 10
    '\n    Return a set containing all those specs from all_specs that depend on\n    providers at the given dependency type.\n    '
    dependents = set()
    for s in all_specs:
        for dep in s.traverse(deptype=deptype):
            if dep in providers:
                dependents.add(s)
    return dependents

def filter_exclude(specs, exclude):
    if False:
        for i in range(10):
            print('nop')
    'Filter specs given sequence of exclude regex'
    to_exclude = [re.compile(e) for e in exclude]

    def keep(spec):
        if False:
            print('Hello World!')
        for e in to_exclude:
            if e.match(spec.name):
                return False
        return True
    return filter(keep, specs)

def get_dependencies(specs):
    if False:
        return 10
    'Get set of dependencies (includes specs)'
    retval = set()
    set(map(retval.update, (set(s.traverse()) for s in specs)))
    return retval

class ConflictingProjectionsError(SpackError):
    """Raised when a view has a projections file and is given one manually."""