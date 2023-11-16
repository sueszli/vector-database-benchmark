import sys
import llnl.util.tty as tty
from llnl.util.tty.colify import colify
import spack.cmd
import spack.cmd.common.arguments as arguments
import spack.environment as ev
import spack.repo
import spack.store
description = 'show packages that depend on another'
section = 'basic'
level = 'long'

def setup_parser(subparser):
    if False:
        print('Hello World!')
    subparser.add_argument('-i', '--installed', action='store_true', default=False, help='list installed dependents of an installed spec instead of possible dependents of a package')
    subparser.add_argument('-t', '--transitive', action='store_true', default=False, help='show all transitive dependents')
    arguments.add_common_arguments(subparser, ['spec'])

def inverted_dependencies():
    if False:
        i = 10
        return i + 15
    'Iterate through all packages and return a dictionary mapping package\n    names to possible dependencies.\n\n    Virtual packages are included as sources, so that you can query\n    dependents of, e.g., `mpi`, but virtuals are not included as\n    actual dependents.\n    '
    dag = {}
    for pkg_cls in spack.repo.PATH.all_package_classes():
        dag.setdefault(pkg_cls.name, set())
        for dep in pkg_cls.dependencies:
            deps = [dep]
            if spack.repo.PATH.is_virtual(dep):
                deps += [s.name for s in spack.repo.PATH.providers_for(dep)]
            for d in deps:
                dag.setdefault(d, set()).add(pkg_cls.name)
    return dag

def get_dependents(pkg_name, ideps, transitive=False, dependents=None):
    if False:
        return 10
    'Get all dependents for a package.\n\n    Args:\n        pkg_name (str): name of the package whose dependents should be returned\n        ideps (dict): dictionary of dependents, from inverted_dependencies()\n        transitive (bool or None): return transitive dependents when True\n    '
    if dependents is None:
        dependents = set()
    if pkg_name in dependents:
        return set()
    dependents.add(pkg_name)
    direct = ideps[pkg_name]
    if transitive:
        for dep_name in direct:
            get_dependents(dep_name, ideps, transitive, dependents)
    dependents.update(direct)
    return dependents

def dependents(parser, args):
    if False:
        i = 10
        return i + 15
    specs = spack.cmd.parse_specs(args.spec)
    if len(specs) != 1:
        tty.die('spack dependents takes only one spec.')
    if args.installed:
        env = ev.active_environment()
        spec = spack.cmd.disambiguate_spec(specs[0], env)
        format_string = '{name}{@version}{%compiler}{/hash:7}'
        if sys.stdout.isatty():
            tty.msg('Dependents of %s' % spec.cformat(format_string))
        deps = spack.store.STORE.db.installed_relatives(spec, 'parents', args.transitive)
        if deps:
            spack.cmd.display_specs(deps, long=True)
        else:
            print('No dependents')
    else:
        spec = specs[0]
        ideps = inverted_dependencies()
        dependents = get_dependents(spec.name, ideps, args.transitive)
        dependents.remove(spec.name)
        if dependents:
            colify(sorted(dependents))
        else:
            print('No dependents')