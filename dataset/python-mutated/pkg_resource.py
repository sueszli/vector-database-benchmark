"""
Resources needed by pkg providers
"""
import copy
import fnmatch
import logging
import os
import pprint
import salt.utils.data
import salt.utils.versions
import salt.utils.yaml
from salt.exceptions import SaltInvocationError
log = logging.getLogger(__name__)
__SUFFIX_NOT_NEEDED = ('x86_64', 'noarch')

def _repack_pkgs(pkgs, normalize=True):
    if False:
        print('Hello World!')
    '\n    Repack packages specified using "pkgs" argument to pkg states into a single\n    dictionary\n    '
    if normalize and 'pkg.normalize_name' in __salt__:
        _normalize_name = __salt__['pkg.normalize_name']
    else:
        _normalize_name = lambda pkgname: pkgname
    repacked_pkgs = {_normalize_name(str(x)): str(y) if y is not None else y for (x, y) in salt.utils.data.repack_dictlist(pkgs).items()}
    if len(pkgs) != len(repacked_pkgs):
        raise SaltInvocationError('You are passing a list of packages that contains duplicated packages names: {}. This cannot be processed. In case you are targeting different versions of the same package, please target them individually'.format(pkgs))
    return repacked_pkgs

def pack_sources(sources, normalize=True):
    if False:
        print('Hello World!')
    '\n    Accepts list of dicts (or a string representing a list of dicts) and packs\n    the key/value pairs into a single dict.\n\n    ``\'[{"foo": "salt://foo.rpm"}, {"bar": "salt://bar.rpm"}]\'`` would become\n    ``{"foo": "salt://foo.rpm", "bar": "salt://bar.rpm"}``\n\n    normalize : True\n        Normalize the package name by removing the architecture, if the\n        architecture of the package is different from the architecture of the\n        operating system. The ability to disable this behavior is useful for\n        poorly-created packages which include the architecture as an actual\n        part of the name, such as kernel modules which match a specific kernel\n        version.\n\n        .. versionadded:: 2015.8.0\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt \'*\' pkg_resource.pack_sources \'[{"foo": "salt://foo.rpm"}, {"bar": "salt://bar.rpm"}]\'\n    '
    if normalize and 'pkg.normalize_name' in __salt__:
        _normalize_name = __salt__['pkg.normalize_name']
    else:
        _normalize_name = lambda pkgname: pkgname
    if isinstance(sources, str):
        try:
            sources = salt.utils.yaml.safe_load(sources)
        except salt.utils.yaml.parser.ParserError as err:
            log.error(err)
            return {}
    ret = {}
    for source in sources:
        if not isinstance(source, dict) or len(source) != 1:
            log.error('Invalid input: %s', pprint.pformat(sources))
            log.error('Input must be a list of 1-element dicts')
            return {}
        else:
            key = next(iter(source))
            ret[_normalize_name(key)] = source[key]
    return ret

def parse_targets(name=None, pkgs=None, sources=None, saltenv='base', normalize=True, **kwargs):
    if False:
        while True:
            i = 10
    "\n    Parses the input to pkg.install and returns back the package(s) to be\n    installed. Returns a list of packages, as well as a string noting whether\n    the packages are to come from a repository or a binary package.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' pkg_resource.parse_targets\n    "
    if '__env__' in kwargs:
        kwargs.pop('__env__')
    if __grains__['os'] == 'MacOS' and sources:
        log.warning('Parameter "sources" ignored on MacOS hosts.')
    version = kwargs.get('version')
    if pkgs and sources:
        log.error('Only one of "pkgs" and "sources" can be used.')
        return (None, None)
    elif 'advisory_ids' in kwargs:
        if pkgs:
            log.error('Cannot use "advisory_ids" and "pkgs" at the same time')
            return (None, None)
        elif kwargs['advisory_ids']:
            return (kwargs['advisory_ids'], 'advisory')
        else:
            return ([name], 'advisory')
    elif pkgs:
        if version is not None:
            log.warning("'version' argument will be ignored for multiple package targets")
        pkgs = _repack_pkgs(pkgs, normalize=normalize)
        if not pkgs:
            return (None, None)
        else:
            return (pkgs, 'repository')
    elif sources and __grains__['os'] != 'MacOS':
        if version is not None:
            log.warning("'version' argument will be ignored for multiple package targets")
        sources = pack_sources(sources, normalize=normalize)
        if not sources:
            return (None, None)
        srcinfo = []
        for (pkg_name, pkg_src) in sources.items():
            if __salt__['config.valid_fileproto'](pkg_src):
                srcinfo.append(__salt__['cp.cache_file'](pkg_src, saltenv))
            else:
                if not os.path.isabs(pkg_src):
                    raise SaltInvocationError('Path {} for package {} is either not absolute or an invalid protocol'.format(pkg_src, pkg_name))
                srcinfo.append(pkg_src)
        return (srcinfo, 'file')
    elif name:
        if normalize:
            _normalize_name = __salt__.get('pkg.normalize_name', lambda pkgname: pkgname)
            packed = {_normalize_name(x): version for x in name.split(',')}
        else:
            packed = {x: version for x in name.split(',')}
        return (packed, 'repository')
    else:
        log.error('No package sources provided')
        return (None, None)

def version(*names, **kwargs):
    if False:
        while True:
            i = 10
    "\n    Common interface for obtaining the version of installed packages.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' pkg_resource.version vim\n        salt '*' pkg_resource.version foo bar baz\n        salt '*' pkg_resource.version 'python*'\n    "
    ret = {}
    versions_as_list = salt.utils.data.is_true(kwargs.pop('versions_as_list', False))
    pkg_glob = False
    if names:
        pkgs = __salt__['pkg.list_pkgs'](versions_as_list=True, **kwargs)
        for name in names:
            if '*' in name:
                pkg_glob = True
                for match in fnmatch.filter(pkgs, name):
                    ret[match] = pkgs.get(match, [])
            else:
                ret[name] = pkgs.get(name, [])
    if not versions_as_list:
        __salt__['pkg_resource.stringify'](ret)
    if len(ret) == 1 and (not pkg_glob):
        try:
            return next(iter(ret.values()))
        except StopIteration:
            return ''
    return ret

def add_pkg(pkgs, name, pkgver):
    if False:
        while True:
            i = 10
    "\n    Add a package to a dict of installed packages.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' pkg_resource.add_pkg '{}' bind 9\n    "
    try:
        pkgs.setdefault(name, []).append(pkgver)
    except AttributeError as exc:
        log.exception(exc)

def sort_pkglist(pkgs):
    if False:
        i = 10
        return i + 15
    '\n    Accepts a dict obtained from pkg.list_pkgs() and sorts in place the list of\n    versions for any packages that have multiple versions installed, so that\n    two package lists can be compared to one another.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt \'*\' pkg_resource.sort_pkglist \'["3.45", "2.13"]\'\n    '
    try:
        for key in pkgs:
            pkgs[key] = sorted(set(pkgs[key]))
    except AttributeError as exc:
        log.exception(exc)

def stringify(pkgs):
    if False:
        i = 10
        return i + 15
    "\n    Takes a dict of package name/version information and joins each list of\n    installed versions into a string.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' pkg_resource.stringify 'vim: 7.127'\n    "
    try:
        for key in pkgs:
            pkgs[key] = ','.join(pkgs[key])
    except AttributeError as exc:
        log.exception(exc)

def version_clean(verstr):
    if False:
        i = 10
        return i + 15
    "\n    Clean the version string removing extra data.\n    This function will simply try to call ``pkg.version_clean``.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' pkg_resource.version_clean <version_string>\n    "
    if verstr and 'pkg.version_clean' in __salt__:
        return __salt__['pkg.version_clean'](verstr)
    return verstr

def version_compare(ver1, oper, ver2, ignore_epoch=False):
    if False:
        while True:
            i = 10
    "\n    .. versionadded:: 3001\n\n    Perform a version comparison, using (where available) platform-specific\n    version comparison tools to make the comparison.\n\n    ver1\n        The first version to be compared\n\n    oper\n        One of `==`, `!=`, `>=`, `<=`, `>`, `<`\n\n    ver2\n        The second version to be compared\n\n    .. note::\n        To avoid shell interpretation, each of the above values should be\n        quoted when this function is used on the CLI.\n\n    ignore_epoch : False\n        If ``True``, both package versions will have their epoch prefix\n        stripped before comparison.\n\n    This function is useful in Jinja templates, to perform specific actions\n    when a package's version meets certain criteria. For example:\n\n    .. code-block:: jinja\n\n        {%- set postfix_version = salt.pkg.version('postfix') %}\n        {%- if postfix_version and salt.pkg_resource.version_compare(postfix_version, '>=', '3.3', ignore_epoch=True) %}\n          {#- do stuff #}\n        {%- endif %}\n\n    CLI Examples:\n\n    .. code-block:: bash\n\n        salt myminion pkg_resource.version_compare '3.5' '<=' '2.4'\n        salt myminion pkg_resource.version_compare '3.5' '<=' '2.4' ignore_epoch=True\n    "
    return salt.utils.versions.compare(ver1, oper, ver2, ignore_epoch=ignore_epoch, cmp_func=__salt__.get('version_cmp'))

def check_extra_requirements(pkgname, pkgver):
    if False:
        return 10
    "\n    Check if the installed package already has the given requirements.\n    This function will return the result of ``pkg.check_extra_requirements`` if\n    this function exists for the minion, otherwise it will return True.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' pkg_resource.check_extra_requirements <pkgname> <extra_requirements>\n    "
    if pkgver and 'pkg.check_extra_requirements' in __salt__:
        return __salt__['pkg.check_extra_requirements'](pkgname, pkgver)
    return True

def format_pkg_list(packages, versions_as_list, attr):
    if False:
        i = 10
        return i + 15
    '\n    Formats packages according to parameters for list_pkgs.\n    '
    ret = copy.deepcopy(packages)
    if attr:
        ret_attr = {}
        requested_attr = {'epoch', 'version', 'release', 'arch', 'install_date', 'install_date_time_t'}
        if attr != 'all':
            requested_attr &= set(attr + ['version'] + ['arch'])
        for name in ret:
            if 'pkg.parse_arch' in __salt__:
                _parse_arch = __salt__['pkg.parse_arch'](name)
            else:
                _parse_arch = {'name': name, 'arch': None}
            _name = _parse_arch['name']
            _arch = _parse_arch['arch']
            versions = []
            pkgname = None
            for all_attr in ret[name]:
                filtered_attr = {}
                for key in requested_attr:
                    if key in all_attr:
                        filtered_attr[key] = all_attr[key]
                versions.append(filtered_attr)
                if _name and filtered_attr.get('arch', None) == _arch:
                    pkgname = _name
            ret_attr.setdefault(pkgname or name, []).extend(versions)
        return ret_attr
    for name in ret:
        ret[name] = [format_version(d['epoch'], d['version'], d['release']) for d in ret[name]]
    if not versions_as_list:
        stringify(ret)
    return ret

def format_version(epoch, version, release):
    if False:
        while True:
            i = 10
    '\n    Formats a version string for list_pkgs.\n    '
    full_version = '{}:{}'.format(epoch, version) if epoch else version
    if release:
        full_version += '-{}'.format(release)
    return full_version