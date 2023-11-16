"""
Configure ``portage(5)``
"""
import logging
import os
import shutil
import salt.utils.compat
import salt.utils.data
import salt.utils.files
import salt.utils.path
import salt.utils.stringutils
try:
    import portage
    HAS_PORTAGE = True
except ImportError:
    HAS_PORTAGE = False
    import sys
    if os.path.isdir('/usr/lib/portage/pym'):
        try:
            sys.path.insert(0, '/usr/lib/portage/pym')
            import portage
            HAS_PORTAGE = True
        except ImportError:
            pass
BASE_PATH = '/etc/portage/package.{0}'
SUPPORTED_CONFS = ('accept_keywords', 'env', 'license', 'mask', 'properties', 'unmask', 'use')
log = logging.getLogger(__name__)

def __virtual__():
    if False:
        while True:
            i = 10
    '\n    Confirm this module is on a Gentoo based system.\n    '
    if HAS_PORTAGE and __grains__['os'] == 'Gentoo':
        return 'portage_config'
    return (False, 'portage_config execution module cannot be loaded: only available on Gentoo with portage installed.')

def _get_portage():
    if False:
        return 10
    "\n    portage module must be reloaded or it can't catch the changes\n    in portage.* which had been added after when the module was loaded\n    "
    return salt.utils.compat.reload(portage)

def _porttree():
    if False:
        i = 10
        return i + 15
    return portage.db[portage.root]['porttree']

def _get_config_file(conf, atom):
    if False:
        while True:
            i = 10
    '\n    Parse the given atom, allowing access to its parts\n    Success does not mean that the atom exists, just that it\n    is in the correct format.\n    Returns none if the atom is invalid.\n    '
    if '*' in atom:
        parts = portage.dep.Atom(atom, allow_wildcard=True)
        if not parts:
            return
        if parts.cp == '*/*':
            relative_path = parts.repo or 'gentoo'
        elif str(parts.cp).endswith('/*'):
            relative_path = str(parts.cp).split('/')[0] + '_'
        else:
            relative_path = os.path.join(*[x for x in os.path.split(parts.cp) if x != '*'])
    else:
        relative_path = _p_to_cp(atom)
        if not relative_path:
            return
    complete_file_path = BASE_PATH.format(conf) + '/' + relative_path
    return complete_file_path

def _p_to_cp(p):
    if False:
        i = 10
        return i + 15
    '\n    Convert a package name or a DEPEND atom to category/package format.\n    Raises an exception if program name is ambiguous.\n    '
    try:
        ret = portage.dep_getkey(p)
        if ret:
            return ret
    except portage.exception.InvalidAtom:
        pass
    try:
        ret = _porttree().dbapi.xmatch('bestmatch-visible', p)
        if ret:
            return portage.dep_getkey(ret)
    except portage.exception.InvalidAtom:
        pass
    try:
        ret = _porttree().dbapi.xmatch('match-all', p)
        if ret:
            return portage.cpv_getkey(ret[0])
    except portage.exception.InvalidAtom:
        pass
    return None

def _get_cpv(cp, installed=True):
    if False:
        print('Hello World!')
    '\n    add version to category/package\n    @cp - name of package in format category/name\n    @installed - boolean value, if False, function returns cpv\n    for latest available package\n    '
    if installed:
        return _get_portage().db[portage.root]['vartree'].dep_bestmatch(cp)
    else:
        return _porttree().dep_bestmatch(cp)

def enforce_nice_config():
    if False:
        for i in range(10):
            print('nop')
    "\n    Enforce a nice tree structure for /etc/portage/package.* configuration\n    files.\n\n    .. seealso::\n       :py:func:`salt.modules.ebuild.ex_mod_init`\n         for information on automatically running this when pkg is used.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' portage_config.enforce_nice_config\n    "
    _convert_all_package_confs_to_dir()
    _order_all_package_confs()

def _convert_all_package_confs_to_dir():
    if False:
        i = 10
        return i + 15
    '\n    Convert all /etc/portage/package.* configuration files to directories.\n    '
    for conf_file in SUPPORTED_CONFS:
        _package_conf_file_to_dir(conf_file)

def _order_all_package_confs():
    if False:
        for i in range(10):
            print('nop')
    '\n    Place all entries in /etc/portage/package.* config dirs in the correct\n    file.\n    '
    for conf_file in SUPPORTED_CONFS:
        _package_conf_ordering(conf_file)
    _unify_keywords()

def _unify_keywords():
    if False:
        return 10
    '\n    Merge /etc/portage/package.keywords and\n    /etc/portage/package.accept_keywords.\n    '
    old_path = BASE_PATH.format('keywords')
    if os.path.exists(old_path):
        if os.path.isdir(old_path):
            for triplet in salt.utils.path.os_walk(old_path):
                for file_name in triplet[2]:
                    file_path = '{}/{}'.format(triplet[0], file_name)
                    with salt.utils.files.fopen(file_path) as fh_:
                        for line in fh_:
                            line = salt.utils.stringutils.to_unicode(line).strip()
                            if line and (not line.startswith('#')):
                                append_to_package_conf('accept_keywords', string=line)
            shutil.rmtree(old_path)
        else:
            with salt.utils.files.fopen(old_path) as fh_:
                for line in fh_:
                    line = salt.utils.stringutils.to_unicode(line).strip()
                    if line and (not line.startswith('#')):
                        append_to_package_conf('accept_keywords', string=line)
            os.remove(old_path)

def _package_conf_file_to_dir(file_name):
    if False:
        for i in range(10):
            print('nop')
    '\n    Convert a config file to a config directory.\n    '
    if file_name in SUPPORTED_CONFS:
        path = BASE_PATH.format(file_name)
        if os.path.exists(path):
            if os.path.isdir(path):
                return False
            else:
                os.rename(path, path + '.tmpbak')
                os.mkdir(path, 493)
                os.rename(path + '.tmpbak', os.path.join(path, 'tmp'))
                return True
        else:
            os.mkdir(path, 493)
            return True

def _package_conf_ordering(conf, clean=True, keep_backup=False):
    if False:
        return 10
    '\n    Move entries in the correct file.\n    '
    if conf in SUPPORTED_CONFS:
        rearrange = []
        path = BASE_PATH.format(conf)
        backup_files = []
        for triplet in salt.utils.path.os_walk(path):
            for file_name in triplet[2]:
                file_path = '{}/{}'.format(triplet[0], file_name)
                cp = triplet[0][len(path) + 1:] + '/' + file_name
                shutil.copy(file_path, file_path + '.bak')
                backup_files.append(file_path + '.bak')
                if cp[0] == '/' or len(cp.split('/')) > 2:
                    with salt.utils.files.fopen(file_path) as fp_:
                        rearrange.extend(salt.utils.data.decode(fp_.readlines()))
                    os.remove(file_path)
                else:
                    new_contents = ''
                    with salt.utils.files.fopen(file_path, 'r+') as file_handler:
                        for line in file_handler:
                            line = salt.utils.stringutils.to_unicode(line)
                            try:
                                atom = line.strip().split()[0]
                            except IndexError:
                                new_contents += line
                            else:
                                if atom[0] == '#' or portage.dep_getkey(atom) == cp:
                                    new_contents += line
                                else:
                                    rearrange.append(line.strip())
                        if new_contents:
                            file_handler.seek(0)
                            file_handler.truncate(len(new_contents))
                            file_handler.write(new_contents)
                    if not new_contents:
                        os.remove(file_path)
        for line in rearrange:
            append_to_package_conf(conf, string=line)
        if not keep_backup:
            for bfile in backup_files:
                try:
                    os.remove(bfile)
                except OSError:
                    pass
        if clean:
            for triplet in salt.utils.path.os_walk(path):
                if not triplet[1] and (not triplet[2]) and (triplet[0] != path):
                    shutil.rmtree(triplet[0])

def _check_accept_keywords(approved, flag):
    if False:
        while True:
            i = 10
    'check compatibility of accept_keywords'
    if flag in approved:
        return False
    elif flag.startswith('~') and flag[1:] in approved or '~' + flag in approved:
        return False
    else:
        return True

def _merge_flags(new_flags, old_flags=None, conf='any'):
    if False:
        i = 10
        return i + 15
    '\n    Merges multiple lists of flags removing duplicates and resolving conflicts\n    giving priority to lasts lists.\n    '
    if not old_flags:
        old_flags = []
    args = [old_flags, new_flags]
    if conf == 'accept_keywords':
        tmp = new_flags + [i for i in old_flags if _check_accept_keywords(new_flags, i)]
    else:
        tmp = portage.flatten(args)
    flags = {}
    for flag in tmp:
        if flag[0] == '-':
            flags[flag[1:]] = False
        else:
            flags[flag] = True
    tmp = []
    for (key, val) in flags.items():
        if val:
            tmp.append(key)
        else:
            tmp.append('-' + key)
    tmp.sort(key=lambda x: x.lstrip('-'))
    return tmp

def append_to_package_conf(conf, atom='', flags=None, string='', overwrite=False):
    if False:
        print('Hello World!')
    '\n    Append a string or a list of flags for a given package or DEPEND atom to a\n    given configuration file.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt \'*\' portage_config.append_to_package_conf use string="app-admin/salt ldap -libvirt"\n        salt \'*\' portage_config.append_to_package_conf use atom="> = app-admin/salt-0.14.1" flags="[\'ldap\', \'-libvirt\']"\n    '
    if flags is None:
        flags = []
    if conf in SUPPORTED_CONFS:
        if not string:
            if '/' not in atom:
                atom = _p_to_cp(atom)
                if not atom:
                    return
            string = '{} {}'.format(atom, ' '.join(flags))
            new_flags = list(flags)
        else:
            atom = string.strip().split()[0]
            new_flags = [flag for flag in string.strip().split(' ') if flag][1:]
            if '/' not in atom:
                atom = _p_to_cp(atom)
                string = '{} {}'.format(atom, ' '.join(new_flags))
                if not atom:
                    return
        to_delete_if_empty = []
        if conf == 'accept_keywords':
            if '-~ARCH' in new_flags:
                new_flags.remove('-~ARCH')
                to_delete_if_empty.append(atom)
            if '~ARCH' in new_flags:
                new_flags.remove('~ARCH')
                append_to_package_conf(conf, string=atom, overwrite=overwrite)
                if not new_flags:
                    return
        new_flags.sort(key=lambda x: x.lstrip('-'))
        complete_file_path = _get_config_file(conf, atom)
        pdir = os.path.dirname(complete_file_path)
        if not os.path.exists(pdir):
            os.makedirs(pdir, 493)
        try:
            shutil.copy(complete_file_path, complete_file_path + '.bak')
        except OSError:
            pass
        try:
            file_handler = salt.utils.files.fopen(complete_file_path, 'r+')
        except OSError:
            file_handler = salt.utils.files.fopen(complete_file_path, 'w+')
        new_contents = ''
        added = False
        try:
            for l in file_handler:
                l_strip = l.strip()
                if l_strip == '':
                    new_contents += '\n'
                elif l_strip[0] == '#':
                    new_contents += l
                elif l_strip.split()[0] == atom:
                    if l_strip in to_delete_if_empty:
                        continue
                    if overwrite:
                        new_contents += string.strip() + '\n'
                        added = True
                    else:
                        old_flags = [flag for flag in l_strip.split(' ') if flag][1:]
                        if conf == 'accept_keywords':
                            if not old_flags:
                                new_contents += l
                                if not new_flags:
                                    added = True
                                continue
                            elif not new_flags:
                                continue
                        merged_flags = _merge_flags(new_flags, old_flags, conf)
                        if merged_flags:
                            new_contents += '{} {}\n'.format(atom, ' '.join(merged_flags))
                        else:
                            new_contents += '{}\n'.format(atom)
                        added = True
                else:
                    new_contents += l
            if not added:
                new_contents += string.strip() + '\n'
        except Exception as exc:
            log.error('Failed to write to %s: %s', complete_file_path, exc)
        else:
            file_handler.seek(0)
            file_handler.truncate(len(new_contents))
            file_handler.write(new_contents)
        finally:
            file_handler.close()
        try:
            os.remove(complete_file_path + '.bak')
        except OSError:
            pass

def append_use_flags(atom, uses=None, overwrite=False):
    if False:
        print('Hello World!')
    '\n    Append a list of use flags for a given package or DEPEND atom\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt \'*\' portage_config.append_use_flags "app-admin/salt[ldap, -libvirt]"\n        salt \'*\' portage_config.append_use_flags ">=app-admin/salt-0.14.1" "[\'ldap\', \'-libvirt\']"\n    '
    if not uses:
        uses = portage.dep.dep_getusedeps(atom)
    if not uses:
        return
    atom = atom[:atom.rfind('[')]
    append_to_package_conf('use', atom=atom, flags=uses, overwrite=overwrite)

def get_flags_from_package_conf(conf, atom):
    if False:
        print('Hello World!')
    "\n    Get flags for a given package or DEPEND atom.\n    Warning: This only works if the configuration files tree is in the correct\n    format (the one enforced by enforce_nice_config)\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' portage_config.get_flags_from_package_conf license salt\n    "
    if conf in SUPPORTED_CONFS:
        package_file = _get_config_file(conf, atom)
        if '/' not in atom:
            atom = _p_to_cp(atom)
        has_wildcard = '*' in atom
        if has_wildcard:
            match_list = set(atom)
        else:
            try:
                match_list = set(_porttree().dbapi.xmatch('match-all', atom))
            except AttributeError:
                return []
        flags = []
        try:
            with salt.utils.files.fopen(package_file) as fp_:
                for line in fp_:
                    line = salt.utils.stringutils.to_unicode(line).strip()
                    line_package = line.split()[0]
                    if has_wildcard:
                        found_match = line_package == atom
                    else:
                        line_list = _porttree().dbapi.xmatch('match-all', line_package)
                        found_match = match_list.issubset(line_list)
                    if found_match:
                        f_tmp = [flag for flag in line.strip().split(' ') if flag][1:]
                        if f_tmp:
                            flags.extend(f_tmp)
                        else:
                            flags.append('~ARCH')
            return _merge_flags(flags)
        except OSError:
            return []

def has_flag(conf, atom, flag):
    if False:
        i = 10
        return i + 15
    "\n    Verify if the given package or DEPEND atom has the given flag.\n    Warning: This only works if the configuration files tree is in the correct\n    format (the one enforced by enforce_nice_config)\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' portage_config.has_flag license salt Apache-2.0\n    "
    if flag in get_flags_from_package_conf(conf, atom):
        return True
    return False

def get_missing_flags(conf, atom, flags):
    if False:
        for i in range(10):
            print('nop')
    '\n    Find out which of the given flags are currently not set.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt \'*\' portage_config.get_missing_flags use salt "[\'ldap\', \'-libvirt\', \'openssl\']"\n    '
    new_flags = []
    for flag in flags:
        if not has_flag(conf, atom, flag):
            new_flags.append(flag)
    return new_flags

def has_use(atom, use):
    if False:
        while True:
            i = 10
    "\n    Verify if the given package or DEPEND atom has the given use flag.\n    Warning: This only works if the configuration files tree is in the correct\n    format (the one enforced by enforce_nice_config)\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' portage_config.has_use salt libvirt\n    "
    return has_flag('use', atom, use)

def is_present(conf, atom):
    if False:
        i = 10
        return i + 15
    "\n    Tell if a given package or DEPEND atom is present in the configuration\n    files tree.\n    Warning: This only works if the configuration files tree is in the correct\n    format (the one enforced by enforce_nice_config)\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' portage_config.is_present unmask salt\n    "
    if conf in SUPPORTED_CONFS:
        if not isinstance(atom, portage.dep.Atom):
            atom = portage.dep.Atom(atom, allow_wildcard=True)
        has_wildcard = '*' in atom
        package_file = _get_config_file(conf, str(atom))
        if has_wildcard:
            match_list = set(atom)
        else:
            match_list = set(_porttree().dbapi.xmatch('match-all', atom))
        try:
            with salt.utils.files.fopen(package_file) as fp_:
                for line in fp_:
                    line = salt.utils.stringutils.to_unicode(line).strip()
                    line_package = line.split()[0]
                    if has_wildcard:
                        if line_package == str(atom):
                            return True
                    else:
                        line_list = _porttree().dbapi.xmatch('match-all', line_package)
                        if match_list.issubset(line_list):
                            return True
        except OSError:
            pass
        return False

def get_iuse(cp):
    if False:
        while True:
            i = 10
    '\n    .. versionadded:: 2015.8.0\n\n    Gets the current IUSE flags from the tree.\n\n    @type: cpv: string\n    @param cpv: cat/pkg\n    @rtype list\n    @returns [] or the list of IUSE flags\n    '
    cpv = _get_cpv(cp)
    try:
        dirty_flags = _porttree().dbapi.aux_get(cpv, ['IUSE'])[0].split()
        return list(set(dirty_flags))
    except Exception as e:
        return []

def get_installed_use(cp, use='USE'):
    if False:
        while True:
            i = 10
    '\n    .. versionadded:: 2015.8.0\n\n    Gets the installed USE flags from the VARDB.\n\n    @type: cp: string\n    @param cp: cat/pkg\n    @type use: string\n    @param use: 1 of ["USE", "PKGUSE"]\n    @rtype list\n    @returns [] or the list of IUSE flags\n    '
    portage = _get_portage()
    cpv = _get_cpv(cp)
    return portage.db[portage.root]['vartree'].dbapi.aux_get(cpv, [use])[0].split()

def filter_flags(use, use_expand_hidden, usemasked, useforced):
    if False:
        for i in range(10):
            print('nop')
    '\n    .. versionadded:: 2015.8.0\n\n    Filter function to remove hidden or otherwise not normally\n    visible USE flags from a list.\n\n    @type use: list\n    @param use: the USE flag list to be filtered.\n    @type use_expand_hidden: list\n    @param  use_expand_hidden: list of flags hidden.\n    @type usemasked: list\n    @param usemasked: list of masked USE flags.\n    @type useforced: list\n    @param useforced: the forced USE flags.\n    @rtype: list\n    @return the filtered USE flags.\n    '
    portage = _get_portage()
    for f in use_expand_hidden:
        f = f.lower() + '_'
        for x in use:
            if f in x:
                use.remove(x)
    archlist = portage.settings['PORTAGE_ARCHLIST'].split()
    for a in use[:]:
        if a in archlist:
            use.remove(a)
    masked = usemasked + useforced
    for a in use[:]:
        if a in masked:
            use.remove(a)
    return use

def get_all_cpv_use(cp):
    if False:
        while True:
            i = 10
    '\n    .. versionadded:: 2015.8.0\n\n    Uses portage to determine final USE flags and settings for an emerge.\n\n    @type cp: string\n    @param cp: eg cat/pkg\n    @rtype: lists\n    @return  use, use_expand_hidden, usemask, useforce\n    '
    cpv = _get_cpv(cp)
    portage = _get_portage()
    use = None
    _porttree().dbapi.settings.unlock()
    try:
        _porttree().dbapi.settings.setcpv(cpv, mydb=portage.portdb)
        use = portage.settings['PORTAGE_USE'].split()
        use_expand_hidden = portage.settings['USE_EXPAND_HIDDEN'].split()
        usemask = list(_porttree().dbapi.settings.usemask)
        useforce = list(_porttree().dbapi.settings.useforce)
    except KeyError:
        _porttree().dbapi.settings.reset()
        _porttree().dbapi.settings.lock()
        return ([], [], [], [])
    _porttree().dbapi.settings.reset()
    _porttree().dbapi.settings.lock()
    return (use, use_expand_hidden, usemask, useforce)

def get_cleared_flags(cp):
    if False:
        print('Hello World!')
    '\n    .. versionadded:: 2015.8.0\n\n    Uses portage for compare use flags which is used for installing package\n    and use flags which now exist int /etc/portage/package.use/\n\n    @type cp: string\n    @param cp: eg cat/pkg\n    @rtype: tuple\n    @rparam: tuple with two lists - list of used flags and\n    list of flags which will be used\n    '
    cpv = _get_cpv(cp)
    (final_use, use_expand_hidden, usemasked, useforced) = get_all_cpv_use(cpv)
    inst_flags = filter_flags(get_installed_use(cpv), use_expand_hidden, usemasked, useforced)
    final_flags = filter_flags(final_use, use_expand_hidden, usemasked, useforced)
    return (inst_flags, final_flags)

def is_changed_uses(cp):
    if False:
        return 10
    '\n    .. versionadded:: 2015.8.0\n\n    Uses portage for determine if the use flags of installed package\n    is compatible with use flags in portage configs.\n\n    @type cp: string\n    @param cp: eg cat/pkg\n    '
    cpv = _get_cpv(cp)
    (i_flags, conf_flags) = get_cleared_flags(cpv)
    for i in i_flags:
        try:
            conf_flags.remove(i)
        except ValueError:
            return True
    return True if conf_flags else False