"""
Support for Gentoolkit

"""
import os
HAS_GENTOOLKIT = False
try:
    from gentoolkit.eclean import clean, cli
    from gentoolkit.eclean import exclude as excludemod
    from gentoolkit.eclean import search
    HAS_GENTOOLKIT = True
except ImportError:
    pass
__virtualname__ = 'gentoolkit'

def __virtual__():
    if False:
        i = 10
        return i + 15
    '\n    Only work on Gentoo systems with gentoolkit installed\n    '
    if __grains__['os'] == 'Gentoo' and HAS_GENTOOLKIT:
        return __virtualname__
    return (False, 'The gentoolkitmod execution module cannot be loaded: either the system is not Gentoo or the gentoolkit.eclean python module not available')

def revdep_rebuild(lib=None):
    if False:
        i = 10
        return i + 15
    "\n    Fix up broken reverse dependencies\n\n    lib\n        Search for reverse dependencies for a particular library rather\n        than every library on the system. It can be a full path to a\n        library or basic regular expression.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' gentoolkit.revdep_rebuild\n    "
    cmd = 'revdep-rebuild -i --quiet --no-progress'
    if lib is not None:
        cmd += ' --library={}'.format(lib)
    return __salt__['cmd.retcode'](cmd, python_shell=False) == 0

def _pretty_size(size):
    if False:
        while True:
            i = 10
    '\n    Print sizes in a similar fashion as eclean\n    '
    units = [' G', ' M', ' K', ' B']
    while units and size >= 1000:
        size = size / 1024.0
        units.pop()
    return '{}{}'.format(round(size, 1), units[-1])

def _parse_exclude(exclude_file):
    if False:
        print('Hello World!')
    '\n    Parse an exclude file.\n\n    Returns a dict as defined in gentoolkit.eclean.exclude.parseExcludeFile\n    '
    if os.path.isfile(exclude_file):
        exclude = excludemod.parseExcludeFile(exclude_file, lambda x: None)
    else:
        exclude = dict()
    return exclude

def eclean_dist(destructive=False, package_names=False, size_limit=0, time_limit=0, fetch_restricted=False, exclude_file='/etc/eclean/distfiles.exclude'):
    if False:
        return 10
    '\n    Clean obsolete portage sources\n\n    destructive\n        Only keep minimum for reinstallation\n\n    package_names\n        Protect all versions of installed packages. Only meaningful if used\n        with destructive=True\n\n    size_limit <size>\n        Don\'t delete distfiles bigger than <size>.\n        <size> is a size specification: "10M" is "ten megabytes",\n        "200K" is "two hundreds kilobytes", etc. Units are: G, M, K and B.\n\n    time_limit <time>\n        Don\'t delete distfiles files modified since <time>\n        <time> is an amount of time: "1y" is "one year", "2w" is\n        "two weeks", etc. Units are: y (years), m (months), w (weeks),\n        d (days) and h (hours).\n\n    fetch_restricted\n        Protect fetch-restricted files. Only meaningful if used with\n        destructive=True\n\n    exclude_file\n        Path to exclusion file. Default is /etc/eclean/distfiles.exclude\n        This is the same default eclean-dist uses. Use None if this file\n        exists and you want to ignore.\n\n    Returns a dict containing the cleaned, saved, and deprecated dists:\n\n    .. code-block:: python\n\n        {\'cleaned\': {<dist file>: <size>},\n         \'deprecated\': {<package>: <dist file>},\n         \'saved\': {<package>: <dist file>},\n         \'total_cleaned\': <size>}\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt \'*\' gentoolkit.eclean_dist destructive=True\n    '
    if exclude_file is None:
        exclude = None
    else:
        try:
            exclude = _parse_exclude(exclude_file)
        except excludemod.ParseExcludeFileException as e:
            ret = {e: 'Invalid exclusion file: {}'.format(exclude_file)}
            return ret
    if time_limit != 0:
        time_limit = cli.parseTime(time_limit)
    if size_limit != 0:
        size_limit = cli.parseSize(size_limit)
    clean_size = 0
    engine = search.DistfilesSearch(lambda x: None)
    (clean_me, saved, deprecated) = engine.findDistfiles(destructive=destructive, package_names=package_names, size_limit=size_limit, time_limit=time_limit, fetch_restricted=fetch_restricted, exclude=exclude)
    cleaned = dict()

    def _eclean_progress_controller(size, key, *args):
        if False:
            while True:
                i = 10
        cleaned[key] = _pretty_size(size)
        return True
    if clean_me:
        cleaner = clean.CleanUp(_eclean_progress_controller)
        clean_size = cleaner.clean_dist(clean_me)
    ret = {'cleaned': cleaned, 'saved': saved, 'deprecated': deprecated, 'total_cleaned': _pretty_size(clean_size)}
    return ret

def eclean_pkg(destructive=False, package_names=False, time_limit=0, exclude_file='/etc/eclean/packages.exclude'):
    if False:
        for i in range(10):
            print('nop')
    '\n    Clean obsolete binary packages\n\n    destructive\n        Only keep minimum for reinstallation\n\n    package_names\n        Protect all versions of installed packages. Only meaningful if used\n        with destructive=True\n\n    time_limit <time>\n        Don\'t delete distfiles files modified since <time>\n        <time> is an amount of time: "1y" is "one year", "2w" is\n        "two weeks", etc. Units are: y (years), m (months), w (weeks),\n        d (days) and h (hours).\n\n    exclude_file\n        Path to exclusion file. Default is /etc/eclean/packages.exclude\n        This is the same default eclean-pkg uses. Use None if this file\n        exists and you want to ignore.\n\n    Returns a dict containing the cleaned binary packages:\n\n    .. code-block:: python\n\n        {\'cleaned\': {<dist file>: <size>},\n         \'total_cleaned\': <size>}\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt \'*\' gentoolkit.eclean_pkg destructive=True\n    '
    if exclude_file is None:
        exclude = None
    else:
        try:
            exclude = _parse_exclude(exclude_file)
        except excludemod.ParseExcludeFileException as e:
            ret = {e: 'Invalid exclusion file: {}'.format(exclude_file)}
            return ret
    if time_limit != 0:
        time_limit = cli.parseTime(time_limit)
    clean_size = 0
    clean_me = search.findPackages(None, destructive=destructive, package_names=package_names, time_limit=time_limit, exclude=exclude, pkgdir=search.pkgdir)
    cleaned = dict()

    def _eclean_progress_controller(size, key, *args):
        if False:
            i = 10
            return i + 15
        cleaned[key] = _pretty_size(size)
        return True
    if clean_me:
        cleaner = clean.CleanUp(_eclean_progress_controller)
        clean_size = cleaner.clean_pkgs(clean_me, search.pkgdir)
    ret = {'cleaned': cleaned, 'total_cleaned': _pretty_size(clean_size)}
    return ret

def _glsa_list_process_output(output):
    if False:
        while True:
            i = 10
    '\n    Process output from glsa_check_list into a dict\n\n    Returns a dict containing the glsa id, description, status, and CVEs\n    '
    ret = dict()
    for line in output:
        try:
            (glsa_id, status, desc) = line.split(None, 2)
            if 'U' in status:
                status += ' Not Affected'
            elif 'N' in status:
                status += ' Might be Affected'
            elif 'A' in status:
                status += ' Applied (injected)'
            if 'CVE' in desc:
                (desc, cves) = desc.rsplit(None, 1)
                cves = cves.split(',')
            else:
                cves = list()
            ret[glsa_id] = {'description': desc, 'status': status, 'CVEs': cves}
        except ValueError:
            pass
    return ret

def glsa_check_list(glsa_list):
    if False:
        for i in range(10):
            print('nop')
    "\n    List the status of Gentoo Linux Security Advisories\n\n    glsa_list\n         can contain an arbitrary number of GLSA ids, filenames\n         containing GLSAs or the special identifiers 'all' and 'affected'\n\n    Returns a dict containing glsa ids with a description, status, and CVEs:\n\n    .. code-block:: python\n\n        {<glsa_id>: {'description': <glsa_description>,\n         'status': <glsa status>,\n         'CVEs': [<list of CVEs>]}}\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' gentoolkit.glsa_check_list 'affected'\n    "
    cmd = 'glsa-check --quiet --nocolor --cve --list '
    if isinstance(glsa_list, list):
        for glsa in glsa_list:
            cmd += glsa + ' '
    elif glsa_list == 'all' or glsa_list == 'affected':
        cmd += glsa_list
    ret = dict()
    out = __salt__['cmd.run'](cmd, python_shell=False).split('\n')
    ret = _glsa_list_process_output(out)
    return ret