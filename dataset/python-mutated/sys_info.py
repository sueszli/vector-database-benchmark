from __future__ import annotations
import platform
from ansible.module_utils import distro
from ansible.module_utils.common._utils import get_all_subclasses
__all__ = ('get_distribution', 'get_distribution_version', 'get_platform_subclass')

def get_distribution():
    if False:
        for i in range(10):
            print('nop')
    '\n    Return the name of the distribution the module is running on.\n\n    :rtype: NativeString or None\n    :returns: Name of the distribution the module is running on\n\n    This function attempts to determine what distribution the code is running\n    on and return a string representing that value. If the platform is Linux\n    and the distribution cannot be determined, it returns ``OtherLinux``.\n    '
    distribution = distro.id().capitalize()
    if platform.system() == 'Linux':
        if distribution == 'Amzn':
            distribution = 'Amazon'
        elif distribution == 'Rhel':
            distribution = 'Redhat'
        elif not distribution:
            distribution = 'OtherLinux'
    return distribution

def get_distribution_version():
    if False:
        while True:
            i = 10
    '\n    Get the version of the distribution the code is running on\n\n    :rtype: NativeString or None\n    :returns: A string representation of the version of the distribution. If it\n    cannot determine the version, it returns an empty string. If this is not run on\n    a Linux machine it returns None.\n    '
    version = None
    needs_best_version = frozenset((u'centos', u'debian'))
    version = distro.version()
    distro_id = distro.id()
    if version is not None:
        if distro_id in needs_best_version:
            version_best = distro.version(best=True)
            if distro_id == u'centos':
                version = u'.'.join(version_best.split(u'.')[:2])
            if distro_id == u'debian':
                version = version_best
    else:
        version = u''
    return version

def get_distribution_codename():
    if False:
        return 10
    "\n    Return the code name for this Linux Distribution\n\n    :rtype: NativeString or None\n    :returns: A string representation of the distribution's codename or None if not a Linux distro\n    "
    codename = None
    if platform.system() == 'Linux':
        os_release_info = distro.os_release_info()
        codename = os_release_info.get('version_codename')
        if codename is None:
            codename = os_release_info.get('ubuntu_codename')
        if codename is None and distro.id() == 'ubuntu':
            lsb_release_info = distro.lsb_release_info()
            codename = lsb_release_info.get('codename')
        if codename is None:
            codename = distro.codename()
            if codename == u'':
                codename = None
    return codename

def get_platform_subclass(cls):
    if False:
        i = 10
        return i + 15
    '\n    Finds a subclass implementing desired functionality on the platform the code is running on\n\n    :arg cls: Class to find an appropriate subclass for\n    :returns: A class that implements the functionality on this platform\n\n    Some Ansible modules have different implementations depending on the platform they run on.  This\n    function is used to select between the various implementations and choose one.  You can look at\n    the implementation of the Ansible :ref:`User module<user_module>` module for an example of how to use this.\n\n    This function replaces ``basic.load_platform_subclass()``.  When you port code, you need to\n    change the callers to be explicit about instantiating the class.  For instance, code in the\n    Ansible User module changed from::\n\n    .. code-block:: python\n\n        # Old\n        class User:\n            def __new__(cls, args, kwargs):\n                return load_platform_subclass(User, args, kwargs)\n\n        # New\n        class User:\n            def __new__(cls, *args, **kwargs):\n                new_cls = get_platform_subclass(User)\n                return super(cls, new_cls).__new__(new_cls)\n    '
    this_platform = platform.system()
    distribution = get_distribution()
    subclass = None
    if distribution is not None:
        for sc in get_all_subclasses(cls):
            if sc.distribution is not None and sc.distribution == distribution and (sc.platform == this_platform):
                subclass = sc
    if subclass is None:
        for sc in get_all_subclasses(cls):
            if sc.platform == this_platform and sc.distribution is None:
                subclass = sc
    if subclass is None:
        subclass = cls
    return subclass