"""
Module for managing container and VM images

.. versionadded:: 2014.7.0
"""
import logging
import os
import pprint
import shlex
import uuid
import salt.syspaths
import salt.utils.kickstart
import salt.utils.path
import salt.utils.preseed
import salt.utils.stringutils
import salt.utils.validate.path
import salt.utils.yast
from salt.exceptions import SaltInvocationError
log = logging.getLogger(__name__)
CMD_MAP = {'yum': ('yum', 'rpm'), 'deb': ('debootstrap',), 'pacman': ('pacman',)}
EPEL_URL = 'http://download.fedoraproject.org/pub/epel/6/i386/epel-release-6-8.noarch.rpm'

def __virtual__():
    if False:
        for i in range(10):
            print('nop')
    '\n    By default, this will be available on all platforms; but not all distros\n    will necessarily be supported\n    '
    return True

def bootstrap(platform, root, img_format='dir', fs_format='ext2', fs_opts=None, arch=None, flavor=None, repo_url=None, static_qemu=None, img_size=None, mount_dir=None, pkg_cache=None, pkgs=None, exclude_pkgs=None, epel_url=EPEL_URL):
    if False:
        for i in range(10):
            print('nop')
    '\n    Create an image for a specific platform.\n\n    Please note that this function *MUST* be run as root, as images that are\n    created make files belonging to root.\n\n    platform\n        Which platform to use to create the image. Currently supported platforms\n        are rpm, deb and pacman.\n\n    root\n        Local path to create the root of the image filesystem.\n\n    img_format\n        Which format to create the image in. By default, just copies files into\n        a directory on the local filesystem (``dir``). Future support will exist\n        for ``sparse``.\n\n    fs_format\n        When using a non-``dir`` ``img_format``, which filesystem to format the\n        image to. By default, ``ext2``.\n\n    fs_opts\n        When using a non-``dir`` ``img_format``, a dict of opts may be\n        specified.\n\n    arch\n        Architecture to install packages for, if supported by the underlying\n        bootstrap tool. Currently only used for deb.\n\n    flavor\n        Which flavor of operating system to install. This correlates to a\n        specific directory on the distribution repositories. For instance,\n        ``wheezy`` on Debian.\n\n    repo_url\n        Mainly important for Debian-based repos. Base URL for the mirror to\n        install from. (e.x.: http://ftp.debian.org/debian/)\n\n    static_qemu\n        Local path to the static qemu binary required for this arch.\n        (e.x.: /usr/bin/qemu-amd64-static)\n\n    pkg_confs\n        The location of the conf files to copy into the image, to point the\n        installer to the right repos and configuration.\n\n    img_size\n        If img_format is not ``dir``, then the size of the image must be\n        specified.\n\n    mount_dir\n        If img_format is not ``dir``, then the image must be mounted somewhere.\n        If the ``mount_dir`` is not specified, then it will be created at\n        ``/opt/salt-genesis.<random_uuid>``. This directory will be unmounted\n        and removed when the process is finished.\n\n    pkg_cache\n        This points to a directory containing a cache of package files to be\n        copied to the image. It does not need to be specified.\n\n    pkgs\n        A list of packages to be installed on this image. For RedHat, this\n        will include ``yum``, ``centos-release`` and ``iputils`` by default.\n\n    exclude_pkgs\n        A list of packages to be excluded. If you do not want to install the\n        defaults, you need to include them in this list.\n\n    epel_url\n        The URL to download the EPEL release package from.\n\n    CLI Examples:\n\n    .. code-block:: bash\n\n        salt myminion genesis.bootstrap pacman /root/arch\n        salt myminion genesis.bootstrap rpm /root/redhat\n        salt myminion genesis.bootstrap deb /root/wheezy arch=amd64             flavor=wheezy static_qemu=/usr/bin/qemu-x86_64-static\n\n    '
    if img_format not in ('dir', 'sparse'):
        raise SaltInvocationError('The img_format must be "sparse" or "dir"')
    if img_format == 'dir':
        if not __salt__['file.directory_exists'](root):
            try:
                __salt__['file.mkdir'](root)
            except Exception as exc:
                return {'Error': salt.utils.stringutils.to_unicode(pprint.pformat(exc))}
    elif img_format == 'sparse':
        if not img_size:
            raise SaltInvocationError('An img_size must be specified for a sparse file')
        if not mount_dir:
            mount_dir = '/opt/salt-genesis.{}'.format(uuid.uuid4())
        __salt__['file.mkdir'](mount_dir, 'root', 'root', '755')
        __salt__['cmd.run'](('fallocate', '-l', img_size, root), python_shell=False)
        _mkpart(root, fs_format, fs_opts, mount_dir)
        loop1 = __salt__['cmd.run']('losetup -f')
        log.debug('First loop device is %s', loop1)
        __salt__['cmd.run']('losetup {} {}'.format(loop1, root))
        loop2 = __salt__['cmd.run']('losetup -f')
        log.debug('Second loop device is %s', loop2)
        start = str(2048 * 2048)
        __salt__['cmd.run']('losetup -o {} {} {}'.format(start, loop2, loop1))
        __salt__['mount.mount'](mount_dir, loop2)
        _populate_cache(platform, pkg_cache, mount_dir)
    if mount_dir:
        root = mount_dir
    if pkgs is None:
        pkgs = []
    if exclude_pkgs is None:
        exclude_pkgs = []
    if platform in ('rpm', 'yum'):
        _bootstrap_yum(root, pkgs=pkgs, exclude_pkgs=exclude_pkgs, epel_url=epel_url)
    elif platform == 'deb':
        _bootstrap_deb(root, arch=arch, flavor=flavor, repo_url=repo_url, static_qemu=static_qemu, pkgs=pkgs, exclude_pkgs=exclude_pkgs)
    elif platform == 'pacman':
        _bootstrap_pacman(root, img_format=img_format, pkgs=pkgs, exclude_pkgs=exclude_pkgs)
    if img_format != 'dir':
        blkinfo = __salt__['disk.blkid'](loop2)
        __salt__['file.replace']('{}/boot/grub/grub.cfg'.format(mount_dir), 'ad4103fa-d940-47ca-8506-301d8071d467', blkinfo[loop2]['UUID'])
        __salt__['mount.umount'](root)
        __salt__['cmd.run']('losetup -d {}'.format(loop2))
        __salt__['cmd.run']('losetup -d {}'.format(loop1))
        __salt__['file.rmdir'](mount_dir)

def _mkpart(root, fs_format, fs_opts, mount_dir):
    if False:
        print('Hello World!')
    '\n    Make a partition, and make it bootable\n\n    .. versionadded:: 2015.8.0\n    '
    __salt__['partition.mklabel'](root, 'msdos')
    loop1 = __salt__['cmd.run']('losetup -f')
    log.debug('First loop device is %s', loop1)
    __salt__['cmd.run']('losetup {} {}'.format(loop1, root))
    part_info = __salt__['partition.list'](loop1)
    start = str(2048 * 2048) + 'B'
    end = part_info['info']['size']
    __salt__['partition.mkpart'](loop1, 'primary', start=start, end=end)
    __salt__['partition.set'](loop1, '1', 'boot', 'on')
    part_info = __salt__['partition.list'](loop1)
    loop2 = __salt__['cmd.run']('losetup -f')
    log.debug('Second loop device is %s', loop2)
    start = start.rstrip('B')
    __salt__['cmd.run']('losetup -o {} {} {}'.format(start, loop2, loop1))
    _mkfs(loop2, fs_format, fs_opts)
    __salt__['mount.mount'](mount_dir, loop2)
    __salt__['cmd.run'](('grub-install', '--target=i386-pc', '--debug', '--no-floppy', '--modules=part_msdos linux', '--boot-directory={}/boot'.format(mount_dir), loop1), python_shell=False)
    __salt__['mount.umount'](mount_dir)
    __salt__['cmd.run']('losetup -d {}'.format(loop2))
    __salt__['cmd.run']('losetup -d {}'.format(loop1))
    return part_info

def _mkfs(root, fs_format, fs_opts=None):
    if False:
        for i in range(10):
            print('nop')
    '\n    Make a filesystem using the appropriate module\n\n    .. versionadded:: 2015.8.0\n    '
    if fs_opts is None:
        fs_opts = {}
    if fs_format in ('ext2', 'ext3', 'ext4'):
        __salt__['extfs.mkfs'](root, fs_format, **fs_opts)
    elif fs_format in ('btrfs',):
        __salt__['btrfs.mkfs'](root, **fs_opts)
    elif fs_format in ('xfs',):
        __salt__['xfs.mkfs'](root, **fs_opts)

def _populate_cache(platform, pkg_cache, mount_dir):
    if False:
        i = 10
        return i + 15
    '\n    If a ``pkg_cache`` directory is specified, then use it to populate the\n    disk image.\n    '
    if not pkg_cache:
        return
    if not os.path.isdir(pkg_cache):
        return
    if platform == 'pacman':
        cache_dir = '{}/var/cache/pacman/pkg'.format(mount_dir)
    __salt__['file.mkdir'](cache_dir, 'root', 'root', '755')
    __salt__['file.copy'](pkg_cache, cache_dir, recurse=True, remove_existing=True)

def _bootstrap_yum(root, pkg_confs='/etc/yum*', pkgs=None, exclude_pkgs=None, epel_url=EPEL_URL):
    if False:
        while True:
            i = 10
    '\n    Bootstrap an image using the yum tools\n\n    root\n        The root of the image to install to. Will be created as a directory if\n        it does not exist. (e.x.: /root/arch)\n\n    pkg_confs\n        The location of the conf files to copy into the image, to point yum\n        to the right repos and configuration.\n\n    pkgs\n        A list of packages to be installed on this image. For RedHat, this\n        will include ``yum``, ``centos-release`` and ``iputils`` by default.\n\n    exclude_pkgs\n        A list of packages to be excluded. If you do not want to install the\n        defaults, you need to include them in this list.\n\n    epel_url\n        The URL to download the EPEL release package from.\n\n    TODO: Set up a pre-install overlay, to copy files into /etc/ and so on,\n        which are required for the install to work.\n    '
    if pkgs is None:
        pkgs = []
    elif isinstance(pkgs, str):
        pkgs = pkgs.split(',')
    default_pkgs = ('yum', 'centos-release', 'iputils')
    for pkg in default_pkgs:
        if pkg not in pkgs:
            pkgs.append(pkg)
    if exclude_pkgs is None:
        exclude_pkgs = []
    elif isinstance(exclude_pkgs, str):
        exclude_pkgs = exclude_pkgs.split(',')
    for pkg in exclude_pkgs:
        pkgs.remove(pkg)
    _make_nodes(root)
    release_files = [rf for rf in os.listdir('/etc') if rf.endswith('release')]
    __salt__['cmd.run']('cp /etc/resolv/conf {rfs} {root}/etc'.format(root=shlex.quote(root), rfs=' '.join(release_files)))
    __salt__['cmd.run']('cp -r {rfs} {root}/etc'.format(root=shlex.quote(root), rfs=' '.join(release_files)))
    __salt__['cmd.run']('cp -r {confs} {root}/etc'.format(root=shlex.quote(root), confs=shlex.quote(pkg_confs)))
    yum_args = ['yum', 'install', '--installroot={}'.format(shlex.quote(root)), '-y'] + pkgs
    __salt__['cmd.run'](yum_args, python_shell=False)
    if 'epel-release' not in exclude_pkgs:
        __salt__['cmd.run'](('rpm', '--root={}'.format(shlex.quote(root)), '-Uvh', epel_url), python_shell=False)

def _bootstrap_deb(root, arch, flavor, repo_url=None, static_qemu=None, pkgs=None, exclude_pkgs=None):
    if False:
        while True:
            i = 10
    '\n    Bootstrap an image using the Debian tools\n\n    root\n        The root of the image to install to. Will be created as a directory if\n        it does not exist. (e.x.: /root/wheezy)\n\n    arch\n        Architecture of the target image. (e.x.: amd64)\n\n    flavor\n        Flavor of Debian to install. (e.x.: wheezy)\n\n    repo_url\n        Base URL for the mirror to install from.\n        (e.x.: http://ftp.debian.org/debian/)\n\n    static_qemu\n        Local path to the static qemu binary required for this arch.\n        (e.x.: /usr/bin/qemu-amd64-static)\n\n    pkgs\n        A list of packages to be installed on this image.\n\n    exclude_pkgs\n        A list of packages to be excluded.\n    '
    if repo_url is None:
        repo_url = 'http://ftp.debian.org/debian/'
    if not salt.utils.path.which('debootstrap'):
        log.error('Required tool debootstrap is not installed.')
        return False
    if static_qemu and (not salt.utils.validate.path.is_executable(static_qemu)):
        log.error('Required tool qemu not present/readable at: %s', static_qemu)
        return False
    if isinstance(pkgs, (list, tuple)):
        pkgs = ','.join(pkgs)
    if isinstance(exclude_pkgs, (list, tuple)):
        exclude_pkgs = ','.join(exclude_pkgs)
    deb_args = ['debootstrap', '--foreign', '--arch', shlex.quote(arch)]
    if pkgs:
        deb_args += ['--include', shlex.quote(pkgs)]
    if exclude_pkgs:
        deb_args += ['--exclude', shlex.quote(exclude_pkgs)]
    deb_args += [shlex.quote(flavor), shlex.quote(root), shlex.quote(repo_url)]
    __salt__['cmd.run'](deb_args, python_shell=False)
    if static_qemu:
        __salt__['cmd.run']('cp {qemu} {root}/usr/bin/'.format(qemu=shlex.quote(static_qemu), root=shlex.quote(root)))
    env = {'DEBIAN_FRONTEND': 'noninteractive', 'DEBCONF_NONINTERACTIVE_SEEN': 'true', 'LC_ALL': 'C', 'LANGUAGE': 'C', 'LANG': 'C', 'PATH': '/sbin:/bin:/usr/bin'}
    __salt__['cmd.run']('chroot {root} /debootstrap/debootstrap --second-stage'.format(root=shlex.quote(root)), env=env)
    __salt__['cmd.run']('chroot {root} dpkg --configure -a'.format(root=shlex.quote(root)), env=env)

def _bootstrap_pacman(root, pkg_confs='/etc/pacman*', img_format='dir', pkgs=None, exclude_pkgs=None):
    if False:
        while True:
            i = 10
    '\n    Bootstrap an image using the pacman tools\n\n    root\n        The root of the image to install to. Will be created as a directory if\n        it does not exist. (e.x.: /root/arch)\n\n    pkg_confs\n        The location of the conf files to copy into the image, to point pacman\n        to the right repos and configuration.\n\n    img_format\n        The image format to be used. The ``dir`` type needs no special\n        treatment, but others need special treatment.\n\n    pkgs\n        A list of packages to be installed on this image. For Arch Linux, this\n        will include ``pacman``, ``linux``, ``grub``, and ``systemd-sysvcompat``\n        by default.\n\n    exclude_pkgs\n        A list of packages to be excluded. If you do not want to install the\n        defaults, you need to include them in this list.\n    '
    _make_nodes(root)
    if pkgs is None:
        pkgs = []
    elif isinstance(pkgs, str):
        pkgs = pkgs.split(',')
    default_pkgs = ('pacman', 'linux', 'systemd-sysvcompat', 'grub')
    for pkg in default_pkgs:
        if pkg not in pkgs:
            pkgs.append(pkg)
    if exclude_pkgs is None:
        exclude_pkgs = []
    elif isinstance(exclude_pkgs, str):
        exclude_pkgs = exclude_pkgs.split(',')
    for pkg in exclude_pkgs:
        pkgs.remove(pkg)
    if img_format != 'dir':
        __salt__['mount.mount']('{}/proc'.format(root), '/proc', fstype='', opts='bind')
        __salt__['mount.mount']('{}/dev'.format(root), '/dev', fstype='', opts='bind')
    __salt__['file.mkdir']('{}/var/lib/pacman/local'.format(root), 'root', 'root', '755')
    pac_files = [rf for rf in os.listdir('/etc') if rf.startswith('pacman.')]
    for pac_file in pac_files:
        __salt__['cmd.run']('cp -r /etc/{} {}/etc'.format(pac_file, shlex.quote(root)))
    __salt__['file.copy']('/var/lib/pacman/sync', '{}/var/lib/pacman/sync'.format(root), recurse=True)
    pacman_args = ['pacman', '--noconfirm', '-r', shlex.quote(root), '-S'] + pkgs
    __salt__['cmd.run'](pacman_args, python_shell=False)
    if img_format != 'dir':
        __salt__['mount.umount']('{}/proc'.format(root))
        __salt__['mount.umount']('{}/dev'.format(root))

def _make_nodes(root):
    if False:
        i = 10
        return i + 15
    '\n    Make the minimum number of nodes inside of /dev/. Based on:\n\n    https://wiki.archlinux.org/index.php/Linux_Containers\n    '
    dirs = (('{}/etc'.format(root), 'root', 'root', '755'), ('{}/dev'.format(root), 'root', 'root', '755'), ('{}/proc'.format(root), 'root', 'root', '755'), ('{}/dev/pts'.format(root), 'root', 'root', '755'), ('{}/dev/shm'.format(root), 'root', 'root', '1755'))
    nodes = (('{}/dev/null'.format(root), 'c', 1, 3, 'root', 'root', '666'), ('{}/dev/zero'.format(root), 'c', 1, 5, 'root', 'root', '666'), ('{}/dev/random'.format(root), 'c', 1, 8, 'root', 'root', '666'), ('{}/dev/urandom'.format(root), 'c', 1, 9, 'root', 'root', '666'), ('{}/dev/tty'.format(root), 'c', 5, 0, 'root', 'root', '666'), ('{}/dev/tty0'.format(root), 'c', 4, 0, 'root', 'root', '666'), ('{}/dev/console'.format(root), 'c', 5, 1, 'root', 'root', '600'), ('{}/dev/full'.format(root), 'c', 1, 7, 'root', 'root', '666'), ('{}/dev/initctl'.format(root), 'p', 0, 0, 'root', 'root', '600'), ('{}/dev/ptmx'.format(root), 'c', 5, 2, 'root', 'root', '666'))
    for path in dirs:
        __salt__['file.mkdir'](*path)
    for path in nodes:
        __salt__['file.mknod'](*path)

def avail_platforms():
    if False:
        return 10
    '\n    Return which platforms are available\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt myminion genesis.avail_platforms\n    '
    ret = {}
    for platform in CMD_MAP:
        ret[platform] = True
        for cmd in CMD_MAP[platform]:
            if not salt.utils.path.which(cmd):
                ret[platform] = False
    return ret

def pack(name, root, path=None, pack_format='tar', compress='bzip2'):
    if False:
        i = 10
        return i + 15
    "\n    Pack up a directory structure, into a specific format\n\n    CLI Examples:\n\n    .. code-block:: bash\n\n        salt myminion genesis.pack centos /root/centos\n        salt myminion genesis.pack centos /root/centos pack_format='tar'\n    "
    if pack_format == 'tar':
        _tar(name, root, path, compress)

def unpack(name, dest=None, path=None, pack_format='tar', compress='bz2'):
    if False:
        i = 10
        return i + 15
    '\n    Unpack an image into a directory structure\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt myminion genesis.unpack centos /root/centos\n    '
    if pack_format == 'tar':
        _untar(name, dest, path, compress)

def _tar(name, root, path=None, compress='bzip2'):
    if False:
        print('Hello World!')
    '\n    Pack up image in a tar format\n    '
    if path is None:
        path = os.path.join(salt.syspaths.BASE_FILE_ROOTS_DIR, 'img')
    if not __salt__['file.directory_exists'](path):
        try:
            __salt__['file.mkdir'](path)
        except Exception as exc:
            return {'Error': salt.utils.stringutils.to_unicode(pprint.pformat(exc))}
    (compression, ext) = _compress(compress)
    tarfile = '{}/{}.tar.{}'.format(path, name, ext)
    out = __salt__['archive.tar'](options='{}pcf'.format(compression), tarfile=tarfile, sources='.', dest=root)

def _untar(name, dest=None, path=None, compress='bz2'):
    if False:
        while True:
            i = 10
    '\n    Unpack a tarball to be used as a container\n    '
    if path is None:
        path = os.path.join(salt.syspaths.BASE_FILE_ROOTS_DIR, 'img')
    if not dest:
        dest = path
    if not __salt__['file.directory_exists'](dest):
        try:
            __salt__['file.mkdir'](dest)
        except Exception as exc:
            return {'Error': salt.utils.stringutils.to_unicode(pprint.pformat(exc))}
    (compression, ext) = _compress(compress)
    tarfile = '{}/{}.tar.{}'.format(path, name, ext)
    out = __salt__['archive.tar'](options='{}xf'.format(compression), tarfile=tarfile, dest=dest)

def _compress(compress):
    if False:
        while True:
            i = 10
    '\n    Resolve compression flags\n    '
    if compress in ('bz2', 'bzip2', 'j'):
        compression = 'j'
        ext = 'bz2'
    elif compress in ('gz', 'gzip', 'z'):
        compression = 'z'
        ext = 'gz'
    elif compress in ('xz', 'a', 'J'):
        compression = 'J'
        ext = 'xz'
    return (compression, ext)

def ldd_deps(filename, ret=None):
    if False:
        for i in range(10):
            print('nop')
    '\n    Recurse through a set of dependencies reported by ``ldd``, to find\n    associated dependencies.\n\n    Please note that this does not necessarily resolve all (non-package)\n    dependencies for a file; but it does help.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt myminion genesis.ldd_deps bash\n        salt myminion genesis.ldd_deps /bin/bash\n    '
    if not os.path.exists(filename):
        filename = salt.utils.path.which(filename)
    if ret is None:
        ret = []
    out = __salt__['cmd.run'](('ldd', filename), python_shell=False)
    for line in out.splitlines():
        if not line.strip():
            continue
        dep_path = ''
        if '=>' in line:
            comps = line.split(' => ')
            dep_comps = comps[1].strip().split()
            if os.path.exists(dep_comps[0]):
                dep_path = dep_comps[0]
        else:
            dep_comps = line.strip().split()
            if os.path.exists(dep_comps[0]):
                dep_path = dep_comps[0]
        if dep_path:
            if dep_path not in ret:
                ret.append(dep_path)
                new_deps = ldd_deps(dep_path, ret)
                for dep in new_deps:
                    if dep not in ret:
                        ret.append(dep)
    return ret

def mksls(fmt, src, dst=None):
    if False:
        for i in range(10):
            print('nop')
    '\n    Convert an installation file/script to an SLS file. Currently supports\n    ``kickstart``, ``preseed``, and ``autoyast``.\n\n    CLI Examples:\n\n    .. code-block:: bash\n\n        salt <minion> genesis.mksls kickstart /path/to/kickstart.cfg\n        salt <minion> genesis.mksls kickstart /path/to/kickstart.cfg /path/to/dest.sls\n\n    .. versionadded:: 2015.8.0\n    '
    if fmt == 'kickstart':
        return salt.utils.kickstart.mksls(src, dst)
    elif fmt == 'preseed':
        return salt.utils.preseed.mksls(src, dst)
    elif fmt == 'autoyast':
        return salt.utils.yast.mksls(src, dst)