"""
Salt module to manage Unix cryptsetup jobs and the crypttab file

.. versionadded:: 2018.3.0
"""
import logging
import os
import re
import salt.utils.files
import salt.utils.platform
import salt.utils.stringutils
from salt.exceptions import CommandExecutionError
log = logging.getLogger(__name__)
__virtualname__ = 'cryptdev'

def __virtual__():
    if False:
        while True:
            i = 10
    '\n    Only load on POSIX-like systems\n    '
    if salt.utils.platform.is_windows():
        return (False, 'The cryptdev module cannot be loaded: not a POSIX-like system')
    return True

class _crypttab_entry:
    """
    Utility class for manipulating crypttab entries. Primarily we're parsing,
    formatting, and comparing lines. Parsing emits dicts expected from
    crypttab() or raises a ValueError.
    """

    class ParseError(ValueError):
        """Error raised when a line isn't parsible as a crypttab entry"""
    crypttab_keys = ('name', 'device', 'password', 'options')
    crypttab_format = '{name: <12} {device: <44} {password: <22} {options}\n'

    @classmethod
    def dict_from_line(cls, line, keys=crypttab_keys):
        if False:
            while True:
                i = 10
        if len(keys) != 4:
            raise ValueError('Invalid key array: {}'.format(keys))
        if line.startswith('#'):
            raise cls.ParseError('Comment!')
        comps = line.split()
        if len(comps) == 3:
            comps += ['']
        if len(comps) != 4:
            raise cls.ParseError('Invalid Entry!')
        return dict(zip(keys, comps))

    @classmethod
    def from_line(cls, *args, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        return cls(**cls.dict_from_line(*args, **kwargs))

    @classmethod
    def dict_to_line(cls, entry):
        if False:
            for i in range(10):
                print('nop')
        return cls.crypttab_format.format(**entry)

    def __str__(self):
        if False:
            for i in range(10):
                print('nop')
        'String value, only works for full repr'
        return self.dict_to_line(self.criteria)

    def __repr__(self):
        if False:
            return 10
        'Always works'
        return repr(self.criteria)

    def pick(self, keys):
        if False:
            while True:
                i = 10
        'Returns an instance with just those keys'
        subset = {key: self.criteria[key] for key in keys}
        return self.__class__(**subset)

    def __init__(self, **criteria):
        if False:
            while True:
                i = 10
        'Store non-empty, non-null values to use as filter'
        self.criteria = {key: salt.utils.stringutils.to_unicode(value) for (key, value) in criteria.items() if value is not None}

    @staticmethod
    def norm_path(path):
        if False:
            print('Hello World!')
        'Resolve equivalent paths equivalently'
        return os.path.normcase(os.path.normpath(path))

    def match(self, line):
        if False:
            i = 10
            return i + 15
        'Compare potentially partial criteria against a complete line'
        entry = self.dict_from_line(line)
        for (key, value) in self.criteria.items():
            if entry[key] != value:
                return False
        return True

def active():
    if False:
        i = 10
        return i + 15
    '\n    List existing device-mapper device details.\n    '
    ret = {}
    devices = __salt__['cmd.run_stdout']('dmsetup ls --target crypt')
    out_regex = re.compile('(?P<devname>\\S+)\\s+\\((?P<major>\\d+), (?P<minor>\\d+)\\)')
    log.debug(devices)
    for line in devices.split('\n'):
        match = out_regex.match(line)
        if match:
            dev_info = match.groupdict()
            ret[dev_info['devname']] = dev_info
        else:
            log.warning('dmsetup output does not match expected format')
    return ret

def crypttab(config='/etc/crypttab'):
    if False:
        return 10
    "\n    List the contents of the crypttab\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' cryptdev.crypttab\n    "
    ret = {}
    if not os.path.isfile(config):
        return ret
    with salt.utils.files.fopen(config) as ifile:
        for line in ifile:
            line = salt.utils.stringutils.to_unicode(line).rstrip('\n')
            try:
                entry = _crypttab_entry.dict_from_line(line)
                entry['options'] = entry['options'].split(',')
                while entry['name'] in ret:
                    entry['name'] += '_'
                ret[entry.pop('name')] = entry
            except _crypttab_entry.ParseError:
                pass
    return ret

def rm_crypttab(name, config='/etc/crypttab'):
    if False:
        print('Hello World!')
    "\n    Remove the named mapping from the crypttab. If the described entry does not\n    exist, nothing is changed, but the command succeeds by returning\n    ``'absent'``. If a line is removed, it returns ``'change'``.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' cryptdev.rm_crypttab foo\n    "
    modified = False
    criteria = _crypttab_entry(name=name)
    lines = []
    try:
        with salt.utils.files.fopen(config, 'r') as ifile:
            for line in ifile:
                line = salt.utils.stringutils.to_unicode(line)
                try:
                    if criteria.match(line):
                        modified = True
                    else:
                        lines.append(line)
                except _crypttab_entry.ParseError:
                    lines.append(line)
    except OSError as exc:
        msg = 'Could not read from {0}: {1}'
        raise CommandExecutionError(msg.format(config, exc))
    if modified:
        try:
            with salt.utils.files.fopen(config, 'w+') as ofile:
                ofile.writelines((salt.utils.stringutils.to_str(line) for line in lines))
        except OSError as exc:
            msg = 'Could not write to {0}: {1}'
            raise CommandExecutionError(msg.format(config, exc))
    return 'change' if modified else 'absent'

def set_crypttab(name, device, password='none', options='', config='/etc/crypttab', test=False, match_on='name'):
    if False:
        i = 10
        return i + 15
    "\n    Verify that this device is represented in the crypttab, change the device to\n    match the name passed, or add the name if it is not present.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' cryptdev.set_crypttab foo /dev/sdz1 mypassword swap,size=256\n    "
    if options is None:
        options = ''
    elif isinstance(options, str):
        pass
    elif isinstance(options, list):
        options = ','.join(options)
    else:
        msg = 'options must be a string or list of strings'
        raise CommandExecutionError(msg)
    entry_args = {'name': name, 'device': device, 'password': password if password is not None else 'none', 'options': options}
    lines = []
    ret = None
    if isinstance(match_on, list):
        pass
    elif not isinstance(match_on, str):
        msg = 'match_on must be a string or list of strings'
        raise CommandExecutionError(msg)
    else:
        match_on = [match_on]
    entry = _crypttab_entry(**entry_args)
    try:
        criteria = entry.pick(match_on)
    except KeyError:
        filterFn = lambda key: key not in _crypttab_entry.crypttab_keys
        invalid_keys = filter(filterFn, match_on)
        msg = 'Unrecognized keys in match_on: "{}"'.format(invalid_keys)
        raise CommandExecutionError(msg)
    if not os.path.isfile(config):
        raise CommandExecutionError('Bad config file "{}"'.format(config))
    try:
        with salt.utils.files.fopen(config, 'r') as ifile:
            for line in ifile:
                line = salt.utils.stringutils.to_unicode(line)
                try:
                    if criteria.match(line):
                        ret = 'present'
                        if entry.match(line):
                            lines.append(line)
                        else:
                            ret = 'change'
                            lines.append(str(entry))
                    else:
                        lines.append(line)
                except _crypttab_entry.ParseError:
                    lines.append(line)
    except OSError as exc:
        msg = "Couldn't read from {0}: {1}"
        raise CommandExecutionError(msg.format(config, exc))
    if ret is None:
        lines.append(str(entry))
        ret = 'new'
    if ret != 'present':
        if not test:
            try:
                with salt.utils.files.fopen(config, 'w+') as ofile:
                    ofile.writelines((salt.utils.stringutils.to_str(line) for line in lines))
            except OSError:
                msg = 'File not writable {0}'
                raise CommandExecutionError(msg.format(config))
    return ret

def open(name, device, keyfile):
    if False:
        while True:
            i = 10
    "\n    Open a crypt device using ``cryptsetup``. The ``keyfile`` must not be\n    ``None`` or ``'none'``, because ``cryptsetup`` will otherwise ask for the\n    password interactively.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' cryptdev.open foo /dev/sdz1 /path/to/keyfile\n    "
    if keyfile is None or keyfile == 'none' or keyfile == '-':
        raise CommandExecutionError('For immediate crypt device mapping, keyfile must not be none')
    code = __salt__['cmd.retcode']('cryptsetup open --key-file {} {} {}'.format(keyfile, device, name))
    return code == 0

def close(name):
    if False:
        return 10
    "\n    Close a crypt device using ``cryptsetup``.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' cryptdev.close foo\n    "
    code = __salt__['cmd.retcode']('cryptsetup close {}'.format(name))
    return code == 0