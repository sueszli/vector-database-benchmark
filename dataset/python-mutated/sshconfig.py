"""
Parses roster entries out of Host directives from SSH config

.. code-block:: bash

    salt-ssh --roster sshconfig '*' -r "echo hi"
"""
import collections
import fnmatch
import logging
import os
import re
import salt.utils.files
import salt.utils.stringutils
log = logging.getLogger(__name__)
_SSHConfRegex = collections.namedtuple('_SSHConfRegex', ['target_field', 'pattern'])
_ROSTER_FIELDS = (_SSHConfRegex(target_field='user', pattern='\\s+User (.*)'), _SSHConfRegex(target_field='port', pattern='\\s+Port (.*)'), _SSHConfRegex(target_field='priv', pattern='\\s+IdentityFile (.*)'))

def _get_ssh_config_file(opts):
    if False:
        while True:
            i = 10
    '\n    :return: Path to the .ssh/config file - usually <home>/.ssh/config\n    '
    ssh_config_file = opts.get('ssh_config_file')
    if not os.path.isfile(ssh_config_file):
        raise OSError('Cannot find SSH config file')
    if not os.access(ssh_config_file, os.R_OK):
        raise OSError('Cannot access SSH config file: {}'.format(ssh_config_file))
    return ssh_config_file

def parse_ssh_config(lines):
    if False:
        print('Hello World!')
    '\n    Parses lines from the SSH config to create roster targets.\n\n    :param lines: Individual lines from the ssh config file\n    :return: Dictionary of targets in similar style to the flat roster\n    '
    hosts = []
    for line in lines:
        line = salt.utils.stringutils.to_unicode(line)
        if not line or line.startswith('#'):
            continue
        elif line.startswith('Host '):
            hosts.append([])
        hosts[-1].append(line)
    targets = collections.OrderedDict()
    for host_data in hosts:
        target = collections.OrderedDict()
        hostnames = host_data[0].split()[1:]
        for line in host_data[1:]:
            for field in _ROSTER_FIELDS:
                match = re.match(field.pattern, line)
                if match:
                    target[field.target_field] = match.group(1).strip()
        for hostname in hostnames:
            targets[hostname] = target
    wildcard_targets = []
    non_wildcard_targets = []
    for target in targets.keys():
        if '*' in target or '?' in target:
            wildcard_targets.append(target)
        else:
            non_wildcard_targets.append(target)
    for pattern in wildcard_targets:
        for candidate in non_wildcard_targets:
            if fnmatch.fnmatch(candidate, pattern):
                targets[candidate].update(targets[pattern])
        del targets[pattern]
    for target in targets:
        targets[target]['host'] = target
    return targets

def targets(tgt, tgt_type='glob', **kwargs):
    if False:
        while True:
            i = 10
    '\n    Return the targets from the flat yaml file, checks opts for location but\n    defaults to /etc/salt/roster\n    '
    ssh_config_file = _get_ssh_config_file(__opts__)
    with salt.utils.files.fopen(ssh_config_file, 'r') as fp:
        all_minions = parse_ssh_config([line.rstrip() for line in fp])
    rmatcher = RosterMatcher(all_minions, tgt, tgt_type)
    matched = rmatcher.targets()
    return matched

class RosterMatcher:
    """
    Matcher for the roster data structure
    """

    def __init__(self, raw, tgt, tgt_type):
        if False:
            return 10
        self.tgt = tgt
        self.tgt_type = tgt_type
        self.raw = raw

    def targets(self):
        if False:
            print('Hello World!')
        '\n        Execute the correct tgt_type routine and return\n        '
        try:
            return getattr(self, 'ret_{}_minions'.format(self.tgt_type))()
        except AttributeError:
            return {}

    def ret_glob_minions(self):
        if False:
            return 10
        '\n        Return minions that match via glob\n        '
        minions = {}
        for minion in self.raw:
            if fnmatch.fnmatch(minion, self.tgt):
                data = self.get_data(minion)
                if data:
                    minions[minion] = data
        return minions

    def get_data(self, minion):
        if False:
            for i in range(10):
                print('nop')
        '\n        Return the configured ip\n        '
        if isinstance(self.raw[minion], str):
            return {'host': self.raw[minion]}
        if isinstance(self.raw[minion], dict):
            return self.raw[minion]
        return False