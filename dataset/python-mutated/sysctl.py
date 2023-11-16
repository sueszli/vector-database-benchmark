from __future__ import annotations
import re
from ansible.module_utils.common.text.converters import to_text

def get_sysctl(module, prefixes):
    if False:
        return 10
    sysctl_cmd = module.get_bin_path('sysctl')
    cmd = [sysctl_cmd]
    cmd.extend(prefixes)
    sysctl = dict()
    try:
        (rc, out, err) = module.run_command(cmd)
    except (IOError, OSError) as e:
        module.warn('Unable to read sysctl: %s' % to_text(e))
        rc = 1
    if rc == 0:
        key = ''
        value = ''
        for line in out.splitlines():
            if not line.strip():
                continue
            if line.startswith(' '):
                value += '\n' + line
                continue
            if key:
                sysctl[key] = value.strip()
            try:
                (key, value) = re.split('\\s?=\\s?|: ', line, maxsplit=1)
            except Exception as e:
                module.warn('Unable to split sysctl line (%s): %s' % (to_text(line), to_text(e)))
        if key:
            sysctl[key] = value.strip()
    return sysctl