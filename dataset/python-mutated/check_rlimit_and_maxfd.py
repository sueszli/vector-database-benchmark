from __future__ import annotations
import resource
import subprocess
from ansible.module_utils.basic import AnsibleModule

def main():
    if False:
        for i in range(10):
            print('nop')
    module = AnsibleModule(argument_spec=dict())
    rlimit_nofile = resource.getrlimit(resource.RLIMIT_NOFILE)
    try:
        maxfd = subprocess.MAXFD
    except AttributeError:
        maxfd = -1
    module.exit_json(rlimit_nofile=rlimit_nofile, maxfd=maxfd, infinity=resource.RLIM_INFINITY)
if __name__ == '__main__':
    main()