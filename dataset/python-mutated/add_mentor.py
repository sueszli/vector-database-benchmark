import os
import re
import socket
import sys
from argparse import ArgumentParser
from typing import List
import requests
parser = ArgumentParser(description='Give a mentor ssh access to this machine.')
parser.add_argument('username', help='GitHub username of the mentor.')
parser.add_argument('--remove', help='Remove his/her key from the machine.', action='store_true')
append_key = '#<{username}>{{{{\n{key}\n#}}}}<{username}>\n'

def get_mentor_keys(username: str) -> List[str]:
    if False:
        print('Hello World!')
    url = f'https://api.github.com/users/{username}/keys'
    r = requests.get(url)
    if r.status_code != 200:
        print('Cannot connect to GitHub...')
        sys.exit(1)
    keys = r.json()
    if not keys:
        print(f'Mentor "{username}" has no public key.')
        sys.exit(1)
    return [key['key'] for key in keys]
if __name__ == '__main__':
    args = parser.parse_args()
    authorized_keys = os.path.expanduser('~/.ssh/authorized_keys')
    if args.remove:
        remove_re = re.compile(f'#<{args.username}>{{{{.+}}}}<{args.username}>(\\n)?', re.DOTALL | re.MULTILINE)
        with open(authorized_keys, 'r+') as f:
            old_content = f.read()
            new_content = re.sub(remove_re, '', old_content)
            f.seek(0)
            f.write(new_content)
            f.truncate()
        print(f"Successfully removed {args.username}' SSH key!")
    else:
        keys = get_mentor_keys(args.username)
        with open(authorized_keys, 'a') as f:
            for key in keys:
                f.write(append_key.format(username=args.username, key=key))
        print(f"Successfully added {args.username}'s SSH key!")
        print('Can you let your mentor know that they can connect to this machine with:\n')
        print(f'    $ ssh zulipdev@{socket.gethostname()}\n')