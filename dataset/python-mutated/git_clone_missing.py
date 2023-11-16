"""
Rule: git_clone_missing

Correct missing `git clone` command when pasting a git URL

```sh
>>> https://github.com/nvbn/thefuck.git
git clone https://github.com/nvbn/thefuck.git
```

Author: Miguel Guthridge
"""
from six.moves.urllib import parse
from thefuck.utils import which

def match(command):
    if False:
        return 10
    if len(command.script_parts) != 1:
        return False
    if which(command.script_parts[0]) or not ('No such file or directory' in command.output or 'not found' in command.output or 'is not recognised as' in command.output):
        return False
    url = parse.urlparse(command.script, scheme='ssh')
    if not url.netloc and url.scheme != 'ssh':
        return False
    if url.scheme == 'ssh' and (not ('@' in command.script and ':' in command.script)):
        return False
    return url.scheme in ['http', 'https', 'ssh']

def get_new_command(command):
    if False:
        print('Hello World!')
    return 'git clone ' + command.script