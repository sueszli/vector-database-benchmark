"""
Script to generate README.md
"""
from pwn import *
out = '# Examples\nWhile these examples should all work, they are not very representative of\nthe pwntools project.\n\nWe have a plan to create a separate repository with examples, primarily\nexploits. Until we do so, we recommend new users to look at\nhttps://docs.pwntools.com, as this is a better overview of our features.\n\nIn no particular order the docstrings for each example:\n\n'

def append_example(_arg, top, names):
    if False:
        print('Hello World!')
    global out
    for name in names:
        if not (name.endswith('.py') and name != __file__):
            continue
        path = os.path.join(top, name)[2:]
        log.info('-> %s' % path)
        data = read(path).strip().decode()
        if data[0:3] not in ('"""', "'''"):
            log.warning('  Has no docstring!')
            continue
        try:
            i = data.index(data[0:3], 3)
        except ValueError:
            log.warning('  Docstring is weird')
            continue
        doc = util.safeeval.const(data[0:i + 3])
        out += '* `%s`\n' % path
        out += '```%s```\n' % doc
for (path, dirs, files) in os.walk('.', onerror=None):
    append_example(dirs, path, sorted(files))
write('README.md', out)