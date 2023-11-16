"""jc - JSON Convert `hash sum` command output parser

This parser works with the following hash calculation utilities:
- `md5`
- `md5sum`
- `shasum`
- `sha1sum`
- `sha224sum`
- `sha256sum`
- `sha384sum`
- `sha512sum`

Usage (cli):

    $ md5sum file.txt | jc --hashsum

or

    $ jc md5sum file.txt

Usage (module):

    import jc
    result = jc.parse('hashsum', md5sum_command_output)

Schema:

    [
      {
        "filename":     string,
        "hash":         string,
      }
    ]

Examples:

    $ md5sum * | jc --hashsum -p
    [
      {
        "filename": "devtoolset-3-gcc-4.9.2-6.el7.x86_64.rpm",
        "hash": "65fc958c1add637ec23c4b137aecf3d3"
      },
      {
        "filename": "digout",
        "hash": "5b9312ee5aff080927753c63a347707d"
      },
      {
        "filename": "dmidecode.out",
        "hash": "716fd11c2ac00db109281f7110b8fb9d"
      },
      {
        "filename": "file with spaces in the name",
        "hash": "d41d8cd98f00b204e9800998ecf8427e"
      },
      {
        "filename": "id-centos.out",
        "hash": "4295be239a14ad77ef3253103de976d2"
      },
      {
        "filename": "ifcfg.json",
        "hash": "01fda0d9ba9a75618b072e64ff512b43"
      },
      ...
    ]
"""
import jc.utils

class info:
    """Provides parser metadata (version, author, etc.)"""
    version = '1.2'
    description = 'hashsum command parser (`md5sum`, `shasum`, etc.)'
    author = 'Kelly Brazil'
    author_email = 'kellyjonbrazil@gmail.com'
    details = 'Parses MD5 and SHA hash program output'
    compatible = ['linux', 'darwin', 'cygwin', 'aix', 'freebsd']
    magic_commands = ['md5sum', 'md5', 'shasum', 'sha1sum', 'sha224sum', 'sha256sum', 'sha384sum', 'sha512sum']
    tags = ['command']
__version__ = info.version

def _process(proc_data):
    if False:
        print('Hello World!')
    '\n    Final processing to conform to the schema.\n\n    Parameters:\n\n        proc_data:   (List of Dictionaries) raw structured data to process\n\n    Returns:\n\n        List of Dictionaries. Structured data to conform to the schema.\n    '
    return proc_data

def parse(data, raw=False, quiet=False):
    if False:
        return 10
    '\n    Main text parsing function\n\n    Parameters:\n\n        data:        (string)  text data to parse\n        raw:         (boolean) unprocessed output if True\n        quiet:       (boolean) suppress warning messages if True\n\n    Returns:\n\n        List of Dictionaries. Raw or processed structured data.\n    '
    jc.utils.compatibility(__name__, info.compatible, quiet)
    jc.utils.input_type_check(data)
    raw_output = []
    if jc.utils.has_data(data):
        for line in filter(None, data.splitlines()):
            if line.startswith('MD5 ('):
                file_hash = line.split('=', maxsplit=1)[1].strip()
                file_name = line.split('=', maxsplit=1)[0].strip()
                file_name = file_name[5:]
                file_name = file_name[:-1]
            else:
                file_hash = line.split(maxsplit=1)[0]
                file_name = line.split(maxsplit=1)[1]
            item = {'filename': file_name, 'hash': file_hash}
            raw_output.append(item)
    if raw:
        return raw_output
    else:
        return _process(raw_output)