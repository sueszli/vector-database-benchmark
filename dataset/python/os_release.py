"""jc - JSON Convert `/etc/os-release` file parser

This parser is an alias to the Key/Value parser (`--kv`).

Usage (cli):

    $ cat /etc/os-release | jc --os-release

Usage (module):

    import jc
    result = jc.parse('os_release', os_release_output)

Schema:

    {
        "<key>":     string
    }

Examples:

    $ cat /etc/os-release | jc --os-release -p
    {
      "NAME": "CentOS Linux",
      "VERSION": "7 (Core)",
      "ID": "centos",
      "ID_LIKE": "rhel fedora",
      "VERSION_ID": "7",
      "PRETTY_NAME": "CentOS Linux 7 (Core)",
      "ANSI_COLOR": "0;31",
      "CPE_NAME": "cpe:/o:centos:centos:7",
      "HOME_URL": "https://www.centos.org/",
      "BUG_REPORT_URL": "https://bugs.centos.org/",
      "CENTOS_MANTISBT_PROJECT": "CentOS-7",
      "CENTOS_MANTISBT_PROJECT_VERSION": "7",
      "REDHAT_SUPPORT_PRODUCT": "centos",
      "REDHAT_SUPPORT_PRODUCT_VERSION": "7"
    }

    $ cat /etc/os-release | jc --os-release -p -r
    {
      "NAME": "\\"CentOS Linux\\"",
      "VERSION": "\\"7 (Core)\\"",
      "ID": "\\"centos\\"",
      "ID_LIKE": "\\"rhel fedora\\"",
      "VERSION_ID": "\\"7\\"",
      "PRETTY_NAME": "\\"CentOS Linux 7 (Core)\\"",
      "ANSI_COLOR": "\\"0;31\\"",
      "CPE_NAME": "\\"cpe:/o:centos:centos:7\\"",
      "HOME_URL": "\\"https://www.centos.org/\\"",
      "BUG_REPORT_URL": "\\"https://bugs.centos.org/\\"",
      "CENTOS_MANTISBT_PROJECT": "\\"CentOS-7\\"",
      "CENTOS_MANTISBT_PROJECT_VERSION": "\\"7\\"",
      "REDHAT_SUPPORT_PRODUCT": "\\"centos\\"",
      "REDHAT_SUPPORT_PRODUCT_VERSION": "\\"7\\""
    }
"""
from jc.jc_types import JSONDictType
import jc.parsers.kv
import jc.utils


class info():
    """Provides parser metadata (version, author, etc.)"""
    version = '1.0'
    description = '`/etc/os-release` file parser'
    author = 'Kelly Brazil'
    author_email = 'kellyjonbrazil@gmail.com'
    details = 'Using the Key/Value parser'
    compatible = ['linux', 'darwin', 'cygwin', 'win32', 'aix', 'freebsd']
    tags = ['file', 'standard', 'string']


__version__ = info.version


def _process(proc_data: JSONDictType) -> JSONDictType:
    """
    Final processing to conform to the schema.

    Parameters:

        proc_data:   (Dictionary) raw structured data to process

    Returns:

        Dictionary. Structured to conform to the schema.
    """
    return jc.parsers.kv._process(proc_data)


def parse(
    data: str,
    raw: bool = False,
    quiet: bool = False
) -> JSONDictType:
    """
    Main text parsing function

    Parameters:

        data:        (string)  text data to parse
        raw:         (boolean) unprocessed output if True
        quiet:       (boolean) suppress warning messages if True

    Returns:

        Dictionary. Raw or processed structured data.
    """
    jc.utils.compatibility(__name__, info.compatible, quiet)
    raw_output = jc.parsers.kv.parse(data, raw, quiet)

    return raw_output if raw else _process(raw_output)
