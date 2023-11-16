import fnmatch
import os.path
import re
import salt.utils.stringutils
from salt.exceptions import CommandExecutionError

class InputSanitizer:

    @staticmethod
    def trim(value):
        if False:
            return 10
        '\n        Raise an exception if value is empty. Otherwise strip it down.\n        :param value:\n        :return:\n        '
        value = (value or '').strip()
        if not value:
            raise CommandExecutionError('Empty value during sanitation')
        return str(value)

    @staticmethod
    def filename(value):
        if False:
            print('Hello World!')
        '\n        Remove everything that would affect paths in the filename\n\n        :param value:\n        :return:\n        '
        return re.sub('[^a-zA-Z0-9.-_ ]', '', os.path.basename(InputSanitizer.trim(value)))

    @staticmethod
    def hostname(value):
        if False:
            while True:
                i = 10
        '\n        Clean value for RFC1123.\n\n        :param value:\n        :return:\n        '
        return re.sub('[^a-zA-Z0-9.-]', '', InputSanitizer.trim(value)).strip('.')
    id = hostname
clean = InputSanitizer()

def mask_args_value(data, mask):
    if False:
        print('Hello World!')
    '\n    Mask a line in the data, which matches "mask".\n\n    This can be used for cases where values in your roster file may contain\n    sensitive data such as IP addresses, passwords, user names, etc.\n\n    Note that this works only when ``data`` is a single string (i.e. when the\n    data in the roster is formatted as ``key: value`` pairs in YAML syntax).\n\n    :param data: String data, already rendered.\n    :param mask: Mask that matches a single line\n\n    :return:\n    '
    if not mask:
        return data
    out = []
    for line in data.split(os.linesep):
        if fnmatch.fnmatch(line.strip(), mask) and ':' in line:
            (key, value) = line.split(':', 1)
            out.append('{}: {}'.format(salt.utils.stringutils.to_unicode(key.strip()), '** hidden **'))
        else:
            out.append(line)
    return '\n'.join(out)