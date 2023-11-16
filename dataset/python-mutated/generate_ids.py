"""Common code for generating file or revision ids."""
from __future__ import absolute_import
from bzrlib.lazy_import import lazy_import
lazy_import(globals(), '\nimport time\n\nfrom bzrlib import (\n    config,\n    errors,\n    osutils,\n    )\n')
from bzrlib import lazy_regex
_file_id_chars_re = lazy_regex.lazy_compile('[^\\w.]')
_rev_id_chars_re = lazy_regex.lazy_compile('[^-\\w.+@]')
_gen_file_id_suffix = None
_gen_file_id_serial = 0

def _next_id_suffix():
    if False:
        for i in range(10):
            print('nop')
    'Create a new file id suffix that is reasonably unique.\n\n    On the first call we combine the current time with 64 bits of randomness to\n    give a highly probably globally unique number. Then each call in the same\n    process adds 1 to a serial number we append to that unique value.\n    '
    global _gen_file_id_suffix, _gen_file_id_serial
    if _gen_file_id_suffix is None:
        _gen_file_id_suffix = '-%s-%s-' % (osutils.compact_date(time.time()), osutils.rand_chars(16))
    _gen_file_id_serial += 1
    return _gen_file_id_suffix + str(_gen_file_id_serial)

def gen_file_id(name):
    if False:
        print('Hello World!')
    "Return new file id for the basename 'name'.\n\n    The uniqueness is supplied from _next_id_suffix.\n    "
    ascii_word_only = str(_file_id_chars_re.sub('', name.lower()))
    short_no_dots = ascii_word_only.lstrip('.')[:20]
    return short_no_dots + _next_id_suffix()

def gen_root_id():
    if False:
        i = 10
        return i + 15
    'Return a new tree-root file id.'
    return gen_file_id('tree_root')

def gen_revision_id(username, timestamp=None):
    if False:
        print('Hello World!')
    'Return new revision-id.\n\n    :param username: The username of the committer, in the format returned by\n        config.username().  This is typically a real name, followed by an\n        email address. If found, we will use just the email address portion.\n        Otherwise we flatten the real name, and use that.\n    :return: A new revision id.\n    '
    try:
        user_or_email = config.extract_email_address(username)
    except errors.NoEmailInUsername:
        user_or_email = username
    user_or_email = user_or_email.lower()
    user_or_email = user_or_email.replace(' ', '_')
    user_or_email = _rev_id_chars_re.sub('', user_or_email)
    unique_chunk = osutils.rand_chars(16)
    if timestamp is None:
        timestamp = time.time()
    rev_id = u'-'.join((user_or_email, osutils.compact_date(timestamp), unique_chunk))
    return rev_id.encode('utf8')