"""This module implements the :origin:`searxng_msg <babel.cfg>` extractor to
extract messages from:

- :origin:`searx/searxng.msg`

The ``searxng.msg`` files are selected by Babel_, see Babel's configuration in
:origin:`babel.cfg`::

    searxng_msg = searx.babel_extract.extract
    ...
    [searxng_msg: **/searxng.msg]

A ``searxng.msg`` file is a python file that is *executed* by the
:py:obj:`extract` function.  Additional ``searxng.msg`` files can be added by:

1. Adding a ``searxng.msg`` file in one of the SearXNG python packages and
2. implement a method in :py:obj:`extract` that yields messages from this file.

.. _Babel: https://babel.pocoo.org/en/latest/index.html

"""
from os import path
SEARXNG_MSG_FILE = 'searxng.msg'
_MSG_FILES = [path.join(path.dirname(__file__), SEARXNG_MSG_FILE)]

def extract(fileobj, keywords, comment_tags, options):
    if False:
        i = 10
        return i + 15
    'Extract messages from ``searxng.msg`` files by a custom extractor_.\n\n    .. _extractor:\n       https://babel.pocoo.org/en/latest/messages.html#writing-extraction-methods\n    '
    if fileobj.name not in _MSG_FILES:
        raise RuntimeError("don't know how to extract messages from %s" % fileobj.name)
    namespace = {}
    exec(fileobj.read(), {}, namespace)
    for name in namespace['__all__']:
        for (k, v) in namespace[name].items():
            yield (0, '_', v, ["%s['%s']" % (name, k)])