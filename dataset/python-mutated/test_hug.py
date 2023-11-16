from functools import partial
from ..utils import isort_test
hug_isort_test = partial(isort_test, profile='hug', known_first_party=['hug'])

def test_hug_code_snippet_one():
    if False:
        for i in range(10):
            print('nop')
    hug_isort_test('\nfrom __future__ import absolute_import\n\nimport asyncio\nimport sys\nfrom collections import OrderedDict, namedtuple\nfrom distutils.util import strtobool\nfrom functools import partial\nfrom itertools import chain\nfrom types import ModuleType\nfrom wsgiref.simple_server import make_server\n\nimport falcon\nfrom falcon import HTTP_METHODS\n\nimport hug.defaults\nimport hug.output_format\nfrom hug import introspect\nfrom hug._version import current\n\nINTRO = """\n/#######################################################################\\\n          `.----``..-------..``.----.\n         :/:::::--:---------:--::::://.\n        .+::::----##/-/oo+:-##----:::://\n        `//::-------/oosoo-------::://.       ##    ##  ##    ##    #####\n          .-:------./++o/o-.------::-`   ```  ##    ##  ##    ##  ##\n             `----.-./+o+:..----.     `.:///. ########  ##    ## ##\n   ```        `----.-::::::------  `.-:::://. ##    ##  ##    ## ##   ####\n  ://::--.``` -:``...-----...` `:--::::::-.`  ##    ##  ##   ##   ##    ##\n  :/:::::::::-:-     `````      .:::::-.`     ##    ##    ####     ######\n   ``.--:::::::.                .:::.`\n         ``..::.                .::         EMBRACE THE APIs OF THE FUTURE\n             ::-                .:-\n             -::`               ::-                   VERSION {0}\n             `::-              -::`\n              -::-`           -::-\n\\########################################################################/\n Copyright (C) 2016 Timothy Edmund Crosley\n Under the MIT License\n""".format(\n    current\n)')

def test_hug_code_snippet_two():
    if False:
        for i in range(10):
            print('nop')
    hug_isort_test('from __future__ import absolute_import\n\nimport functools\nfrom collections import namedtuple\n\nfrom falcon import HTTP_METHODS\n\nimport hug.api\nimport hug.defaults\nimport hug.output_format\nfrom hug import introspect\nfrom hug.format import underscore\n\n\ndef default_output_format(\n    content_type="application/json", apply_globally=False, api=None, cli=False, http=True\n):\n')

def test_hug_code_snippet_three():
    if False:
        return 10
    hug_isort_test('from __future__ import absolute_import\n\nimport argparse\nimport asyncio\nimport os\nimport sys\nfrom collections import OrderedDict\nfrom functools import lru_cache, partial, wraps\n\nimport falcon\nfrom falcon import HTTP_BAD_REQUEST\n\nimport hug._empty as empty\nimport hug.api\nimport hug.output_format\nimport hug.types as types\nfrom hug import introspect\nfrom hug.exceptions import InvalidTypeData\nfrom hug.format import parse_content_type\nfrom hug.types import (\n    MarshmallowInputSchema,\n    MarshmallowReturnSchema,\n    Multiple,\n    OneOf,\n    SmartBoolean,\n    Text,\n    text,\n)\n\nDOC_TYPE_MAP = {str: "String", bool: "Boolean", list: "Multiple", int: "Integer", float: "Float"}\n')