from __future__ import annotations
import ast
from itertools import islice, chain
from types import GeneratorType
from ansible.module_utils.common.text.converters import to_text
from ansible.module_utils.six import string_types
from ansible.parsing.yaml.objects import AnsibleVaultEncryptedUnicode
from ansible.utils.native_jinja import NativeJinjaText
_JSON_MAP = {'true': True, 'false': False, 'null': None}

class Json2Python(ast.NodeTransformer):

    def visit_Name(self, node):
        if False:
            for i in range(10):
                print('nop')
        if node.id not in _JSON_MAP:
            return node
        return ast.Constant(value=_JSON_MAP[node.id])

def ansible_eval_concat(nodes):
    if False:
        return 10
    'Return a string of concatenated compiled nodes. Throw an undefined error\n    if any of the nodes is undefined.\n\n    If the result of concat appears to be a dictionary, list or bool,\n    try and convert it to such using literal_eval, the same mechanism as used\n    in jinja2_native.\n\n    Used in Templar.template() when jinja2_native=False and convert_data=True.\n    '
    head = list(islice(nodes, 2))
    if not head:
        return ''
    if len(head) == 1:
        out = head[0]
        if isinstance(out, NativeJinjaText):
            return out
        out = to_text(out)
    else:
        if isinstance(nodes, GeneratorType):
            nodes = chain(head, nodes)
        out = ''.join([to_text(v) for v in nodes])
    if out.startswith(('{', '[')) or out in ('True', 'False'):
        try:
            out = ast.literal_eval(ast.fix_missing_locations(Json2Python().visit(ast.parse(out, mode='eval'))))
        except (ValueError, SyntaxError, MemoryError):
            pass
    return out

def ansible_concat(nodes):
    if False:
        while True:
            i = 10
    "Return a string of concatenated compiled nodes. Throw an undefined error\n    if any of the nodes is undefined. Other than that it is equivalent to\n    Jinja2's default concat function.\n\n    Used in Templar.template() when jinja2_native=False and convert_data=False.\n    "
    return ''.join([to_text(v) for v in nodes])

def ansible_native_concat(nodes):
    if False:
        for i in range(10):
            print('nop')
    'Return a native Python type from the list of compiled nodes. If the\n    result is a single node, its value is returned. Otherwise, the nodes are\n    concatenated as strings. If the result can be parsed with\n    :func:`ast.literal_eval`, the parsed value is returned. Otherwise, the\n    string is returned.\n\n    https://github.com/pallets/jinja/blob/master/src/jinja2/nativetypes.py\n    '
    head = list(islice(nodes, 2))
    if not head:
        return None
    if len(head) == 1:
        out = head[0]
        if isinstance(out, AnsibleVaultEncryptedUnicode):
            return out.data
        if isinstance(out, NativeJinjaText):
            return out
        if not isinstance(out, string_types):
            return out
    else:
        if isinstance(nodes, GeneratorType):
            nodes = chain(head, nodes)
        out = ''.join([to_text(v) for v in nodes])
    try:
        evaled = ast.literal_eval(ast.parse(out, mode='eval'))
    except (ValueError, SyntaxError, MemoryError):
        return out
    if isinstance(evaled, string_types):
        quote = out[0]
        return f'{quote}{evaled}{quote}'
    return evaled