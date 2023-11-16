"""
Using lexer dynamic_complete
============================

Demonstrates how to use ``lexer='dynamic_complete'`` and ``ambiguity='explicit'``

Sometimes you have data that is highly ambiguous or 'broken' in some sense.
When using ``parser='earley'`` and ``lexer='dynamic_complete'``, Lark will be able
parse just about anything as long as there is a valid way to generate it from
the Grammar, including looking 'into' the Regexes.

This examples shows how to parse a json input where the quotes have been
replaced by underscores: ``{_foo_:{}, _bar_: [], _baz_: __}``
Notice that underscores might still appear inside strings, so a potentially
valid reading of the above is:
``{"foo_:{}, _bar": [], "baz": ""}``
"""
from pprint import pprint
from lark import Lark, Tree, Transformer, v_args
from lark.visitors import Transformer_InPlace
GRAMMAR = '\n%import common.SIGNED_NUMBER\n%import common.WS_INLINE\n%import common.NEWLINE\n%ignore WS_INLINE\n\n?start: value\n\n?value: object\n      | array\n      | string\n      | SIGNED_NUMBER      -> number\n      | "true"             -> true\n      | "false"            -> false\n      | "null"             -> null\n\narray  : "[" (value ("," value)*)? "]"\nobject : "{" (pair ("," pair)*)? "}"\npair   : string ":" value\n\nstring: STRING\nSTRING : ESCAPED_STRING\n\nESCAPED_STRING: QUOTE_CHAR _STRING_ESC_INNER QUOTE_CHAR\nQUOTE_CHAR: "_"\n\n_STRING_INNER: /.*/\n_STRING_ESC_INNER: _STRING_INNER /(?<!\\\\)(\\\\\\\\)*?/\n\n'

def score(tree: Tree):
    if False:
        while True:
            i = 10
    '\n    Scores an option by how many children (and grand-children, and\n    grand-grand-children, ...) it has.\n    This means that the option with fewer large terminals gets selected\n\n    Between\n        object\n          pair\n            string\t_foo_\n            object\n          pair\n            string\t_bar_: [], _baz_\n            string\t__\n\n    and\n\n        object\n          pair\n            string\t_foo_\n            object\n          pair\n            string\t_bar_\n            array\n          pair\n            string\t_baz_\n            string\t__\n\n    this will give the second a higher score. (9 vs 13)\n    '
    return sum((len(t.children) for t in tree.iter_subtrees()))

class RemoveAmbiguities(Transformer_InPlace):
    """
    Selects an option to resolve an ambiguity using the score function above.
    Scores each option and selects the one with the higher score, e.g. the one
    with more nodes.

    If there is a performance problem with the Tree having to many _ambig and
    being slow and to large, this can instead be written as a ForestVisitor.
    Look at the 'Custom SPPF Prioritizer' example.
    """

    def _ambig(self, options):
        if False:
            i = 10
            return i + 15
        return max(options, key=score)

class TreeToJson(Transformer):
    """
    This is the same Transformer as the json_parser example.
    """

    @v_args(inline=True)
    def string(self, s):
        if False:
            return 10
        return s[1:-1].replace('\\"', '"')
    array = list
    pair = tuple
    object = dict
    number = v_args(inline=True)(float)
    null = lambda self, _: None
    true = lambda self, _: True
    false = lambda self, _: False
parser = Lark(GRAMMAR, parser='earley', ambiguity='explicit', lexer='dynamic_complete')
EXAMPLES = ['{_array_:[1,2,3]}', '{_abc_: _array must be of the following format [_1_, _2_, _3_]_}', '{_foo_:{}, _bar_: [], _baz_: __}', '{_error_:_invalid_client_, _error_description_:_AADSTS7000215: Invalid client secret is provided.\\r\\nTrace ID: a0a0aaaa-a0a0-0a00-000a-00a00aaa0a00\\r\\nCorrelation ID: aa0aaa00-0aaa-0000-00a0-00000aaaa0aa\\r\\nTimestamp: 1997-10-10 00:00:00Z_, _error_codes_:[7000215], _timestamp_:_1997-10-10 00:00:00Z_, _trace_id_:_a0a0aaaa-a0a0-0a00-000a-00a00aaa0a00_, _correlation_id_:_aa0aaa00-0aaa-0000-00a0-00000aaaa0aa_, _error_uri_:_https://example.com_}']
for example in EXAMPLES:
    tree = parser.parse(example)
    tree = RemoveAmbiguities().transform(tree)
    result = TreeToJson().transform(tree)
    pprint(result)