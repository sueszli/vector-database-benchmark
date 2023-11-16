"""AST node annotation support.

Adapted from Tangent.
"""
import enum
import gast

class NoValue(enum.Enum):
    """Base class for different types of AST annotations."""

    def of(self, node, default=None):
        if False:
            i = 10
            return i + 15
        return getanno(node, self, default=default)

    def add_to(self, node, value):
        if False:
            print('Hello World!')
        setanno(node, self, value)

    def exists(self, node):
        if False:
            print('Hello World!')
        return hasanno(node, self)

    def __repr__(self):
        if False:
            for i in range(10):
                print('nop')
        return str(self.name)

class Basic(NoValue):
    """Container for basic annotation keys.

  The enum values are used strictly for documentation purposes.
  """
    QN = 'Qualified name, as it appeared in the code. See qual_names.py.'
    SKIP_PROCESSING = 'This node should be preserved as is and not processed any further.'
    INDENT_BLOCK_REMAINDER = 'When a node is annotated with this, the remainder of the block should be indented below it. The annotation contains a tuple (new_body, name_map), where `new_body` is the new indented block and `name_map` allows renaming symbols.'
    ORIGIN = 'Information about the source code that converted code originated from. See origin_information.py.'
    DIRECTIVES = 'User directives associated with a statement or a variable. Typically, they affect the immediately-enclosing statement.'
    EXTRA_LOOP_TEST = 'A special annotation containing additional test code to be executed in for loops.'

class Static(NoValue):
    """Container for static analysis annotation keys.

  The enum values are used strictly for documentation purposes.
  """
    IS_PARAM = 'Symbol is a parameter to the function being analyzed.'
    SCOPE = 'The scope for the annotated node. See activity.py.'
    ARGS_SCOPE = 'The scope for the argument list of a function call.'
    COND_SCOPE = 'The scope for the test node of a conditional statement.'
    BODY_SCOPE = 'The scope for the main body of a statement (True branch for if statements, main body for loops).'
    ORELSE_SCOPE = 'The scope for the orelse body of a statement (False branch for if statements, orelse body for loops).'
    DEFINITIONS = 'Reaching definition information. See reaching_definitions.py.'
    ORIG_DEFINITIONS = 'The value of DEFINITIONS that applied to the original code before any conversion.'
    DEFINED_FNS_IN = 'Local function definitions that may exist when exiting the node. See reaching_fndefs.py'
    DEFINED_VARS_IN = 'Symbols defined when entering the node. See reaching_definitions.py.'
    LIVE_VARS_OUT = 'Symbols live when exiting the node. See liveness.py.'
    LIVE_VARS_IN = 'Symbols live when entering the node. See liveness.py.'
    TYPES = 'Static type information. See type_inference.py.'
    CLOSURE_TYPES = 'Types of closure symbols at each detected call site.'
    VALUE = 'Static value information. See type_inference.py.'
FAIL = object()

def keys(node, field_name='___pyct_anno'):
    if False:
        for i in range(10):
            print('nop')
    if not hasattr(node, field_name):
        return frozenset()
    return frozenset(getattr(node, field_name).keys())

def getanno(node, key, default=FAIL, field_name='___pyct_anno'):
    if False:
        while True:
            i = 10
    if default is FAIL or (hasattr(node, field_name) and key in getattr(node, field_name)):
        return getattr(node, field_name)[key]
    return default

def hasanno(node, key, field_name='___pyct_anno'):
    if False:
        while True:
            i = 10
    return hasattr(node, field_name) and key in getattr(node, field_name)

def setanno(node, key, value, field_name='___pyct_anno'):
    if False:
        return 10
    annotations = getattr(node, field_name, {})
    setattr(node, field_name, annotations)
    annotations[key] = value
    if field_name not in node._fields:
        node._fields += (field_name,)

def delanno(node, key, field_name='___pyct_anno'):
    if False:
        while True:
            i = 10
    annotations = getattr(node, field_name)
    del annotations[key]
    if not annotations:
        delattr(node, field_name)
        node._fields = tuple((f for f in node._fields if f != field_name))

def copyanno(from_node, to_node, key, field_name='___pyct_anno'):
    if False:
        print('Hello World!')
    if hasanno(from_node, key, field_name=field_name):
        setanno(to_node, key, getanno(from_node, key, field_name=field_name), field_name=field_name)

def dup(node, copy_map, field_name='___pyct_anno'):
    if False:
        print('Hello World!')
    'Recursively copies annotations in an AST tree.\n\n  Args:\n    node: ast.AST\n    copy_map: Dict[Hashable, Hashable], maps a source anno key to a destination\n        key. All annotations with the source key will be copied to identical\n        annotations with the destination key.\n    field_name: str\n  '
    for n in gast.walk(node):
        for k in copy_map:
            if hasanno(n, k, field_name):
                setanno(n, copy_map[k], getanno(n, k, field_name), field_name)