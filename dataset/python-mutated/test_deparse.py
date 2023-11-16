from uncompyle6.semantics.fragments import code_deparse as deparse
from xdis.version_info import PYTHON_VERSION_TRIPLE

def map_stmts(x, y):
    if False:
        print('Hello World!')
    x = []
    y = {}
    return (x, y)

def return_stmt(x, y):
    if False:
        print('Hello World!')
    return (x, y)

def try_stmt():
    if False:
        print('Hello World!')
    try:
        x = 1
    except:
        pass
    return x

def for_range_stmt():
    if False:
        return 10
    for i in range(2):
        i + 1

def list_comp():
    if False:
        i = 10
        return i + 15
    [y for y in range(3)]

def get_parsed_for_fn(fn):
    if False:
        return 10
    code = fn.__code__
    return deparse(code, version=PYTHON_VERSION_TRIPLE)

def check_expect(expect, parsed, fn_name):
    if False:
        for i in range(10):
            print('nop')
    debug = False
    i = 2
    max_expect = len(expect)
    for (name, offset) in sorted(parsed.offsets.keys()):
        assert i + 1 <= max_expect, '%s: ran out if items in testing node' % fn_name
        nodeInfo = parsed.offsets[name, offset]
        node = nodeInfo.node
        extractInfo = parsed.extract_node_info(node)
        assert expect[i] == extractInfo.selectedLine, '%s: line %s expect:\n%s\ngot:\n%s' % (fn_name, i, expect[i], extractInfo.selectedLine)
        assert expect[i + 1] == extractInfo.markerLine, 'line %s expect:\n%s\ngot:\n%s' % (i + 1, expect[i + 1], extractInfo.markerLine)
        i += 3
        if debug:
            print(node.offset)
            print(extractInfo.selectedLine)
            print(extractInfo.markerLine)
        (extractInfo, p) = parsed.extract_parent_info(node)
        if extractInfo:
            assert i + 1 < max_expect, 'ran out of items in testing parent'
            if debug:
                print('Contained in...')
                print(extractInfo.selectedLine)
                print(extractInfo.markerLine)
            assert expect[i] == extractInfo.selectedLine, 'parent line %s expect:\n%s\ngot:\n%s' % (i, expect[i], extractInfo.selectedLine)
            assert expect[i + 1] == extractInfo.markerLine, 'parent line %s expect:\n%s\ngot:\n%s' % (i + 1, expect[i + 1], extractInfo.markerLine)
            i += 3
        pass
    pass

def test_stuff():
    if False:
        while True:
            i = 10
    return
    parsed = get_parsed_for_fn(map_stmts)
    expect = '\n-1\nreturn (x, y)\n             ^\nContained in...\nreturn (x, y)\n-------------\n0\nx = []\n    -\nContained in...\nx = []\n    --\n3\nx = []\n-\nContained in...\nx = []\n------\n6\ny = {}\n    -\nContained in...\ny = {}\n    --\n9\ny = {}\n-\nContained in...\ny = {}\n------\n12\nreturn (x, y)\n        -\nContained in...\nreturn (x, y)\n       ------\n15\nreturn (x, y)\n           -\nContained in...\nreturn (x, y)\n       ------\n18\nreturn (x, y)\n       ------\nContained in...\nreturn (x, y)\n-------------\n21\nreturn (x, y)\n-------------\nContained in...\nx = [] ...\n------ ...\n'.split('\n')
    check_expect(expect, parsed, 'map_stmts')
    parsed = get_parsed_for_fn(return_stmt)
    expect = '\n-1\nreturn (x, y)\n             ^\nContained in...\nreturn (x, y)\n-------------\n0\nreturn (x, y)\n        -\nContained in...\nreturn (x, y)\n       ------\n3\nreturn (x, y)\n           -\nContained in...\nreturn (x, y)\n       ------\n6\nreturn (x, y)\n       ------\nContained in...\nreturn (x, y)\n-------------\n9\nreturn (x, y)\n-------------\nContained in...\nreturn (x, y)\n-------------\n'.split('\n')
    check_expect(expect, parsed, 'return_stmt')
    expect = '\n0\nfor i in range(2):\n    -\nContained in...\nfor i in range(2): ...\n------------------ ...\n3\nfor i in range(2):\n         -----\nContained in...\nfor i in range(2):\n         --------\n6\nfor i in range(2):\n               -\nContained in...\nfor i in range(2):\n         --------\n9\nfor i in range(2):\n         --------\nContained in...\nfor i in range(2): ...\n------------------ ...\n12\nfor i in range(2):\n    -\nContained in...\nfor i in range(2): ...\n------------------ ...\n13\nfor i in range(2):\n    -\nContained in...\nfor i in range(2): ...\n------------------ ...\n16\nfor i in range(2):\n    -\nContained in...\nfor i in range(2): ...\n------------------ ...\n19\n    i + 1\n    -\nContained in...\n    i + 1\n    -----\n22\n    i + 1\n        -\nContained in...\n    i + 1\n    -----\n25\n    i + 1\n      -\nContained in...\n    i + 1\n    -----\n27\nreturn\n      ^\nContained in...\n    i + 1\n    -----\n31\nreturn\n------\nContained in...\nfor i in range(2): ...\n------------------ ...\n34\nreturn\n------\nContained in...\nfor i in range(2): ...\n------------------ ...\n.\n'.split('\n')
    parsed = get_parsed_for_fn(for_range_stmt)