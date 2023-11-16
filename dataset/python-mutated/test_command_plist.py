from __future__ import annotations
import re
import gdb
import tests
LINKED_LISTS_BINARY = tests.binaries.get('linked-lists.out')

def startup(start_binary):
    if False:
        return 10
    start_binary(LINKED_LISTS_BINARY)
    gdb.execute('break break_here')
    gdb.execute('run')
    gdb.execute('up')

def test_command_plist_flat_no_flags(start_binary):
    if False:
        print('Hello World!')
    '\n    Tests the plist for a non-nested linked list\n    '
    startup(start_binary)
    expected_out = re.compile('0[xX][0-9a-fA-F]+ <node_a>: {\\s*\n  value = 0,\\s*\n  next = 0[xX][0-9a-fA-F]+ <node_b>\\s*\n}\\s*\n0[xX][0-9a-fA-F]+ <node_b>: {\\s*\n  value = 1,\\s*\n  next = 0[xX][0-9a-fA-F]+ <node_c>\\s*\n}\\s*\n0[xX][0-9a-fA-F]+ <node_c>: {\\s*\n  value = 2,\\s*\n  next = 0x0\\s*\n}')
    result_str = gdb.execute('plist node_a next', to_string=True)
    assert expected_out.match(result_str) is not None

def test_command_plist_flat_field(start_binary):
    if False:
        return 10
    '\n    Tests the plist command for a non-nested linked list with field flag\n    '
    startup(start_binary)
    expected_out = re.compile('0[xX][0-9a-fA-F]+ <node_a>: 0\\s*\n0[xX][0-9a-fA-F]+ <node_b>: 1\\s*\n0[xX][0-9a-fA-F]+ <node_c>: 2\\s*\n')
    result_str = gdb.execute('plist node_a next -f value', to_string=True)
    assert expected_out.match(result_str) is not None

def test_command_plist_flat_sentinel(start_binary):
    if False:
        return 10
    '\n    Tests the plist command for a non-nested linked list with field flag\n    '
    startup(start_binary)
    sentinel = int(gdb.lookup_symbol('node_c')[0].value().address)
    expected_out = re.compile('0[xX][0-9a-fA-F]+ <node_a>: {\\s*\n  value = 0,\\s*\n  next = 0[xX][0-9a-fA-F]+ <node_b>\\s*\n}\\s*\n0[xX][0-9a-fA-F]+ <node_b>: {\\s*\n  value = 1,\\s*\n  next = 0[xX][0-9a-fA-F]+ <node_c>\\s*\n}')
    result_str = gdb.execute(f'plist node_a next -s {sentinel}', to_string=True)
    assert expected_out.match(result_str) is not None

def test_command_plist_nested_direct(start_binary):
    if False:
        return 10
    '\n    Tests the plist for a nested linked list pointing to the outer structure\n    '
    startup(start_binary)
    expected_out = re.compile('0[xX][0-9a-fA-F]+ <inner_b_node_a>: {\\s*\n  value = 0,\\s*\n  inner = {\\s*\n    next = 0[xX][0-9a-fA-F]+ <inner_b_node_b>\\s*\n  }\\s*\n}\\s*\n0[xX][0-9a-fA-F]+ <inner_b_node_b>: {\\s*\n  value = 1,\\s*\n  inner = {\\s*\n    next = 0[xX][0-9a-fA-F]+ <inner_b_node_c>\\s*\n  }\\s*\n}\\s*\n0[xX][0-9a-fA-F]+ <inner_b_node_c>: {\\s*\n  value = 2,\\s*\n  inner = {\\s*\n    next = 0x0\\s*\n  }\\s*\n}')
    result_str = gdb.execute('plist inner_b_node_a -i inner next', to_string=True)
    assert expected_out.match(result_str) is not None

def test_command_plist_nested_indirect(start_binary):
    if False:
        while True:
            i = 10
    '\n    Tests the plist for a nested linked list pointing to the inner structure\n    '
    startup(start_binary)
    expected_out = re.compile('0[xX][0-9a-fA-F]+ <inner_a_node_a>: {\\s*\n  value = 0,\\s*\n  inner = {\\s*\n    next = 0[xX][0-9a-fA-F]+ <inner_a_node_b\\+8>\\s*\n  }\\s*\n}\\s*\n0[xX][0-9a-fA-F]+ <inner_a_node_b>: {\\s*\n  value = 1,\\s*\n  inner = {\\s*\n    next = 0[xX][0-9a-fA-F]+ <inner_a_node_c\\+8>\\s*\n  }\\s*\n}\\s*\n0[xX][0-9a-fA-F]+ <inner_a_node_c>: {\\s*\n  value = 2,\\s*\n  inner = {\\s*\n    next = 0x0\\s*\n  }\\s*\n}')
    result_str = gdb.execute('plist inner_a_node_a -i inner next', to_string=True)
    assert expected_out.match(result_str) is not None