def test_bytes_logging_extended(get_contract_with_gas_estimation, get_logs):
    if False:
        return 10
    code = "\nevent MyLog:\n    arg1: int128\n    arg2: Bytes[64]\n    arg3: int128\n\n@external\ndef foo():\n    log MyLog(667788, b'hellohellohellohellohellohellohellohellohello', 334455)\n    "
    c = get_contract_with_gas_estimation(code)
    log = get_logs(c.foo(transact={}), c, 'MyLog')
    assert log[0].args.arg1 == 667788
    assert log[0].args.arg2 == b'hello' * 9
    assert log[0].args.arg3 == 334455

def test_bytes_logging_extended_variables(get_contract_with_gas_estimation, get_logs):
    if False:
        for i in range(10):
            print('nop')
    code = "\nevent MyLog:\n    arg1: Bytes[64]\n    arg2: Bytes[64]\n    arg3: Bytes[64]\n\n@external\ndef foo():\n    a: Bytes[64] = b'hellohellohellohellohellohellohellohellohello'\n    b: Bytes[64] = b'hellohellohellohellohellohellohellohello'\n    # test literal much smaller than buffer\n    log MyLog(a, b, b'hello')\n    "
    c = get_contract_with_gas_estimation(code)
    log = get_logs(c.foo(transact={}), c, 'MyLog')
    assert log[0].args.arg1 == b'hello' * 9
    assert log[0].args.arg2 == b'hello' * 8
    assert log[0].args.arg3 == b'hello' * 1

def test_bytes_logging_extended_passthrough(get_contract_with_gas_estimation, get_logs):
    if False:
        for i in range(10):
            print('nop')
    code = '\nevent MyLog:\n    arg1: int128\n    arg2: Bytes[64]\n    arg3: int128\n\n@external\ndef foo(a: int128, b: Bytes[64], c: int128):\n    log MyLog(a, b, c)\n    '
    c = get_contract_with_gas_estimation(code)
    log = get_logs(c.foo(333, b'flower' * 8, 444, transact={}), c, 'MyLog')
    assert log[0].args.arg1 == 333
    assert log[0].args.arg2 == b'flower' * 8
    assert log[0].args.arg3 == 444

def test_bytes_logging_extended_storage(get_contract_with_gas_estimation, get_logs):
    if False:
        for i in range(10):
            print('nop')
    code = '\nevent MyLog:\n    arg1: int128\n    arg2: Bytes[64]\n    arg3: int128\n\na: int128\nb: Bytes[64]\nc: int128\n\n@external\ndef foo():\n    log MyLog(self.a, self.b, self.c)\n\n@external\ndef set(x: int128, y: Bytes[64], z: int128):\n    self.a = x\n    self.b = y\n    self.c = z\n    '
    c = get_contract_with_gas_estimation(code)
    c.foo()
    log = get_logs(c.foo(transact={}), c, 'MyLog')
    assert log[0].args.arg1 == 0
    assert log[0].args.arg2 == b''
    assert log[0].args.arg3 == 0
    c.set(333, b'flower' * 8, 444, transact={})
    log = get_logs(c.foo(transact={}), c, 'MyLog')[0]
    assert log.args.arg1 == 333
    assert log.args.arg2 == b'flower' * 8
    assert log.args.arg3 == 444

def test_bytes_logging_extended_mixed_with_lists(get_contract_with_gas_estimation, get_logs):
    if False:
        print('Hello World!')
    code = "\nevent MyLog:\n    arg1: int128[2][2]\n    arg2: Bytes[64]\n    arg3: int128\n    arg4: Bytes[64]\n\n@external\ndef foo():\n    log MyLog(\n        [[24, 26], [12, 10]],\n        b'hellohellohellohellohellohellohellohellohello',\n        314159,\n        b'helphelphelphelphelphelphelphelphelphelphelp'\n    )\n    "
    c = get_contract_with_gas_estimation(code)
    log = get_logs(c.foo(transact={}), c, 'MyLog')[0]
    assert log.args.arg1 == [[24, 26], [12, 10]]
    assert log.args.arg2 == b'hello' * 9
    assert log.args.arg3 == 314159
    assert log.args.arg4 == b'help' * 11