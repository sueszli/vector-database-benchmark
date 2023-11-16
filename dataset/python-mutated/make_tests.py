from __future__ import print_function
import sys
import random
tests = []
with open(sys.argv[1], 'r') as f:
    tests_str = f.read().split('\n')
for t_str in tests_str:
    try:
        tests.append(eval(t_str))
    except Exception:
        pass
random.shuffle(tests)
op_count = {}
test_dic = {}
for test in tests:
    try:
        cnt = op_count.get(test['mnemonic'], 0)
        if cnt > 1000:
            continue
        if test['mnemonic'] != 'PCMPISTRI' and test['mnemonic'] != 'PCMPISTRM' and (test['mnemonic'] != 'PCMPESTRI') and (test['mnemonic'] != 'PCMPESTRM'):
            continue
        op_count[test['mnemonic']] = cnt + 1
        test_dic[f"{test['mnemonic']}_{op_count[test['mnemonic']]}"] = test
    except Exception as e:
        print('EX', e, test)
        pass
print("\nimport unittest\nimport functools\nfrom manticore.core.cpu.x86 import *\nfrom manticore.core.smtlib import Operators\nfrom manticore.native.memory import *\n\n\ndef skipIfNotImplemented(f):\n    # XXX(yan) the inner function name must start with test_\n    @functools.wraps(f)\n    def test_inner(*args, **kwargs):\n        try:\n            return f(*args, **kwargs)\n        except NotImplementedError as e:\n            raise unittest.SkipTest(e.message)\n\n    return test_inner\n\ndef forAllTests(decorator):\n    def decorate(cls):\n        for attr in cls.__dict__:\n            if not attr.startswith('test_'):\n                continue\n            method = getattr(cls, attr)\n            if callable(method):\n                setattr(cls, attr, decorator(method))\n        return cls\n\n    return decorate\n\n@forAllTests(skipIfNotImplemented)\nclass CPUTest(unittest.TestCase):\n    _multiprocess_can_split_ = True\n\n    class ROOperand:\n        ''' Mocking class for operand ronly '''\n        def __init__(self, size, value):\n            self.size = size\n            self.value = value\n        def read(self):\n            return self.value & ((1<<self.size)-1)\n\n    class RWOperand(ROOperand):\n        ''' Mocking class for operand rw '''\n        def write(self, value):\n            self.value = value & ((1<<self.size)-1)\n            return self.value\n")

def isFlag(x):
    if False:
        return 10
    return x in ['OF', 'SF', 'ZF', 'AF', 'PF', 'CF', 'DF']

def regSize(x):
    if False:
        i = 10
        return i + 15
    if x in ('BPL', 'AH', 'CH', 'DH', 'BH', 'AL', 'CL', 'DL', 'BL', 'SIL', 'DIL', 'SIH', 'DIH', 'R8B', 'R9B', 'R10B', 'R11B', 'R12B', 'R13B', 'R14B', 'R15B'):
        return 8
    if x in ('GS', 'AX', 'CX', 'DX', 'BX', 'SI', 'DI', 'R8W', 'R9W', 'R10W', 'R11W', 'R12W', 'R13W', 'R14W', 'R15W'):
        return 16
    if x in ('EAX', 'ECX', 'EDX', 'EBX', 'ESP', 'EBP', 'ESI', 'EDI', 'EIP', 'EFLAGS', 'R8D', 'R9D', 'R10D', 'R11D', 'R12D', 'R13D', 'R14D', 'R15D'):
        return 32
    if x in ('RAX', 'RCX', 'RDX', 'RBX', 'RSP', 'RBP', 'RSI', 'RDI', 'R8', 'R9', 'R10', 'R11', 'R12', 'R13', 'R14', 'R15', 'RIP'):
        return 64
    if x in ('XMM0', 'XMM1', 'XMM2', 'XMM3', 'XMM4', 'XMM5', 'XMM6', 'XMM7', 'XMM8', 'XMM9', 'XMM10', 'XMM11', 'XMM12', 'XMM13', 'XMM14', 'XMM15'):
        return 128
    if x in ('YMM0', 'YMM1', 'YMM2', 'YMM3', 'YMM4', 'YMM5', 'YMM6', 'YMM7', 'YMM8', 'YMM9', 'YMM10', 'YMM11', 'YMM12', 'YMM13', 'YMM14', 'YMM15'):
        return 256
    if x in ('FP0', 'FP1', 'FP2', 'FP3', 'FP4', 'FP5', 'FP6', 'FP7', 'FPSW', 'FPTAG', 'FPCW'):
        raise Exception('FPU not supported')
    raise Exception('%s not supported', x)

def get_maps(test):
    if False:
        i = 10
        return i + 15
    pages = set()
    for addr in test['pre']['memory'].keys():
        pages.add(addr >> 12)
    for addr in test['pos']['memory'].keys():
        pages.add(addr >> 12)
    maps = []
    for p in sorted(pages):
        if len(maps) > 0 and maps[-1][0] + maps[-1][1] == p << 12:
            maps[-1] = (maps[-1][0], maps[-1][1] + 4096)
        else:
            maps.append((p << 12, 4096))
    return maps
for test_name in sorted(test_dic.keys()):
    test = test_dic[test_name]
    bits = {'i386': 32, 'amd64': 64}[test['arch']]
    pc = {'i386': 'EIP', 'amd64': 'RIP'}[test['arch']]
    cpu = {64: 'AMD64Cpu', 32: 'I386Cpu'}[bits]
    print(f"\n    def test_{test_name}(self):\n        ''' Instruction {test_name}\n            Groups: {', '.join(map(str, test['groups']))}\n            {test['disassembly']}\n        '''\n        mem = Memory{bits}()\n        cpu = {cpu}(mem)")
    for (addr, size) in get_maps(test):
        print("        mem.mmap(0x%08x, 0x%x, 'rwx')" % (addr, size))
    memoryContents = ''
    currentAddr = 0
    beginAddr = 0
    for addr in sorted(test['pre']['memory'].iterkeys()):
        if len(memoryContents) == 0:
            memoryContents = '%s' % test['pre']['memory'][addr]
            beginAddr = addr
        elif addr == currentAddr + 1:
            memoryContents = memoryContents + '%s' % test['pre']['memory'][addr]
        else:
            print('        mem.write(0x%08x, %r)' % (beginAddr, memoryContents))
            memoryContents = '%s' % test['pre']['memory'][addr]
            beginAddr = addr
        currentAddr = addr
    if len(memoryContents) > 0:
        print('        mem.write(0x%08x, %r)' % (beginAddr, memoryContents))
    for (reg_name, value) in test['pre']['registers'].items():
        if isFlag(reg_name):
            print(f'        cpu.{reg_name} = {value!r}')
        else:
            print(f'        cpu.{reg_name} = 0x{value:x}')
    print('        cpu.execute()\n    ')
    memoryContents = ''
    currentAddr = 0
    beginAddr = 0
    for addr in sorted(test['pre']['memory'].iterkeys()):
        if len(memoryContents) == 0:
            memoryContents = '%s' % test['pre']['memory'][addr]
            beginAddr = addr
        elif addr == currentAddr + 1:
            memoryContents = memoryContents + '%s' % test['pre']['memory'][addr]
        else:
            print('        self.assertEqual(mem[0x%08x:0x%08x], %r)' % (beginAddr, beginAddr + len(memoryContents), memoryContents))
            memoryContents = '%s' % test['pre']['memory'][addr]
            beginAddr = addr
        currentAddr = addr
    if len(memoryContents) > 0:
        print('        self.assertEqual(mem[0x%08x:0x%08x], %r)' % (beginAddr, beginAddr + len(memoryContents), memoryContents))
    for (reg_name, value) in test['pos']['registers'].items():
        print('        self.assertEqual(cpu.%s, %r)' % (reg_name, value))
for test_name in sorted(test_dic.keys()):
    test = test_dic[test_name]
    bits = {'i386': 32, 'amd64': 64}[test['arch']]
    pc = {'i386': 'EIP', 'amd64': 'RIP'}[test['arch']]
    print("\n    def test_%(test_name)s_symbolic(self):\n        ''' Instruction %(test_name)s\n            Groups: %(groups)s\n            %(disassembly)s\n        '''\n        cs = ConstraintSet()\n        mem = SMemory%(bits)d(cs)\n        cpu = %(cpu)s(mem)" % {'test_name': test_name, 'groups': ', '.join(map(str, test['groups'])), 'disassembly': test['disassembly'], 'bits': bits, 'arch': test['arch'], 'cpu': {64: 'AMD64Cpu', 32: 'I386Cpu'}[bits]})
    for (addr, size) in get_maps(test):
        print("        mem.mmap(0x%08x, 0x%x, 'rwx')" % (addr, size))
    ip = test['pre']['registers'][pc]
    text_size = len(test['text'])
    for (addr, byte) in test['pre']['memory'].items():
        if addr >= ip and addr <= ip + text_size:
            print('        mem[0x%x] = %r' % (addr, byte))
            continue
        print('        addr = cs.new_bitvec(%d)' % bits)
        print('        cs.add(addr == 0x%x)' % addr)
        print('        value = cs.new_bitvec(8)')
        print('        cs.add(value == 0x%x)' % ord(byte))
        print('        mem[addr] = value')
    for (reg_name, value) in test['pre']['registers'].items():
        if reg_name in ('EIP', 'RIP'):
            print('        cpu.%s = 0x%x' % (reg_name, value))
            continue
        if isFlag(reg_name):
            print('        cpu.%s = cs.new_bool()' % reg_name)
            print('        cs.add(cpu.%s == %r)' % (reg_name, value))
        else:
            print('        cpu.%s = cs.new_bitvec(%d)' % (reg_name, regSize(reg_name)))
            print('        cs.add(cpu.%s == 0x%x)' % (reg_name, value))
    pc_reg = 'RIP' if 'RIP' in test['pre']['registers'] else 'EIP'
    print(f"\n        done = False\n        while not done:\n            try:\n                cpu.execute()\n                done = True\n            except ConcretizeRegister as e:\n                symbol = getattr(cpu, e.reg_name)\n                values = solver.get_all_values(cs, symbol)\n                self.assertEqual(len(values), 1)\n                setattr(cpu, e.reg_name, values[0])\n                setattr(cpu, '{pc_reg}', {test['pre']['registers'][pc_reg]:x})")
    frame_base = 'RBP' if 'RBP' in test['pre']['registers'] else 'EBP'
    print(f"\n            except ConcretizeMemory as e:\n                symbol = getattr(cpu, '{frame_base}')\n                if isinstance(symbol, Expression):\n                    values = solver.get_all_values(cs, symbol)\n                    self.assertEqual(len(values), 1)\n                    setattr(cpu, '{frame_base}', values[0])\n                for i in range(e.size):\n                    symbol = mem[e.address+i]\n                    if isinstance(symbol, Expression):\n                        values = solver.get_all_values(cs, symbol)\n                        self.assertEqual(len(values), 1)\n                        mem[e.address+i] = values[0]\n                setattr(cpu, '{pc_reg}', {test['pre']['registers'][pc_reg]:x})")
    print('\n        condition = True')
    for (addr, byte) in test['pos']['memory'].items():
        print(f'        condition = Operators.AND(condition, cpu.read_int(0x{addr:x}, 8)== ord({byte!r}))')
    for (reg_name, value) in test['pos']['registers'].items():
        if isFlag(reg_name):
            print(f'        condition = Operators.AND(condition, cpu.{reg_name} == {value!r})')
        else:
            print(f'        condition = Operators.AND(condition, cpu.{reg_name} == 0x{value:x})')
    print('\n        with cs as temp_cs:\n            temp_cs.add(condition)\n            self.assertTrue(solver.check(temp_cs))\n        with cs as temp_cs:\n            temp_cs.add(condition == False)\n            self.assertFalse(solver.check(temp_cs))\n          ')
print("\nif __name__ == '__main__':\n    unittest.main()\n")