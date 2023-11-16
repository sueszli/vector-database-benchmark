from io import BytesIO
import unittest
import tempfile
import os
import gc
import pickle
import fcntl
import resource
from itertools import *
import sys
from manticore.native.memory import *
from manticore.core.smtlib import Z3Solver, Operators, issymbolic
from manticore.core.smtlib.expression import *
from manticore.core.smtlib.visitors import *
solver = Z3Solver.instance()

class LazyMemoryTest(unittest.TestCase):
    _multiprocess_can_split_ = True

    def test_basic(self):
        if False:
            return 10
        cs = ConstraintSet()
        mem = LazySMemory32(cs)
        mem.mmap(0, 4096, 'rwx', name='map')
        m = mem.maps.pop()
        self.assertIsInstance(m, AnonMap)
        self.assertEqual(m.name, 'map')

    def test_read(self):
        if False:
            i = 10
            return i + 15
        cs = ConstraintSet()
        mem = LazySMemory32(cs)
        mem.mmap(0, 4096, 'rwx', name='map')
        val_mapped = mem.read(0, 4)
        for val in val_mapped:
            self.assertIsInstance(val, bytes)
        with self.assertRaises(InvalidMemoryAccess):
            mem.read(8096, 4)

    def test_sym_read_mapped(self):
        if False:
            return 10
        cs = ConstraintSet()
        mem = LazySMemory32(cs)
        mem.mmap(0, 4096, 'rwx', name='map')
        addr = cs.new_bitvec(32)
        cs.add(addr >= 4092)
        cs.add(addr < 4098)
        with cs as new_cs:
            new_cs.add(mem.valid_ptr(addr))
            vals = solver.get_all_values(new_cs, addr)
            self.assertGreater(len(vals), 0)
            for v in vals:
                print(v)
                self.assertTrue(0 <= v < 4096)
        with cs as new_cs:
            new_cs.add(mem.invalid_ptr(addr))
            vals = solver.get_all_values(new_cs, addr)
            self.assertGreater(len(vals), 0)
            for v in vals:
                self.assertFalse(0 <= v < 4096)
        val = mem.read(addr, 1)[0]
        self.assertIsInstance(val, Expression)

    def test_lazysymbolic_basic_constrained_read(self):
        if False:
            for i in range(10):
                print('nop')
        cs = ConstraintSet()
        mem = LazySMemory32(cs)
        sym = cs.new_bitvec(32)
        cs.add(sym.uge(4095))
        cs.add(sym.ule(4112))
        self.assertRaises(MemoryException, mem.__getitem__, 4096)
        first = mem.mmap(4096, 4096, 'rw')
        self.assertEqual(first, 4096)
        mem.write(4096, b'\x00')
        self.assertEqual(solver.get_all_values(cs, mem[4096]), [b'\x00'])

    def test_arraymap(self):
        if False:
            for i in range(10):
                print('nop')
        m = ArrayMap(4096, 4096, 'rwx', 32)
        (head, tail) = m.split(6144)
        self.assertEqual(head.start, 4096)
        self.assertEqual(tail.start, 6144)
        self.assertEqual(len(head), 2048)
        self.assertEqual(len(tail), 2048)
        self.assertEqual(head.perms, m.perms)
        self.assertEqual(tail.perms, m.perms)
        reduced = m.__reduce__()
        self.assertIs(reduced[0], ArrayMap)
        sel = m[1]
        self.assertIsInstance(sel, ArraySelect)
        pre_array = m._array.array
        m[1] = 1
        post_array = m._array.array
        self.assertIsNot(pre_array, post_array)

    def test_lazysymbolic_mmapfile(self):
        if False:
            i = 10
            return i + 15
        mem = LazySMemory32(ConstraintSet())
        self.assertEqual(len(mem.mappings()), 0)
        rwx_file = tempfile.NamedTemporaryFile('w+b', delete=False)
        rwx_file.file.write(b'a' * 4097)
        rwx_file.close()
        addr_a = mem.mmapFile(0, 4096, 'rwx', rwx_file.name)
        self.assertEqual(len(mem.mappings()), 1)
        self.assertEqual(mem[addr_a], b'a')
        self.assertEqual(mem[addr_a + 4096 // 2], b'a')
        self.assertEqual(mem[addr_a + (4096 - 1)], b'a')
        self.assertRaises(MemoryException, mem.__getitem__, addr_a + 4096)
        rwx_file = tempfile.NamedTemporaryFile('w+b', delete=False)
        rwx_file.file.write(b'b' * 4097)
        rwx_file.close()
        addr_b = mem.mmapFile(0, 4096, 'rw', rwx_file.name)
        self.assertEqual(len(mem.mappings()), 2)
        self.assertEqual(mem[addr_b], b'b')
        self.assertEqual(mem[addr_b + 4096 // 2], b'b')
        self.assertEqual(mem[addr_b + (4096 - 1)], b'b')
        rwx_file = tempfile.NamedTemporaryFile('w+b', delete=False)
        rwx_file.file.write(b'c' * 4097)
        rwx_file.close()
        addr_c = mem.mmapFile(0, 4096, 'rx', rwx_file.name)
        self.assertEqual(len(mem.mappings()), 3)
        self.assertEqual(mem[addr_c], b'c')
        self.assertEqual(mem[addr_c + 4096 // 2], b'c')
        self.assertEqual(mem[addr_c + (4096 - 1)], b'c')
        rwx_file = tempfile.NamedTemporaryFile('w+b', delete=False)
        rwx_file.file.write(b'd' * 4097)
        rwx_file.close()
        addr_d = mem.mmapFile(0, 4096, 'r', rwx_file.name)
        self.assertEqual(len(mem.mappings()), 4)
        self.assertEqual(mem[addr_d], b'd')
        self.assertEqual(mem[addr_d + 4096 // 2], b'd')
        self.assertEqual(mem[addr_d + (4096 - 1)], b'd')
        rwx_file = tempfile.NamedTemporaryFile('w+b', delete=False)
        rwx_file.file.write(b'e' * 4097)
        rwx_file.close()
        addr_e = mem.mmapFile(0, 4096, 'w', rwx_file.name)
        self.assertEqual(len(mem.mappings()), 5)
        self.assertRaises(MemoryException, mem.__getitem__, addr_e)
        self.assertRaises(MemoryException, mem.__getitem__, addr_e + 4096 // 2)
        self.assertRaises(MemoryException, mem.__getitem__, addr_e + (4096 - 1))

    def test_lazysymbolic_map_containing(self):
        if False:
            for i in range(10):
                print('nop')
        cs = ConstraintSet()
        mem = LazySMemory32(cs)
        valid = cs.new_bitvec(32)
        invalid = cs.new_bitvec(32)
        mem.mmap(4096, 4096, 'rw')
        m = list(mem._maps)[0]
        cs.add(valid > 4096)
        cs.add(valid < 4098)
        cs.add(invalid < 4096)
        ret = mem._deref_can_succeed(m, valid, 1)
        self.assertTrue(ret)
        ret = mem._deref_can_succeed(m, invalid, 1)
        self.assertFalse(ret)
        ret = mem._deref_can_succeed(m, 4096, 1)
        self.assertTrue(ret)
        ret = mem._deref_can_succeed(m, 2048, 1)
        self.assertFalse(ret)
        ret = mem._deref_can_succeed(m, 4095, 2)
        self.assertFalse(ret)
        ret = mem._deref_can_succeed(m, 4096, 2)
        self.assertTrue(ret)
        ret = mem._deref_can_succeed(m, 4096, 4095)
        self.assertTrue(ret)
        ret = mem._deref_can_succeed(m, 4096, 4096)
        self.assertFalse(ret)

    @unittest.skip("Disabled because it takes 4+ minutes; get_all_values() isn't returning all possible addresses")
    def test_lazysymbolic_constrained_deref(self):
        if False:
            for i in range(10):
                print('nop')
        cs = ConstraintSet()
        mem = LazySMemory32(cs)
        mem.page_bit_size = 12
        Size = 4096
        PatternSize = 256
        Constant = 72
        ConstantMask = 255
        if False:
            mem.page_bit_size = 10
            Size = 2048
            PatternSize = 128
            Constant = 72
            ConstantMask = 255
        first = mem.mmap(Size, Size, 'rw')
        mem.write(first, bytes(islice(cycle(range(PatternSize)), Size)))
        sym = cs.new_bitvec(32)
        vals = mem.read(sym, 4)
        cs.add(vals[0] == Constant)
        cs.add(vals[1] == Constant + 1)
        possible_addrs = solver.get_all_values(cs, sym)
        print('possible addrs: ', [hex(a) for a in sorted(possible_addrs)])
        for i in possible_addrs:
            self.assertTrue(i & ConstantMask == Constant)
        self.assertEqual(len(possible_addrs), Size // PatternSize)
if __name__ == '__main__':
    unittest.main()