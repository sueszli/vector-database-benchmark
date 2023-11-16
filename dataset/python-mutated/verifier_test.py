import dis
import os
import sys
import types
import unittest
from compiler import compile, opcode_static as op
from inspect import cleandoc
from typing import List, Tuple
sys.path.append(os.path.join(sys.path[0], '..', 'fuzzer'))
import cfgutil
import verifier
from cfgutil import BytecodeOp
from verifier import VerificationError, Verifier

class VerifierTests(unittest.TestCase):

    def convert_bytecodes_to_byte_representation(self, bytecodes: List[Tuple]) -> bytes:
        if False:
            for i in range(10):
                print('nop')
        b_list = []
        if len(bytecodes) == 0:
            return b''
        for i in bytecodes:
            if isinstance(i, tuple) and len(i) == 2:
                b_list.append(i[0])
                b_list.append(i[1])
            else:
                b_list.append(i)
        return bytes(b_list)

    def compile_helper(self, source: str, bytecodes: List[Tuple]=None, consts: tuple=None, varnames: tuple=None, names: tuple=None, freevars: tuple=None, cellvars: tuple=None, stacksize: int=None) -> types.CodeType:
        if False:
            return 10
        code = compile(source, '', 'exec')
        if bytecodes is not None:
            code = code.replace(co_code=self.convert_bytecodes_to_byte_representation(bytecodes))
        if stacksize is not None:
            code = code.replace(co_stacksize=stacksize)
        if consts is not None:
            code = code.replace(co_consts=consts)
        if varnames is not None:
            code = code.replace(co_varnames=varnames)
        if names is not None:
            code = code.replace(co_names=names)
        if freevars is not None:
            code = code.replace(co_freevars=freevars)
        if cellvars is not None:
            code = code.replace(co_cellvars=cellvars)
        return code

class VerifierBasicsTest(VerifierTests):

    def test_length_cannot_be_odd(self):
        if False:
            print('Hello World!')
        code = self.compile_helper('', [(op.opcode.LOAD_CONST, 57), op.opcode.POP_TOP])
        self.assertRaisesRegex(VerificationError, 'Bytecode length cannot be odd', Verifier.validate_code, code)

    def test_length_cannot_be_zero(self):
        if False:
            for i in range(10):
                print('nop')
        code = self.compile_helper('', [])
        self.assertRaisesRegex(VerificationError, 'Bytecode length cannot be zero or negative', Verifier.validate_code, code)

    def test_op_name_must_exist(self):
        if False:
            print('Hello World!')
        code = self.compile_helper('', [(op.opcode.LOAD_CONST, 0), (7, 68)])
        self.assertRaisesRegex(VerificationError, 'Operation 7 at offset 2 does not exist', Verifier.validate_code, code)

    def test_cannot_jump_outside_of_file(self):
        if False:
            i = 10
            return i + 15
        code = self.compile_helper('', [(op.opcode.LOAD_CONST, 0), (op.opcode.JUMP_ABSOLUTE, 153)])
        self.assertRaisesRegex(VerificationError, 'can not jump out of bounds$', Verifier.validate_code, code)

    def test_error_in_nested_code_object(self):
        if False:
            print('Hello World!')
        code = self.compile_helper('', bytecodes=[(op.opcode.LOAD_CONST, 0), (op.opcode.POP_TOP, 0)], consts=(self.compile_helper('', [(op.opcode.LOAD_CONST, 0), (op.opcode.JUMP_ABSOLUTE, 153)]),))
        self.assertRaisesRegex(VerificationError, 'can not jump out of bounds$', Verifier.validate_code, code)

class VerifierCFGTests(VerifierTests):

    def test_cfg_with_single_block(self):
        if False:
            for i in range(10):
                print('nop')
        source = cleandoc('\n            x = 3\n            x += 1')
        code = self.compile_helper(source)
        block_map = Verifier.create_blocks(Verifier.parse_bytecode(code.co_code))
        assert str(block_map) == cleandoc('\n        bb0:\n          LOAD_CONST : 0\n          STORE_NAME : 0\n          LOAD_NAME : 0\n          LOAD_CONST : 1\n          INPLACE_ADD : 0\n          STORE_NAME : 0\n          LOAD_CONST : 2\n          RETURN_VALUE : 0')

    def test_cfg_with_conditional(self):
        if False:
            return 10
        source = cleandoc('\n            x, y = 3, 0\n            if x > 0:\n                y += 1\n            elif x == 0:\n                y += 3\n            else:\n                y += 2')
        code = self.compile_helper(source)
        block_map = Verifier.create_blocks(Verifier.parse_bytecode(code.co_code))
        self.assertEqual(str(block_map), cleandoc('\n        bb0:\n          LOAD_CONST : 0\n          UNPACK_SEQUENCE : 2\n          STORE_NAME : 0\n          STORE_NAME : 1\n          LOAD_NAME : 0\n          LOAD_CONST : 1\n          COMPARE_OP : 4\n          POP_JUMP_IF_FALSE bb2\n        bb1:\n          LOAD_NAME : 1\n          LOAD_CONST : 2\n          INPLACE_ADD : 0\n          STORE_NAME : 1\n          LOAD_CONST : 5\n          RETURN_VALUE : 0\n        bb2:\n          LOAD_NAME : 0\n          LOAD_CONST : 1\n          COMPARE_OP : 2\n          POP_JUMP_IF_FALSE bb4\n        bb3:\n          LOAD_NAME : 1\n          LOAD_CONST : 3\n          INPLACE_ADD : 0\n          STORE_NAME : 1\n          LOAD_CONST : 5\n          RETURN_VALUE : 0\n        bb4:\n          LOAD_NAME : 1\n          LOAD_CONST : 4\n          INPLACE_ADD : 0\n          STORE_NAME : 1\n          LOAD_CONST : 5\n          RETURN_VALUE : 0'))

    def test_cfg_with_loop(self):
        if False:
            for i in range(10):
                print('nop')
        source = cleandoc('\n            arr = []\n            i = 0\n            while i < 10:\n              arr.append(i)\n              i+=1')
        code = self.compile_helper(source)
        block_map = Verifier.create_blocks(Verifier.parse_bytecode(code.co_code))
        self.assertEqual(str(block_map), cleandoc('\n        bb0:\n          BUILD_LIST : 0\n          STORE_NAME : 0\n          LOAD_CONST : 0\n          STORE_NAME : 1\n          LOAD_NAME : 1\n          LOAD_CONST : 1\n          COMPARE_OP : 0\n          POP_JUMP_IF_FALSE bb3\n        bb1:\n          LOAD_NAME : 0\n          LOAD_METHOD : 2\n          LOAD_NAME : 1\n          CALL_METHOD : 1\n          POP_TOP : 0\n          LOAD_NAME : 1\n          LOAD_CONST : 2\n          INPLACE_ADD : 0\n          STORE_NAME : 1\n          LOAD_NAME : 1\n          LOAD_CONST : 1\n          COMPARE_OP : 0\n          POP_JUMP_IF_TRUE bb1\n        bb2:\n          LOAD_CONST : 3\n          RETURN_VALUE : 0\n        bb3:\n          LOAD_CONST : 3\n          RETURN_VALUE : 0'))

    def test_cfg_with_function_call(self):
        if False:
            return 10
        source = cleandoc('\n            arr = []\n            for i in range(10):\n              arr.append(i)')
        code = self.compile_helper(source)
        block_map = Verifier.create_blocks(Verifier.parse_bytecode(code.co_code))
        self.assertEqual(str(block_map), cleandoc('\n        bb0:\n          BUILD_LIST : 0\n          STORE_NAME : 0\n          LOAD_NAME : 1\n          LOAD_CONST : 0\n          CALL_FUNCTION : 1\n          GET_ITER : 0\n        bb1:\n          FOR_ITER bb3\n        bb2:\n          STORE_NAME : 2\n          LOAD_NAME : 0\n          LOAD_METHOD : 3\n          LOAD_NAME : 2\n          CALL_METHOD : 1\n          POP_TOP : 0\n          JUMP_ABSOLUTE bb1\n        bb3:\n          LOAD_CONST : 1\n          RETURN_VALUE : 0'))

    def test_cfg_try_except(self):
        if False:
            for i in range(10):
                print('nop')
        source = cleandoc('\n        y = 9\n        a = 3\n        try:\n            c = y+a\n        except:\n            raise\n        ')
        code = self.compile_helper(source)
        block_map = Verifier.create_blocks(Verifier.parse_bytecode(code.co_code))
        self.assertEqual(str(block_map), cleandoc('\n        bb0:\n          LOAD_CONST : 0\n          STORE_NAME : 0\n          LOAD_CONST : 1\n          STORE_NAME : 1\n          SETUP_FINALLY : 7\n          LOAD_NAME : 0\n          LOAD_NAME : 1\n          BINARY_ADD : 0\n          STORE_NAME : 2\n          POP_BLOCK : 0\n          LOAD_CONST : 2\n          RETURN_VALUE : 0\n        bb1:\n          POP_TOP : 0\n          POP_TOP : 0\n          POP_TOP : 0\n          RAISE_VARARGS : 0'))

    def test_cfg_try_except_else_finally(self):
        if False:
            print('Hello World!')
        source = cleandoc('\n        y = 9\n        a = "b"\n        try:\n            c = y+a\n        except:\n            raise\n        else:\n            y+=1\n        finally:\n            y+=3\n        ')
        code = self.compile_helper(source)
        block_map = Verifier.create_blocks(Verifier.parse_bytecode(code.co_code))
        self.assertEqual(str(block_map), cleandoc('\n        bb0:\n          LOAD_CONST : 0\n          STORE_NAME : 0\n          LOAD_CONST : 1\n          STORE_NAME : 1\n          SETUP_FINALLY : 22\n          SETUP_FINALLY : 6\n          LOAD_NAME : 0\n          LOAD_NAME : 1\n          BINARY_ADD : 0\n          STORE_NAME : 2\n          POP_BLOCK : 0\n          JUMP_FORWARD bb2\n        bb1:\n          POP_TOP : 0\n          POP_TOP : 0\n          POP_TOP : 0\n          RAISE_VARARGS : 0\n        bb2:\n          LOAD_NAME : 0\n          LOAD_CONST : 2\n          INPLACE_ADD : 0\n          STORE_NAME : 0\n          POP_BLOCK : 0\n          LOAD_NAME : 0\n          LOAD_CONST : 3\n          INPLACE_ADD : 0\n          STORE_NAME : 0\n          LOAD_CONST : 4\n          RETURN_VALUE : 0\n        bb3:\n          LOAD_NAME : 0\n          LOAD_CONST : 3\n          INPLACE_ADD : 0\n          STORE_NAME : 0\n          RERAISE : 0'))

    def test_cfg_continue_statement_in_try(self):
        if False:
            i = 10
            return i + 15
        source = cleandoc('\n        for i in range(10):\n            x = 0\n            z = 2\n            try:\n                y = x+z\n                continue\n            except:\n                raise\n            finally:\n                x+=1\n        ')
        code = self.compile_helper(source)
        block_map = Verifier.create_blocks(Verifier.parse_bytecode(code.co_code))
        self.assertEqual(str(block_map), cleandoc('\n        bb0:\n          LOAD_NAME : 0\n          LOAD_CONST : 0\n          CALL_FUNCTION : 1\n          GET_ITER : 0\n        bb1:\n          FOR_ITER bb5\n        bb2:\n          STORE_NAME : 1\n          LOAD_CONST : 1\n          STORE_NAME : 2\n          LOAD_CONST : 2\n          STORE_NAME : 3\n          SETUP_FINALLY : 16\n          SETUP_FINALLY : 11\n          LOAD_NAME : 2\n          LOAD_NAME : 3\n          BINARY_ADD : 0\n          STORE_NAME : 4\n          POP_BLOCK : 0\n          POP_BLOCK : 0\n          LOAD_NAME : 2\n          LOAD_CONST : 3\n          INPLACE_ADD : 0\n          STORE_NAME : 2\n          JUMP_ABSOLUTE bb1\n        bb3:\n          POP_TOP : 0\n          POP_TOP : 0\n          POP_TOP : 0\n          RAISE_VARARGS : 0\n        bb4:\n          LOAD_NAME : 2\n          LOAD_CONST : 3\n          INPLACE_ADD : 0\n          STORE_NAME : 2\n          RERAISE : 0\n        bb5:\n          LOAD_CONST : 4\n          RETURN_VALUE : 0'))

class VerifierStackDepthTests(VerifierTests):

    def test_cannot_pop_from_empty_stack(self):
        if False:
            for i in range(10):
                print('nop')
        code = self.compile_helper('', [(op.opcode.LOAD_CONST, 0), (op.opcode.POP_TOP, 0), (op.opcode.POP_TOP, 0)])
        self.assertRaisesRegex(VerificationError, 'Stack depth -1 dips below minimum of 0 for operation POP_TOP @ offset 4', Verifier.validate_code, code)

    def test_stack_depth_cannot_exceed_max(self):
        if False:
            i = 10
            return i + 15
        code = self.compile_helper('', [(op.opcode.LOAD_CONST, 0), (op.opcode.LOAD_CONST, 0), (op.opcode.LOAD_CONST, 0)], stacksize=1)
        self.assertRaisesRegex(VerificationError, 'Stack depth 2 exceeds maximum of 1 for operation LOAD_CONST @ offset 2', Verifier.validate_code, code)

    def test_branch(self):
        if False:
            i = 10
            return i + 15
        source = cleandoc('\n        x, y = 0, 0\n        if x: y += 1\n        else: y += 3')
        code = self.compile_helper(source)
        self.assertTrue(Verifier.validate_code(code))

    def test_for_loop(self):
        if False:
            return 10
        source = cleandoc('\n        x = 0\n        for i in range(10):\n          x += 1')
        code = self.compile_helper(source)
        self.assertTrue(Verifier.validate_code(code))

    def test_while_loop(self):
        if False:
            for i in range(10):
                print('nop')
        source = cleandoc('\n        i = 0\n        while i < 10:\n          i+=1')
        code = self.compile_helper(source)
        self.assertTrue(Verifier.validate_code(code))

    def test_cannot_pop_from_empty_stack_during_loop(self):
        if False:
            print('Hello World!')
        bytecodes = [(op.opcode.LOAD_CONST, 0), (op.opcode.UNPACK_SEQUENCE, 2), (op.opcode.STORE_NAME, 0), (op.opcode.STORE_NAME, 1), (op.opcode.LOAD_NAME, 0), (op.opcode.POP_JUMP_IF_FALSE, 11), (op.opcode.LOAD_NAME, 1), (op.opcode.LOAD_CONST, 1), (op.opcode.POP_TOP, 0), (op.opcode.INPLACE_ADD, 0), (op.opcode.STORE_NAME, 1), (op.opcode.JUMP_FORWARD, 4), (op.opcode.LOAD_NAME, 1), (op.opcode.LOAD_CONST, 2), (op.opcode.INPLACE_ADD, 0), (op.opcode.STORE_NAME, 1), (op.opcode.LOAD_CONST, 3), (op.opcode.RETURN_VALUE, 0)]
        code = self.compile_helper('', bytecodes, consts=(1, 2, 3, 4), names=('e', 'e', 'e'), stacksize=10)
        self.assertRaisesRegex(VerificationError, 'Stack depth -1 dips below minimum of 0 for operation STORE_NAME @ offset 20', Verifier.validate_code, code)

    def test_stack_depth_should_not_exceed_max_while_looping(self):
        if False:
            for i in range(10):
                print('nop')
        bytecodes = [(op.opcode.LOAD_CONST, 0), (op.opcode.UNPACK_SEQUENCE, 2), (op.opcode.STORE_NAME, 0), (op.opcode.STORE_NAME, 1), (op.opcode.LOAD_NAME, 0), (op.opcode.POP_JUMP_IF_FALSE, 11), (op.opcode.LOAD_NAME, 1), (op.opcode.LOAD_CONST, 1), (op.opcode.LOAD_CONST, 1), (op.opcode.LOAD_CONST, 1), (op.opcode.LOAD_CONST, 1), (op.opcode.LOAD_CONST, 1), (op.opcode.POP_TOP, 0), (op.opcode.INPLACE_ADD, 0), (op.opcode.STORE_NAME, 1), (op.opcode.JUMP_FORWARD, 4), (op.opcode.LOAD_NAME, 1), (op.opcode.LOAD_CONST, 2), (op.opcode.INPLACE_ADD, 0), (op.opcode.STORE_NAME, 1), (op.opcode.LOAD_CONST, 3), (op.opcode.RETURN_VALUE, 0)]
        code = self.compile_helper('', bytecodes, consts=(1, 2, 3, 4), names=('e', 'e', 'e'))
        self.assertRaisesRegex(VerificationError, 'Stack depth 2 exceeds maximum of 1 for operation UNPACK_SEQUENCE @ offset 2', Verifier.validate_code, code)

    def test_branch_with_nested_conditions(self):
        if False:
            i = 10
            return i + 15
        source = cleandoc('\n        x, y, arr = 0, 0, []\n        if x:\n          y = 5 if x > 3 else 3\n        else:\n          arr.append(x)\n        arr.append(y)')
        code = self.compile_helper(source)
        self.assertTrue(Verifier.validate_code(code))

    def test_nested_loop(self):
        if False:
            print('Hello World!')
        source = cleandoc('\n        x, arr = 0, []\n        for i in range(10):\n          for j in range(12):\n            x+=3\n            arr.append(x)')
        code = self.compile_helper(source)
        self.assertTrue(Verifier.validate_code(code))

    def test_while_loop_with_multiple_conditions(self):
        if False:
            while True:
                i = 10
        source = cleandoc('\n        i, stack = 0, [1, 2, 3, 4]\n        while stack and i < 10:\n          stack.pop()\n          i+=3')
        code = self.compile_helper(source)
        self.assertTrue(Verifier.validate_code(code))

    def test_recursive_function_maintains_stack_depth(self):
        if False:
            return 10
        source = cleandoc('\n        y = 7\n        def f(x):\n          if x == 0:\n            return 0\n          if x == 1:\n            return 1\n          return f(x-1) + f(x-2)\n        print(f(y))')
        code = self.compile_helper(source)
        self.assertTrue(Verifier.validate_code(code))

    def test_try_except_maintains_stack_depth(self):
        if False:
            return 10
        source = cleandoc('\n        y = 9\n        a = 3\n        try:\n            c = y+a\n        except:\n            raise\n        ')
        code = self.compile_helper(source)
        self.assertTrue(Verifier.validate_code(code))

    def test_try_except_else_finally_maintains_stack_depth(self):
        if False:
            i = 10
            return i + 15
        source = cleandoc('\n        y = 9\n        a = "b"\n        try:\n            c = y+a\n        except:\n            raise\n        else:\n            y+=1\n        finally:\n            y+=3\n        ')
        code = self.compile_helper(source)
        self.assertTrue(Verifier.validate_code(code))

    def test_try_except_not_handled_by_except(self):
        if False:
            while True:
                i = 10
        source = cleandoc('\n        y = 9\n        a = "b"\n        try:\n            c = y+a\n        except IndexError:\n            raise\n        finally:\n            y+=3\n        ')
        code = self.compile_helper(source)
        self.assertTrue(Verifier.validate_code(code))

    def test_try_except_with_continue_statement_in_try(self):
        if False:
            print('Hello World!')
        source = cleandoc('\n        for i in range(10):\n            x = 0\n            z = 2\n            try:\n                y = x+z\n                continue\n            except:\n                raise\n            finally:\n                x+=1\n        ')
        code = self.compile_helper(source)
        self.assertTrue(Verifier.validate_code(code))

    def test_try_except_with_break_statement_in_finally(self):
        if False:
            while True:
                i = 10
        source = cleandoc('\n        for i in range(10):\n            x = 0\n            z = 2\n            try:\n                y = x+z\n                continue\n            except:\n                raise\n            finally:\n                break\n        ')
        code = self.compile_helper(source)
        self.assertTrue(Verifier.validate_code(code))

    def test_try_except_with_return_statements(self):
        if False:
            print('Hello World!')
        source = cleandoc('\n        def f(x, y):\n            try:\n                x += y\n                return x\n            except:\n                raise\n            finally:\n                return y\n        ')
        code = self.compile_helper(source)
        self.assertTrue(Verifier.validate_code(code))

class VerifierOpArgTests(VerifierTests):

    def test_LOAD_CONST_with_valid_oparg_index_is_successful(self):
        if False:
            for i in range(10):
                print('nop')
        code = self.compile_helper('', bytecodes=[(op.opcode.LOAD_CONST, 0)], consts=(3,))
        self.assertTrue(Verifier.validate_code(code))

    def test_LOAD_CONST_oparg_type_can_be_any_object(self):
        if False:
            for i in range(10):
                print('nop')
        code = self.compile_helper('', bytecodes=[(op.opcode.LOAD_CONST, 0), (op.opcode.LOAD_CONST, 1), (op.opcode.LOAD_CONST, 2)], consts=(3, None, 'hello'), stacksize=4)
        self.assertTrue(Verifier.validate_code(code))

    def test_LOAD_CONST_with_invalid_oparg_index_raises_exception(self):
        if False:
            for i in range(10):
                print('nop')
        code = self.compile_helper('', bytecodes=[(op.opcode.LOAD_CONST, 0), (op.opcode.LOAD_CONST, 1)], consts=(3,), stacksize=2)
        self.assertRaisesRegex(VerificationError, 'Argument index 1 out of bounds for size 1 for operation LOAD_CONST @ offset 2', Verifier.validate_code, code)

    def test_LOAD_CLASS_with_valid_oparg_index_is_successful(self):
        if False:
            print('Hello World!')
        code = self.compile_helper('', bytecodes=[(op.opcode.LOAD_CLASS, 0)], consts=((1, 3),))
        self.assertTrue(Verifier.validate_code(code))

    def test_LOAD_CLASS_oparg_type_can_be_any_tuple(self):
        if False:
            while True:
                i = 10
        code = self.compile_helper('', bytecodes=[(op.opcode.LOAD_CLASS, 0)], consts=(tuple(),), stacksize=4)
        self.assertTrue(Verifier.validate_code(code))

    def test_LOAD_CLASS_oparg_type_cannot_be_non_tuple(self):
        if False:
            while True:
                i = 10
        code = self.compile_helper('', bytecodes=[(op.opcode.LOAD_CLASS, 0)], consts=(object(),), stacksize=1)
        self.assertRaisesRegex(VerificationError, 'Incorrect oparg type of object, expected tuple for operation LOAD_CLASS @ offset 0', Verifier.validate_code, code)

    def test_LOAD_CLASS_with_invalid_oparg_index_raises_exception(self):
        if False:
            for i in range(10):
                print('nop')
        code = self.compile_helper('', bytecodes=[(op.opcode.LOAD_CONST, 0), (op.opcode.LOAD_CLASS, 1)], consts=(3,), stacksize=2)
        self.assertRaisesRegex(VerificationError, 'Argument index 1 out of bounds for size 1 for operation LOAD_CLASS @ offset 2', Verifier.validate_code, code)

    def test_LOAD_FIELD_with_valid_oparg_index_is_successful(self):
        if False:
            return 10
        code = self.compile_helper('', bytecodes=[(op.opcode.LOAD_CONST, 0)], consts=((1, 3),))
        self.assertTrue(Verifier.validate_code(code))

    def test_LOAD_FIELD_oparg_type_can_be_any_tuple(self):
        if False:
            i = 10
            return i + 15
        code = self.compile_helper('', bytecodes=[(op.opcode.LOAD_FIELD, 0)], consts=(tuple(),), stacksize=1)
        self.assertTrue(Verifier.validate_code(code))

    def test_LOAD_FIELD_oparg_type_cannot_be_non_tuple(self):
        if False:
            return 10
        code = self.compile_helper('', bytecodes=[(op.opcode.LOAD_FIELD, 0)], consts=(object(),), stacksize=1)
        self.assertRaisesRegex(VerificationError, 'Incorrect oparg type of object, expected tuple for operation LOAD_FIELD @ offset 0', Verifier.validate_code, code)

    def test_LOAD_FIELD_with_invalid_oparg_index_raises_exception(self):
        if False:
            print('Hello World!')
        code = self.compile_helper('', bytecodes=[(op.opcode.LOAD_CONST, 0), (op.opcode.LOAD_CONST, 1)], consts=(3,), stacksize=2)
        self.assertRaisesRegex(VerificationError, 'Argument index 1 out of bounds for size 1 for operation LOAD_CONST @ offset 2', Verifier.validate_code, code)

    def test_STORE_FIELD_with_valid_oparg_index_is_successful(self):
        if False:
            return 10
        code = self.compile_helper('', bytecodes=[(op.opcode.LOAD_CONST, 0), (op.opcode.LOAD_CONST, 1), (op.opcode.STORE_FIELD, 1)], consts=(3, (1, 3)), stacksize=4)
        self.assertTrue(Verifier.validate_code(code))

    def test_STORE_FIELD_oparg_type_can_be_any_tuple(self):
        if False:
            return 10
        code = self.compile_helper('', bytecodes=[(op.opcode.LOAD_CONST, 0), (op.opcode.LOAD_CONST, 1), (op.opcode.LOAD_CONST, 2), (op.opcode.STORE_FIELD, 3)], consts=(3, 2, 1, tuple()), stacksize=4)
        self.assertTrue(Verifier.validate_code(code))

    def test_STORE_FIELD_oparg_type_cannot_be_non_tuple(self):
        if False:
            i = 10
            return i + 15
        code = self.compile_helper('', bytecodes=[(op.opcode.LOAD_CONST, 0), (op.opcode.LOAD_CONST, 1), (op.opcode.LOAD_CONST, 2), (op.opcode.STORE_FIELD, 3)], consts=(3, 2, 1, object()), stacksize=4)
        self.assertRaisesRegex(VerificationError, 'Incorrect oparg type of object, expected tuple for operation STORE_FIELD @ offset 6', Verifier.validate_code, code)

    def test_STORE_FIELD_with_invalid_oparg_index_raises_exception(self):
        if False:
            for i in range(10):
                print('nop')
        code = self.compile_helper('', bytecodes=[(op.opcode.LOAD_CONST, 0), (op.opcode.STORE_FIELD, 1)], consts=(3,), stacksize=2)
        self.assertRaisesRegex(VerificationError, 'Argument index 1 out of bounds for size 1 for operation STORE_FIELD @ offset 2', Verifier.validate_code, code)

    def test_CAST_with_valid_oparg_index_is_successful(self):
        if False:
            i = 10
            return i + 15
        code = self.compile_helper('', bytecodes=[(op.opcode.CAST, 0)], consts=((1, 3),), stacksize=4)
        self.assertTrue(Verifier.validate_code(code))

    def test_CAST_oparg_type_can_be_any_tuple(self):
        if False:
            i = 10
            return i + 15
        code = self.compile_helper('', bytecodes=[(op.opcode.CAST, 0)], consts=(tuple(),), stacksize=4)
        self.assertTrue(Verifier.validate_code(code))

    def test_CAST_oparg_type_cannot_be_non_tuple(self):
        if False:
            print('Hello World!')
        code = self.compile_helper('', bytecodes=[(op.opcode.CAST, 0)], consts=(object(),), stacksize=1)
        self.assertRaisesRegex(VerificationError, 'Incorrect oparg type of object, expected tuple for operation CAST @ offset 0', Verifier.validate_code, code)

    def test_CAST_with_invalid_oparg_index_raises_exception(self):
        if False:
            return 10
        code = self.compile_helper('', bytecodes=[(op.opcode.LOAD_CONST, 0), (op.opcode.CAST, 1)], consts=(3,), stacksize=2)
        self.assertRaisesRegex(VerificationError, 'Argument index 1 out of bounds for size 1 for operation CAST @ offset 2', Verifier.validate_code, code)

    def test_PRIMITIVE_BOX_with_valid_oparg_index_is_successful(self):
        if False:
            for i in range(10):
                print('nop')
        code = self.compile_helper('', bytecodes=[(op.opcode.PRIMITIVE_BOX, 0)], consts=((1, 3),), stacksize=1)
        self.assertTrue(Verifier.validate_code(code))

    def test_PRIMITIVE_BOX_oparg_type_can_be_any_tuple(self):
        if False:
            print('Hello World!')
        code = self.compile_helper('', bytecodes=[(op.opcode.PRIMITIVE_BOX, 0)], consts=(tuple(),), stacksize=4)
        self.assertTrue(Verifier.validate_code(code))

    def test_PRIMITIVE_BOX_oparg_type_cannot_be_non_tuple(self):
        if False:
            print('Hello World!')
        code = self.compile_helper('', bytecodes=[(op.opcode.PRIMITIVE_BOX, 0)], consts=(object(),), stacksize=1)
        self.assertRaisesRegex(VerificationError, 'Incorrect oparg type of object, expected tuple for operation PRIMITIVE_BOX @ offset 0', Verifier.validate_code, code)

    def test_PRIMITIVE_BOX_with_invalid_oparg_index_raises_exception(self):
        if False:
            i = 10
            return i + 15
        code = self.compile_helper('', bytecodes=[(op.opcode.LOAD_CONST, 0), (op.opcode.PRIMITIVE_BOX, 1)], consts=(3,), stacksize=2)
        self.assertRaisesRegex(VerificationError, 'Argument index 1 out of bounds for size 1 for operation PRIMITIVE_BOX @ offset 2', Verifier.validate_code, code)

    def test_PRIMITIVE_UNBOX_with_valid_oparg_index_is_successful(self):
        if False:
            for i in range(10):
                print('nop')
        code = self.compile_helper('', bytecodes=[(op.opcode.PRIMITIVE_UNBOX, 0)], consts=((1, 3),), stacksize=1)
        self.assertTrue(Verifier.validate_code(code))

    def test_PRIMITIVE_UNBOX_oparg_type_can_be_any_tuple(self):
        if False:
            while True:
                i = 10
        code = self.compile_helper('', bytecodes=[(op.opcode.PRIMITIVE_UNBOX, 0)], consts=(tuple(),), stacksize=4)
        self.assertTrue(Verifier.validate_code(code))

    def test_PRIMITIVE_UNBOX_oparg_type_cannot_be_non_tuple(self):
        if False:
            return 10
        code = self.compile_helper('', bytecodes=[(op.opcode.PRIMITIVE_UNBOX, 0)], consts=(object(),), stacksize=1)
        self.assertRaisesRegex(VerificationError, 'Incorrect oparg type of object, expected tuple for operation PRIMITIVE_UNBOX @ offset 0', Verifier.validate_code, code)

    def test_PRIMITIVE_UNBOX_with_invalid_oparg_index_raises_exception(self):
        if False:
            i = 10
            return i + 15
        code = self.compile_helper('', bytecodes=[(op.opcode.LOAD_CONST, 0), (op.opcode.PRIMITIVE_UNBOX, 1)], consts=(3,), stacksize=2)
        self.assertRaisesRegex(VerificationError, 'Argument index 1 out of bounds for size 1 for operation PRIMITIVE_UNBOX @ offset 2', Verifier.validate_code, code)

    def test_TP_ALLOC_with_valid_oparg_index_is_successful(self):
        if False:
            while True:
                i = 10
        code = self.compile_helper('', bytecodes=[(op.opcode.TP_ALLOC, 0)], consts=((1, 2),), stacksize=1)
        self.assertTrue(Verifier.validate_code(code))

    def test_TP_ALLOC_oparg_type_can_be_any_tuple(self):
        if False:
            while True:
                i = 10
        code = self.compile_helper('', bytecodes=[(op.opcode.TP_ALLOC, 0)], consts=(tuple(),), stacksize=1)
        self.assertTrue(Verifier.validate_code(code))

    def test_TP_ALLOC_oparg_type_cannot_be_non_tuple(self):
        if False:
            for i in range(10):
                print('nop')
        code = self.compile_helper('', bytecodes=[(op.opcode.TP_ALLOC, 0)], consts=(object(),), stacksize=1)
        self.assertRaisesRegex(VerificationError, 'Incorrect oparg type of object, expected tuple for operation TP_ALLOC @ offset 0', Verifier.validate_code, code)

    def test_TP_ALLOC_with_invalid_oparg_index_raises_exception(self):
        if False:
            i = 10
            return i + 15
        code = self.compile_helper('', bytecodes=[(op.opcode.LOAD_CONST, 0), (op.opcode.TP_ALLOC, 1)], consts=(3,), stacksize=2)
        self.assertRaisesRegex(VerificationError, 'Argument index 1 out of bounds for size 1 for operation TP_ALLOC @ offset 2', Verifier.validate_code, code)

    def test_PRIMITIVE_LOAD_CONST_with_valid_oparg_index_is_successful(self):
        if False:
            for i in range(10):
                print('nop')
        code = self.compile_helper('', bytecodes=[(op.opcode.PRIMITIVE_LOAD_CONST, 0)], consts=(3,), stacksize=1)
        self.assertTrue(Verifier.validate_code(code))

    def test_PRIMITIVE_LOAD_CONST_oparg_type_can_be_any_int(self):
        if False:
            for i in range(10):
                print('nop')
        code = self.compile_helper('', bytecodes=[(op.opcode.PRIMITIVE_LOAD_CONST, 0)], consts=(1,), stacksize=1)
        self.assertTrue(Verifier.validate_code(code))

    def test_PRIMITIVE_LOAD_CONST_oparg_type_cannot_be_non_int(self):
        if False:
            for i in range(10):
                print('nop')
        code = self.compile_helper('', bytecodes=[(op.opcode.PRIMITIVE_LOAD_CONST, 0)], consts=('h',), stacksize=1)
        self.assertRaisesRegex(VerificationError, 'Incorrect oparg type of str, expected int for operation PRIMITIVE_LOAD_CONST @ offset 0', Verifier.validate_code, code)

    def test_PRIMITIVE_LOAD_CONST_with_invalid_oparg_index_raises_exception(self):
        if False:
            while True:
                i = 10
        code = self.compile_helper('', bytecodes=[(op.opcode.LOAD_CONST, 0), (op.opcode.PRIMITIVE_LOAD_CONST, 1)], consts=(3,), stacksize=2)
        self.assertRaisesRegex(VerificationError, 'Argument index 1 out of bounds for size 1 for operation PRIMITIVE_LOAD_CONST @ offset 2', Verifier.validate_code, code)

    def test_REFINE_TYPE_with_valid_oparg_index_is_successful(self):
        if False:
            i = 10
            return i + 15
        code = self.compile_helper('', bytecodes=[(op.opcode.REFINE_TYPE, 0)], consts=(('a', 'b'),), stacksize=1)
        self.assertTrue(Verifier.validate_code(code))

    def test_REFINE_TYPE_oparg_type_can_be_any_tuple(self):
        if False:
            print('Hello World!')
        code = self.compile_helper('', bytecodes=[(op.opcode.REFINE_TYPE, 0)], consts=(('s', 'str', 's'),), stacksize=1)
        self.assertTrue(Verifier.validate_code(code))

    def test_REFINE_TYPE_oparg_cannot_be_non_tuple(self):
        if False:
            print('Hello World!')
        code = self.compile_helper('', bytecodes=[(op.opcode.REFINE_TYPE, 0)], consts=('h',), stacksize=1)
        self.assertRaisesRegex(VerificationError, 'Incorrect oparg type of str, expected tuple for operation REFINE_TYPE @ offset 0', Verifier.validate_code, code)

    def test_REFINE_TYPE_with_invalid_oparg_index_raises_exception(self):
        if False:
            return 10
        code = self.compile_helper('', bytecodes=[(op.opcode.LOAD_CONST, 0), (op.opcode.REFINE_TYPE, 1)], consts=(3,), stacksize=2)
        self.assertRaisesRegex(VerificationError, 'Argument index 1 out of bounds for size 1 for operation REFINE_TYPE @ offset 2', Verifier.validate_code, code)

    def test_LOAD_FAST_with_valid_oparg_index_is_successful(self):
        if False:
            for i in range(10):
                print('nop')
        code = self.compile_helper('', bytecodes=[(op.opcode.LOAD_FAST, 0)], varnames=('h',), stacksize=1)
        self.assertTrue(Verifier.validate_code(code))

    def test_LOAD_FAST_oparg_type_can_be_any_str(self):
        if False:
            while True:
                i = 10
        code = self.compile_helper('', bytecodes=[(op.opcode.LOAD_FAST, 0)], varnames=(str(),), stacksize=1)
        self.assertTrue(Verifier.validate_code(code))

    def test_LOAD_FAST_with_invalid_oparg_index_raises_exception(self):
        if False:
            i = 10
            return i + 15
        code = self.compile_helper('', bytecodes=[(op.opcode.LOAD_FAST, 0), (op.opcode.LOAD_FAST, 1)], varnames=('h',), stacksize=2)
        self.assertRaisesRegex(VerificationError, 'Argument index 1 out of bounds for size 1 for operation LOAD_FAST @ offset 2', Verifier.validate_code, code)

    def test_STORE_FAST_with_valid_oparg_index_is_successful(self):
        if False:
            while True:
                i = 10
        code = self.compile_helper('', bytecodes=[(op.opcode.LOAD_FAST, 0), (op.opcode.STORE_FAST, 1)], varnames=('h', 'h'), stacksize=2)
        self.assertTrue(Verifier.validate_code(code))

    def test_STORE_FAST_oparg_type_can_be_any_str(self):
        if False:
            while True:
                i = 10
        code = self.compile_helper('', bytecodes=[(op.opcode.LOAD_FAST, 0), (op.opcode.STORE_FAST, 1)], varnames=('h', str()), stacksize=2)
        self.assertTrue(Verifier.validate_code(code))

    def test_STORE_FAST_with_invalid_oparg_index_raises_exception(self):
        if False:
            for i in range(10):
                print('nop')
        code = self.compile_helper('', bytecodes=[(op.opcode.LOAD_FAST, 0), (op.opcode.STORE_FAST, 1)], varnames=('h',), stacksize=2)
        self.assertRaisesRegex(VerificationError, 'Argument index 1 out of bounds for size 1 for operation STORE_FAST @ offset 2', Verifier.validate_code, code)

    def test_DELETE_FAST_with_valid_oparg_index_is_successful(self):
        if False:
            while True:
                i = 10
        code = self.compile_helper('', bytecodes=[(op.opcode.DELETE_FAST, 0)], varnames=('h',), stacksize=1)
        self.assertTrue(Verifier.validate_code(code))

    def test_DELETE_FAST_oparg_type_can_be_any_str(self):
        if False:
            while True:
                i = 10
        code = self.compile_helper('', bytecodes=[(op.opcode.DELETE_FAST, 0)], varnames=(str(),), stacksize=1)
        self.assertTrue(Verifier.validate_code(code))

    def test_DELETE_FAST_with_invalid_oparg_index_raises_exception(self):
        if False:
            i = 10
            return i + 15
        code = self.compile_helper('', bytecodes=[(op.opcode.LOAD_FAST, 0), (op.opcode.DELETE_FAST, 1)], varnames=('h',), stacksize=2)
        self.assertRaisesRegex(VerificationError, 'Argument index 1 out of bounds for size 1 for operation DELETE_FAST @ offset 2', Verifier.validate_code, code)

    def test_LOAD_NAME_with_valid_oparg_index_is_successful(self):
        if False:
            i = 10
            return i + 15
        code = self.compile_helper('', bytecodes=[(op.opcode.LOAD_NAME, 0)], names=('h',), stacksize=1)
        self.assertTrue(Verifier.validate_code(code))

    def test_LOAD_NAME_oparg_type_can_be_any_str(self):
        if False:
            for i in range(10):
                print('nop')
        code = self.compile_helper('', bytecodes=[(op.opcode.LOAD_NAME, 0)], names=(str(),), stacksize=1)
        self.assertTrue(Verifier.validate_code(code))

    def test_LOAD_NAME_with_invalid_oparg_index_raises_exception(self):
        if False:
            print('Hello World!')
        code = self.compile_helper('', bytecodes=[(op.opcode.LOAD_NAME, 0), (op.opcode.LOAD_NAME, 1)], names=('h',), stacksize=2)
        self.assertRaisesRegex(VerificationError, 'Argument index 1 out of bounds for size 1 for operation LOAD_NAME @ offset 2', Verifier.validate_code, code)

    def test_LOAD_GLOBAL_with_valid_oparg_index_is_successful(self):
        if False:
            for i in range(10):
                print('nop')
        code = self.compile_helper('', bytecodes=[(op.opcode.LOAD_GLOBAL, 0)], names=('h',), stacksize=1)
        self.assertTrue(Verifier.validate_code(code))

    def test_LOAD_GLOBAL_oparg_type_can_be_any_str(self):
        if False:
            while True:
                i = 10
        code = self.compile_helper('', bytecodes=[(op.opcode.LOAD_GLOBAL, 0)], names=(str(),), stacksize=1)
        self.assertTrue(Verifier.validate_code(code))

    def test_LOAD_GLOBAL_with_invalid_oparg_index_raises_exception(self):
        if False:
            return 10
        code = self.compile_helper('', bytecodes=[(op.opcode.LOAD_NAME, 0), (op.opcode.LOAD_GLOBAL, 1)], names=('h',), stacksize=2)
        self.assertRaisesRegex(VerificationError, 'Argument index 1 out of bounds for size 1 for operation LOAD_GLOBAL @ offset 2', Verifier.validate_code, code)

    def test_STORE_GLOBAL_with_valid_oparg_index_is_successful(self):
        if False:
            i = 10
            return i + 15
        code = self.compile_helper('', bytecodes=[(op.opcode.LOAD_NAME, 0), (op.opcode.STORE_GLOBAL, 1)], names=('h', 'h'), stacksize=2)
        self.assertTrue(Verifier.validate_code(code))

    def test_STORE_GLOBAL_oparg_type_can_be_any_str(self):
        if False:
            return 10
        code = self.compile_helper('', bytecodes=[(op.opcode.LOAD_NAME, 0), (op.opcode.STORE_GLOBAL, 1)], names=('h', str()), stacksize=2)
        self.assertTrue(Verifier.validate_code(code))

    def test_STORE_GLOBAL_with_invalid_oparg_index_raises_exception(self):
        if False:
            print('Hello World!')
        code = self.compile_helper('', bytecodes=[(op.opcode.LOAD_NAME, 0), (op.opcode.STORE_GLOBAL, 1)], names=('h',), stacksize=2)
        self.assertRaisesRegex(VerificationError, 'Argument index 1 out of bounds for size 1 for operation STORE_GLOBAL @ offset 2', Verifier.validate_code, code)

    def test_DELETE_GLOBAL_with_valid_oparg_index_is_successful(self):
        if False:
            for i in range(10):
                print('nop')
        code = self.compile_helper('', bytecodes=[(op.opcode.DELETE_GLOBAL, 0)], names=('h',), stacksize=1)
        self.assertTrue(Verifier.validate_code(code))

    def test_DELETE_GLOBAL_oparg_type_can_be_any_str(self):
        if False:
            print('Hello World!')
        code = self.compile_helper('', bytecodes=[(op.opcode.DELETE_GLOBAL, 0)], names=(str(),), stacksize=1)
        self.assertTrue(Verifier.validate_code(code))

    def test_DELETE_GLOBAL_with_invalid_oparg_index_raises_exception(self):
        if False:
            print('Hello World!')
        code = self.compile_helper('', bytecodes=[(op.opcode.LOAD_NAME, 0), (op.opcode.DELETE_GLOBAL, 1)], names=('h',), stacksize=2)
        self.assertRaisesRegex(VerificationError, 'Argument index 1 out of bounds for size 1 for operation DELETE_GLOBAL @ offset 2', Verifier.validate_code, code)

    def test_STORE_NAME_with_valid_oparg_index_is_successful(self):
        if False:
            print('Hello World!')
        code = self.compile_helper('', bytecodes=[(op.opcode.LOAD_NAME, 0), (op.opcode.STORE_NAME, 1)], names=('h', 'h'), stacksize=2)
        self.assertTrue(Verifier.validate_code(code))

    def test_STORE_NAME_oparg_type_can_be_any_str(self):
        if False:
            return 10
        code = self.compile_helper('', bytecodes=[(op.opcode.LOAD_NAME, 0), (op.opcode.STORE_NAME, 1)], names=('h', str()), stacksize=2)
        self.assertTrue(Verifier.validate_code(code))

    def test_STORE_NAME_with_invalid_oparg_index_raises_exception(self):
        if False:
            return 10
        code = self.compile_helper('', bytecodes=[(op.opcode.LOAD_NAME, 0), (op.opcode.STORE_NAME, 1)], names=('h',), stacksize=2)
        self.assertRaisesRegex(VerificationError, 'Argument index 1 out of bounds for size 1 for operation STORE_NAME @ offset 2', Verifier.validate_code, code)

    def test_DELETE_NAME_with_valid_oparg_index_is_successful(self):
        if False:
            i = 10
            return i + 15
        code = self.compile_helper('', bytecodes=[(op.opcode.DELETE_NAME, 0)], names=('h',), stacksize=1)
        self.assertTrue(Verifier.validate_code(code))

    def test_DELETE_NAME_oparg_type_can_be_any_str(self):
        if False:
            for i in range(10):
                print('nop')
        code = self.compile_helper('', bytecodes=[(op.opcode.DELETE_NAME, 0)], names=(str(),), stacksize=1)
        self.assertTrue(Verifier.validate_code(code))

    def test_DELETE_NAME_with_invalid_oparg_index_raises_exception(self):
        if False:
            for i in range(10):
                print('nop')
        code = self.compile_helper('', bytecodes=[(op.opcode.LOAD_NAME, 0), (op.opcode.DELETE_NAME, 1)], names=('h',), stacksize=2)
        self.assertRaisesRegex(VerificationError, 'Argument index 1 out of bounds for size 1 for operation DELETE_NAME @ offset 2', Verifier.validate_code, code)

    def test_IMPORT_NAME_with_valid_oparg_index_is_successful(self):
        if False:
            i = 10
            return i + 15
        code = self.compile_helper('', bytecodes=[(op.opcode.LOAD_NAME, 0), (op.opcode.IMPORT_NAME, 1)], names=('h', 'h'), stacksize=2)
        self.assertTrue(Verifier.validate_code(code))

    def test_IMPORT_NAME_oparg_type_can_be_any_str(self):
        if False:
            for i in range(10):
                print('nop')
        code = self.compile_helper('', bytecodes=[(op.opcode.LOAD_NAME, 0), (op.opcode.IMPORT_NAME, 1)], names=('h', str()), stacksize=2)
        self.assertTrue(Verifier.validate_code(code))

    def test_IMPORT_NAME_with_invalid_oparg_index_raises_exception(self):
        if False:
            for i in range(10):
                print('nop')
        code = self.compile_helper('', bytecodes=[(op.opcode.LOAD_NAME, 0), (op.opcode.IMPORT_NAME, 1)], names=('h',), stacksize=2)
        self.assertRaisesRegex(VerificationError, 'Argument index 1 out of bounds for size 1 for operation IMPORT_NAME @ offset 2', Verifier.validate_code, code)

    def test_IMPORT_FROM_with_valid_oparg_index_is_successful(self):
        if False:
            for i in range(10):
                print('nop')
        code = self.compile_helper('', bytecodes=[(op.opcode.IMPORT_FROM, 0)], names=('h',), stacksize=1)
        self.assertTrue(Verifier.validate_code(code))

    def test_IMPORT_FROM_oparg_type_can_be_any_str(self):
        if False:
            while True:
                i = 10
        code = self.compile_helper('', bytecodes=[(op.opcode.IMPORT_FROM, 0)], names=(str(),), stacksize=1)
        self.assertTrue(Verifier.validate_code(code))

    def test_IMPORT_FROM_with_invalid_oparg_index_raises_exception(self):
        if False:
            while True:
                i = 10
        code = self.compile_helper('', bytecodes=[(op.opcode.LOAD_NAME, 0), (op.opcode.IMPORT_FROM, 1)], names=('h',), stacksize=2)
        self.assertRaisesRegex(VerificationError, 'Argument index 1 out of bounds for size 1 for operation IMPORT_FROM @ offset 2', Verifier.validate_code, code)

    def test_STORE_ATTR_with_valid_oparg_index_is_successful(self):
        if False:
            while True:
                i = 10
        code = self.compile_helper('', bytecodes=[(op.opcode.LOAD_NAME, 0), (op.opcode.LOAD_NAME, 1), (op.opcode.STORE_ATTR, 2)], names=('h', 'h', 'h'), stacksize=3)
        self.assertTrue(Verifier.validate_code(code))

    def test_STORE_ATTR_oparg_type_can_be_any_str(self):
        if False:
            return 10
        code = self.compile_helper('', bytecodes=[(op.opcode.LOAD_NAME, 0), (op.opcode.LOAD_NAME, 1), (op.opcode.STORE_ATTR, 2)], names=('h', 'h', str()), stacksize=3)
        self.assertTrue(Verifier.validate_code(code))

    def test_STORE_ATTR_with_invalid_oparg_index_raises_exception(self):
        if False:
            while True:
                i = 10
        code = self.compile_helper('', bytecodes=[(op.opcode.LOAD_NAME, 0), (op.opcode.STORE_ATTR, 1)], names=('h',), stacksize=2)
        self.assertRaisesRegex(VerificationError, 'Argument index 1 out of bounds for size 1 for operation STORE_ATTR @ offset 2', Verifier.validate_code, code)

    def test_LOAD_ATTR_with_valid_oparg_index_is_successful(self):
        if False:
            print('Hello World!')
        code = self.compile_helper('', bytecodes=[(op.opcode.LOAD_ATTR, 0)], names=('h',), stacksize=1)
        self.assertTrue(Verifier.validate_code(code))

    def test_LOAD_ATTR_oparg_type_can_be_any_str(self):
        if False:
            i = 10
            return i + 15
        code = self.compile_helper('', bytecodes=[(op.opcode.LOAD_ATTR, 0)], names=(str(),), stacksize=1)
        self.assertTrue(Verifier.validate_code(code))

    def test_LOAD_ATTR_with_invalid_oparg_index_raises_exception(self):
        if False:
            return 10
        code = self.compile_helper('', bytecodes=[(op.opcode.LOAD_NAME, 0), (op.opcode.LOAD_ATTR, 1)], names=('h',), stacksize=2)
        self.assertRaisesRegex(VerificationError, 'Argument index 1 out of bounds for size 1 for operation LOAD_ATTR @ offset 2', Verifier.validate_code, code)

    def test_DELETE_ATTR_with_valid_oparg_index_is_successful(self):
        if False:
            while True:
                i = 10
        code = self.compile_helper('', bytecodes=[(op.opcode.LOAD_NAME, 0), (op.opcode.DELETE_ATTR, 1)], names=('h', 'h'), stacksize=2)
        self.assertTrue(Verifier.validate_code(code))

    def test_DELETE_ATTR_oparg_type_can_be_any_str(self):
        if False:
            i = 10
            return i + 15
        code = self.compile_helper('', bytecodes=[(op.opcode.LOAD_NAME, 0), (op.opcode.DELETE_ATTR, 1)], names=('h', str()), stacksize=2)
        self.assertTrue(Verifier.validate_code(code))

    def test_DELETE_ATTR_with_invalid_oparg_index_raises_exception(self):
        if False:
            return 10
        code = self.compile_helper('', bytecodes=[(op.opcode.LOAD_NAME, 0), (op.opcode.DELETE_ATTR, 1)], names=('h',), stacksize=2)
        self.assertRaisesRegex(VerificationError, 'Argument index 1 out of bounds for size 1 for operation DELETE_ATTR @ offset 2', Verifier.validate_code, code)

    def test_LOAD_METHOD_with_valid_oparg_index_is_successful(self):
        if False:
            return 10
        code = self.compile_helper('', bytecodes=[(op.opcode.LOAD_METHOD, 0)], names=('h',), stacksize=1)
        self.assertTrue(Verifier.validate_code(code))

    def test_LOAD_METHOD_oparg_type_can_be_any_str(self):
        if False:
            return 10
        code = self.compile_helper('', bytecodes=[(op.opcode.LOAD_METHOD, 0)], names=(str(),), stacksize=1)
        self.assertTrue(Verifier.validate_code(code))

    def test_LOAD_METHOD_with_invalid_oparg_index_raises_exception(self):
        if False:
            while True:
                i = 10
        code = self.compile_helper('', bytecodes=[(op.opcode.LOAD_NAME, 0), (op.opcode.LOAD_METHOD, 1)], names=('h',), stacksize=2)
        self.assertRaisesRegex(VerificationError, 'Argument index 1 out of bounds for size 1 for operation LOAD_METHOD @ offset 2', Verifier.validate_code, code)

    def test_LOAD_DEREF_with_valid_oparg_index_in_freevars_is_successful(self):
        if False:
            print('Hello World!')
        code = self.compile_helper('', bytecodes=[(op.opcode.LOAD_DEREF, 0)], freevars=('h',))
        self.assertTrue(Verifier.validate_code(code))

    def test_LOAD_DEREF_with_valid_oparg_index_in_closure_is_successful(self):
        if False:
            for i in range(10):
                print('nop')
        code = self.compile_helper('', bytecodes=[(op.opcode.LOAD_DEREF, 0)], cellvars=('h',))
        self.assertTrue(Verifier.validate_code(code))

    def test_LOAD_DEREF_oparg_type_can_be_any_str(self):
        if False:
            while True:
                i = 10
        code = self.compile_helper('', bytecodes=[(op.opcode.LOAD_DEREF, 0)], freevars=(str(),), stacksize=1)
        self.assertTrue(Verifier.validate_code(code))

    def test_LOAD_DEREF_with_invalid_oparg_index_raises_exception(self):
        if False:
            for i in range(10):
                print('nop')
        code = self.compile_helper('', bytecodes=[(op.opcode.LOAD_DEREF, 0)])
        self.assertRaisesRegex(VerificationError, 'Argument index 0 out of bounds for size 0 for operation LOAD_DEREF @ offset 0', Verifier.validate_code, code)

    def test_STORE_DEREF_with_valid_oparg_index_in_freevars_is_successful(self):
        if False:
            for i in range(10):
                print('nop')
        code = self.compile_helper('', bytecodes=[(op.opcode.LOAD_DEREF, 0), (op.opcode.STORE_DEREF, 1)], freevars=('h', 'h'))
        self.assertTrue(Verifier.validate_code(code))

    def test_STORE_DEREF_with_valid_oparg_index_in_closure_is_successful(self):
        if False:
            for i in range(10):
                print('nop')
        code = self.compile_helper('', bytecodes=[(op.opcode.LOAD_DEREF, 0), (op.opcode.STORE_DEREF, 1)], cellvars=('h', 'h'))
        self.assertTrue(Verifier.validate_code(code))

    def test_STORE_DEREF_oparg_type_can_be_any_str(self):
        if False:
            print('Hello World!')
        code = self.compile_helper('', bytecodes=[(op.opcode.LOAD_DEREF, 0), (op.opcode.STORE_DEREF, 0)], freevars=(str(), str()), stacksize=1)
        self.assertTrue(Verifier.validate_code(code))

    def test_STORE_DEREF_with_invalid_oparg_index_raises_exception(self):
        if False:
            print('Hello World!')
        code = self.compile_helper('', bytecodes=[(op.opcode.STORE_DEREF, 0)])
        self.assertRaisesRegex(VerificationError, 'Argument index 0 out of bounds for size 0 for operation STORE_DEREF @ offset 0', Verifier.validate_code, code)

    def test_DELETE_DEREF_with_valid_oparg_index_in_freevars_is_successful(self):
        if False:
            i = 10
            return i + 15
        code = self.compile_helper('', bytecodes=[(op.opcode.DELETE_DEREF, 0)], freevars=('h',))
        self.assertTrue(Verifier.validate_code(code))

    def test_DELETE_DEREF_with_valid_oparg_index_in_closure_is_successful(self):
        if False:
            i = 10
            return i + 15
        code = self.compile_helper('', bytecodes=[(op.opcode.DELETE_DEREF, 0)], cellvars=('h',))
        self.assertTrue(Verifier.validate_code(code))

    def test_DELETE_DEREF_oparg_type_can_be_any_str(self):
        if False:
            while True:
                i = 10
        code = self.compile_helper('', bytecodes=[(op.opcode.DELETE_DEREF, 0)], freevars=(str(),), stacksize=1)
        self.assertTrue(Verifier.validate_code(code))

    def test_DELETE_DEREF_with_invalid_oparg_index_raises_exception(self):
        if False:
            i = 10
            return i + 15
        code = self.compile_helper('', bytecodes=[(op.opcode.DELETE_DEREF, 0)])
        self.assertRaisesRegex(VerificationError, 'Argument index 0 out of bounds for size 0 for operation DELETE_DEREF @ offset 0', Verifier.validate_code, code)

    def test_LOAD_CLASSDEREF_with_valid_oparg_index_in_freevars_is_successful(self):
        if False:
            return 10
        code = self.compile_helper('', bytecodes=[(op.opcode.LOAD_CLASSDEREF, 0)], freevars=('h',))
        self.assertTrue(Verifier.validate_code(code))

    def test_LOAD_CLASSDEREF_with_valid_oparg_index_in_closure_is_successful(self):
        if False:
            print('Hello World!')
        code = self.compile_helper('', bytecodes=[(op.opcode.LOAD_CLASSDEREF, 0)], cellvars=('h',))
        self.assertTrue(Verifier.validate_code(code))

    def test_LOAD_CLASSDEREF_oparg_type_can_be_any_str(self):
        if False:
            i = 10
            return i + 15
        code = self.compile_helper('', bytecodes=[(op.opcode.LOAD_CLASSDEREF, 0)], freevars=(str(),), stacksize=1)
        self.assertTrue(Verifier.validate_code(code))

    def test_LOAD_CLASSDEREF_with_invalid_oparg_index_raises_exception(self):
        if False:
            return 10
        code = self.compile_helper('', bytecodes=[(op.opcode.LOAD_CLASSDEREF, 0)])
        self.assertRaisesRegex(VerificationError, 'Argument index 0 out of bounds for size 0 for operation LOAD_CLASSDEREF @ offset 0', Verifier.validate_code, code)

    def test_COMPARE_OP_with_valid_oparg_index_is_successful(self):
        if False:
            i = 10
            return i + 15
        code = self.compile_helper('', bytecodes=[(op.opcode.LOAD_CONST, 0), (op.opcode.COMPARE_OP, 0)])
        self.assertTrue(Verifier.validate_code(code))

    def test_COMPARE_OP_with_invalid_oparg_index_raises_exception(self):
        if False:
            i = 10
            return i + 15
        code = self.compile_helper('', bytecodes=[(op.opcode.LOAD_CONST, 0), (op.opcode.COMPARE_OP, 15)], stacksize=2)
        self.assertRaisesRegex(VerificationError, 'Argument index 15 out of bounds for size 7 for operation COMPARE_OP @ offset 2', Verifier.validate_code, code)

    def test_LOAD_CLOSURE_with_valid_oparg_index_is_successful(self):
        if False:
            return 10
        code = self.compile_helper('', bytecodes=[(op.opcode.LOAD_CLOSURE, 0)], cellvars=('h',))
        self.assertTrue(Verifier.validate_code(code))

    def test_LOAD_CLOSURE_oparg_can_be_any_str(self):
        if False:
            while True:
                i = 10
        code = self.compile_helper('', bytecodes=[(op.opcode.LOAD_CLOSURE, 0)], cellvars=(str(),))
        self.assertTrue(Verifier.validate_code(code))

    def test_LOAD_CLOSURE_with_invalid_oparg_index_raises_exception(self):
        if False:
            i = 10
            return i + 15
        code = self.compile_helper('', bytecodes=[(op.opcode.LOAD_CLOSURE, 0), (op.opcode.LOAD_CLOSURE, 15)], cellvars=('h',), stacksize=2)
        self.assertRaisesRegex(VerificationError, 'Argument index 15 out of bounds for size 1 for operation LOAD_CLOSURE @ offset 2', Verifier.validate_code, code)
if __name__ == '__main__':
    unittest.main()