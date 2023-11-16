"""
File name is purposefully not test_* to run this test separately.
"""
import inspect
import unittest
import os
import shutil
from manticore.platforms.evm import EVMWorld
from manticore.core.smtlib import operators, ConstraintSet
from manticore.ethereum import ManticoreEVM
from manticore.ethereum.plugins import KeepOnlyIfStorageChanges
from manticore.utils import config
consts = config.get_group('core')
consts.mprocessing = consts.mprocessing.single
THIS_DIR = os.path.dirname(os.path.abspath(__file__))

class EthSha3TestSymbolicate(unittest.TestCase):

    def setUp(self):
        if False:
            print('Hello World!')
        evm_consts = config.get_group('evm')
        evm_consts.sha3 = evm_consts.sha3.symbolicate
        self.mevm = ManticoreEVM()
        self.worksp = self.mevm.workspace

    def tearDown(self):
        if False:
            i = 10
            return i + 15
        self.mevm = None
        shutil.rmtree(self.worksp)

    def ManticoreEVM(self):
        if False:
            for i in range(10):
                print('nop')
        return self.mevm

    def test_example1(self):
        if False:
            i = 10
            return i + 15
        source_code = '\n        contract IsThisVulnerable {\n            event Log(string);\n            function foo(uint input) payable public{\n                if (sha3(input) == 0x12341234){\n                    emit Log("Found a bug");\n                }\n            }\n        }\n        '
        m = self.ManticoreEVM()
        owner = m.create_account(balance=10000000, name='owner')
        attacker = m.create_account(balance=10000000, name='attacker')
        contract = m.solidity_create_contract(source_code, owner=owner, name='contract')
        symbolic_input = m.make_symbolic_value()
        contract.foo(symbolic_input)
        found = 0
        for st in m.all_states:
            if not m.fix_unsound_symbolication(st):
                m.kill_state(st)
                continue
            found += len(st.platform.logs)
        self.assertEqual(found, 0)
        self.assertEqual(m.count_all_states(), 1)

    def test_example2(self):
        if False:
            while True:
                i = 10
        source_code = '\n        contract IsThisVulnerable {\n            event Log(string);\n            function foo(uint x, uint y) payable public{\n                if (x == uint256(sha3(y))){\n                    emit Log("Found a bug");\n                }\n            }\n        }\n        '
        m = self.ManticoreEVM()
        owner = m.create_account(balance=10000000, name='owner')
        attacker = m.create_account(balance=10000000, name='attacker')
        contract = m.solidity_create_contract(source_code, owner=owner, name='contract')
        x = m.make_symbolic_value()
        y = m.make_symbolic_value()
        contract.foo(x, y)
        found = 0
        for st in m.all_states:
            if not m.fix_unsound_symbolication(st):
                m.kill_state(st)
                continue
            found += len(st.platform.logs)
        self.assertEqual(found, 1)
        self.assertEqual(m.count_all_states(), 2)

    def test_example3(self):
        if False:
            i = 10
            return i + 15
        source_code = '\n        contract IsThisVulnerable {\n            event Log(string);\n            function foo(uint x, uint y) payable public{\n                if (sha3(x) == sha3(y)){\n                    emit Log("Found a bug");\n                }\n            }\n        }\n        '
        m = self.ManticoreEVM()
        owner = m.create_account(balance=10000000, name='owner')
        attacker = m.create_account(balance=10000000, name='attacker')
        contract = m.solidity_create_contract(source_code, owner=owner, name='contract')
        x = m.make_symbolic_value()
        y = m.make_symbolic_value()
        contract.foo(x, y)
        found = 0
        for st in m.all_states:
            if not m.fix_unsound_symbolication(st):
                m.kill_state(st)
                continue
            found += len(st.platform.logs)
        self.assertEqual(found, 1)
        self.assertEqual(m.count_all_states(), 2)

    def test_example4(self):
        if False:
            for i in range(10):
                print('nop')
        source_code = '\n        contract IsThisVulnerable {\n            event Log(string);\n            function foo(uint x, uint y) payable public{\n                if (sha3(x) == sha3(y)){\n                    if (x != 10) {\n                        emit Log("Found a bug"); //Reachable \n                    }\n                }\n            }\n        }\n        '
        m = self.ManticoreEVM()
        owner = m.create_account(balance=10000000, name='owner')
        attacker = m.create_account(balance=10000000, name='attacker')
        contract = m.solidity_create_contract(source_code, owner=owner, name='contract')
        x = m.make_symbolic_value()
        y = m.make_symbolic_value()
        contract.foo(x, y)
        found = 0
        for st in m.all_states:
            if not m.fix_unsound_symbolication(st):
                m.kill_state(st)
                continue
            found += len(st.platform.logs)
        self.assertEqual(found, 1)
        self.assertEqual(m.count_all_states(), 3)

    def test_example5(self):
        if False:
            return 10
        source_code = '\n        contract IsThisVulnerable {\n            event Log(string);\n            function foo(uint x, uint y) payable public{\n                if (sha3(x) == sha3(y)){\n                    if (x != 10 && y != 10) {\n                        emit Log("Found a bug"); //Reachable \n                    }\n                }\n            }\n        }\n        '
        m = self.ManticoreEVM()
        owner = m.create_account(balance=10000000, name='owner')
        attacker = m.create_account(balance=10000000, name='attacker')
        contract = m.solidity_create_contract(source_code, owner=owner, name='contract')
        x = m.make_symbolic_value()
        y = m.make_symbolic_value()
        contract.foo(x, y)
        found = 0
        for st in m.all_states:
            if not m.fix_unsound_symbolication(st):
                m.kill_state(st)
                continue
            found += len(st.platform.logs)
        self.assertEqual(found, 1)
        self.assertEqual(m.count_all_states(), 3)

    def test_example6(self):
        if False:
            i = 10
            return i + 15
        source_code = '\n        contract IsThisVulnerable {\n            event Log(string);\n            function foo(uint x, uint y) payable public{\n                if (x == uint256(sha3(y))){\n                    if(y == 10){\n                        emit Log("Found a bug");\n                    }\n                }\n            }\n        }\n        '
        m = self.ManticoreEVM()
        owner = m.create_account(balance=10000000, name='owner')
        attacker = m.create_account(balance=10000000, name='attacker')
        contract = m.solidity_create_contract(source_code, owner=owner, name='contract')
        x = m.make_symbolic_value()
        y = m.make_symbolic_value()
        contract.foo(x, y)
        found = 0
        for st in m.all_states:
            if not m.fix_unsound_symbolication(st):
                m.kill_state(st)
                continue
            found += len(st.platform.logs)
        self.assertEqual(found, 1)
        self.assertEqual(m.count_all_states(), 3)

    def test_example7(self):
        if False:
            for i in range(10):
                print('nop')
        source_code = '\n        contract IsThisVulnerable {\n            event Log(string);\n            function foo(uint x, uint y) payable public{\n                if (sha3(x) == sha3(y)){\n                    if (x == 10) {\n                        emit Log("Found a bug"); //Reachable \n                    }\n                }\n            }\n        }\n        '
        m = self.ManticoreEVM()
        owner = m.create_account(balance=10000000, name='owner')
        attacker = m.create_account(balance=10000000, name='attacker')
        contract = m.solidity_create_contract(source_code, owner=owner, name='contract')
        x = m.make_symbolic_value()
        y = m.make_symbolic_value()
        contract.foo(x, y)
        found = 0
        for st in m.all_states:
            if not m.fix_unsound_symbolication(st):
                m.kill_state(st)
                continue
            found += len(st.platform.logs)
        self.assertEqual(found, 1)
        self.assertEqual(m.count_all_states(), 3)

    def test_example8(self):
        if False:
            i = 10
            return i + 15
        source_code = '\n        contract IsThisVulnerable {\n            event Log(string);\n            function foo(uint x, uint y) payable public{\n                if (sha3(x) == sha3(y)){\n                    if (x == 10) {\n                        emit Log("Found a bug"); //Reachable \n                    }\n                }\n            }\n        }\n        '
        m = self.ManticoreEVM()
        owner = m.create_account(balance=10000000, name='owner')
        attacker = m.create_account(balance=10000000, name='attacker')
        contract = m.solidity_create_contract(source_code, owner=owner, name='contract')
        x = m.make_symbolic_value()
        y = m.make_symbolic_value()
        contract.foo(x, y)
        found = 0
        for st in m.all_states:
            if not m.fix_unsound_symbolication(st):
                m.kill_state(st)
                continue
            found += len(st.platform.logs)
        self.assertEqual(found, 1)
        self.assertEqual(m.count_all_states(), 3)

    def test_essence1(self):
        if False:
            while True:
                i = 10
        source_code = '\n        contract I_Choose_Not_To_Run {\n            event Log(string);\n            function foo(bytes x) public {\n                // x1 keccak\n                if (keccak256("tob") == keccak256(abi.encodePacked(x))){\n                    emit Log("bug");\n                }\n            }\n        }\n        '
        m = self.ManticoreEVM()
        owner = m.create_account(balance=10000000, name='owner')
        attacker = m.create_account(balance=10000000, name='attacker')
        contract = m.solidity_create_contract(source_code, owner=owner, name='contract')
        x = m.make_symbolic_buffer(3)
        contract.foo(x)
        found = 0
        for st in m.all_states:
            if not m.fix_unsound_symbolication(st):
                m.kill_state(st)
                continue
            m.generate_testcase(st)
            found += len(st.platform.logs)
        self.assertEqual(found, 1)
        self.assertEqual(m.count_all_states(), 2)

    def test_essence2(self):
        if False:
            while True:
                i = 10
        source_code = '\n        contract I_Choose_Not_To_Run {\n            event Log(string);\n            function foo(bytes x) public {\n                //# x10 keccak\n//if(keccak256(keccak256(keccak256(keccak256(keccak256(keccak256(keccak256(keccak256(keccak256(keccak256("tob"))))))))))==keccak256(keccak256(keccak256(keccak256(keccak256(keccak256(keccak256(keccak256(keccak256(keccak256(abi.encodePacked(x))))))))))))\nif(keccak256(keccak256(keccak256(keccak256(keccak256(keccak256(keccak256(keccak256(keccak256(keccak256("tob")))))))))) ==  keccak256(keccak256(keccak256(keccak256(keccak256(keccak256(keccak256(keccak256(keccak256(keccak256(abi.encodePacked(x)) ))))))))))\n\n                {\n                    emit Log("bug");\n                }\n            }\n        }\n        '
        m = self.ManticoreEVM()
        owner = m.create_account(balance=10000000, name='owner')
        attacker = m.create_account(balance=10000000, name='attacker')
        contract = m.solidity_create_contract(source_code, owner=owner, name='contract')
        x = m.make_symbolic_buffer(3)
        contract.foo(x)
        found = 0
        for st in m.all_states:
            if not m.fix_unsound_symbolication(st):
                m.kill_state(st)
                continue
            m.generate_testcase(st)
            found += len(st.platform.logs)
        self.assertEqual(found, 1)
        self.assertEqual(m.count_all_states(), 2)

    def test_essence3(self):
        if False:
            for i in range(10):
                print('nop')
        source_code = 'contract Sha3_Multiple_tx{\n            event Log(string);\n            bytes32 val;\n            function foo(uint x) public {\n                if (x == 12345){\n                    val = keccak256(keccak256(uint(6789)));\n                }\n                else{\n                    if (keccak256(val) == keccak256(keccak256(keccak256(x)))){\n                        emit Log("bug");\n                    }\n                }\n            }\n        }\n\n        '
        m = self.ManticoreEVM()
        m.register_plugin(KeepOnlyIfStorageChanges())
        owner = m.create_account(balance=10000000, name='owner')
        attacker = m.create_account(balance=10000000, name='attacker')
        contract = m.solidity_create_contract(source_code, owner=owner, name='contract')
        x1 = m.make_symbolic_value()
        contract.foo(x1)
        x2 = m.make_symbolic_value()
        contract.foo(x2)
        for st in m.all_states:
            if not m.fix_unsound_symbolication(st):
                m.kill_state(st)
                continue
        self.assertEqual(m.count_all_states(), 4)
        found = 0
        for st in m.all_states:
            m.generate_testcase(st)
            found += len(st.platform.logs)
        self.assertEqual(found, 1)

class EthSha3TestConcrete(unittest.TestCase):

    def setUp(self):
        if False:
            while True:
                i = 10
        evm_consts = config.get_group('evm')
        evm_consts.sha3 = evm_consts.sha3.concretize
        self.mevm = ManticoreEVM()
        self.worksp = self.mevm.workspace

    def tearDown(self):
        if False:
            for i in range(10):
                print('nop')
        self.mevm = None
        shutil.rmtree(self.worksp)

    def ManticoreEVM(self):
        if False:
            i = 10
            return i + 15
        return self.mevm

    def test_example_concrete_1(self):
        if False:
            while True:
                i = 10
        source_code = '\n        contract IsThisVulnerable {\n            event Log(string);\n            function foo(uint x, uint y) payable public{\n                if (sha3(x) == sha3(y)){\n                    if (x != 10 && y != 10) {\n                        emit Log("Found a bug"); //Reachable \n                    }\n                }\n            }\n        }\n        '
        m = self.ManticoreEVM()
        owner = m.create_account(balance=10000000, name='owner')
        attacker = m.create_account(balance=10000000, name='attacker')
        contract = m.solidity_create_contract(source_code, owner=owner, name='contract')
        x = m.make_symbolic_value()
        y = m.make_symbolic_value()
        contract.foo(x, y)
        found = 0
        for st in m.all_states:
            if not m.fix_unsound_symbolication(st):
                m.kill_state(st)
                continue
            found += len(st.platform.logs)
        self.assertEqual(found, 1)
        self.assertEqual(m.count_all_states(), 1)

class EthSha3TestFake(EthSha3TestSymbolicate):

    def setUp(self):
        if False:
            return 10
        evm_consts = config.get_group('evm')
        self.saved_sha3 = evm_consts.sha3
        evm_consts.sha3 = evm_consts.sha3.fake
        self.mevm = ManticoreEVM()
        self.worksp = self.mevm.workspace

    def tearDown(self):
        if False:
            print('Hello World!')
        evm_consts = config.get_group('evm')
        evm_consts.sha3 = self.saved_sha3

    def test_example1(self):
        if False:
            for i in range(10):
                print('nop')
        pass

    def test_essence3(self):
        if False:
            return 10
        source_code = 'contract Sha3_Multiple_tx{\n            event Log(string);\n            bytes32 val;\n            function foo(uint x) public {\n                if (x == 12345){\n                    val = keccak256(keccak256(uint(6789)));\n                }\n                else{\n                    if (keccak256(val) == keccak256(keccak256(keccak256(x)))){\n                        emit Log("bug");\n                    }\n                }\n            }\n        }\n\n        '
        m = self.ManticoreEVM()
        m.register_plugin(KeepOnlyIfStorageChanges())
        owner = m.create_account(balance=10000000, name='owner')
        attacker = m.create_account(balance=10000000, name='attacker')
        contract = m.solidity_create_contract(source_code, owner=owner, name='contract')
        x1 = m.make_symbolic_value()
        contract.foo(x1)
        x2 = m.make_symbolic_value()
        contract.foo(x2)
        for st in m.all_states:
            if not m.fix_unsound_symbolication(st):
                m.kill_state(st)
                continue
        self.assertTrue(m.count_all_states() >= 4)
        found = 0
        for st in m.all_states:
            m.generate_testcase(st)
            found += len(st.platform.logs)
        self.assertTrue(found >= 1)
if __name__ == '__main__':
    unittest.main()