import os
import unittest
import shutil
from manticore.ethereum.plugins import VerboseTrace, KeepOnlyIfStorageChanges
from manticore.ethereum import ManticoreEVM
THIS_DIR = os.path.dirname(os.path.abspath(__file__))

class EthPluginsTests(unittest.TestCase):

    def setUp(self):
        if False:
            while True:
                i = 10
        self.mevm = ManticoreEVM()

    def tearDown(self):
        if False:
            i = 10
            return i + 15
        ws = self.mevm.workspace
        del self.mevm
        shutil.rmtree(ws)

    def test_ignore_states(self):
        if False:
            i = 10
            return i + 15
        m = self.mevm
        m.register_plugin(KeepOnlyIfStorageChanges())
        filename = os.path.join(THIS_DIR, 'contracts', 'absurdrepetition.sol')
        with m.kill_timeout():
            m.multi_tx_analysis(filename)
        for st in m.all_states:
            if st.platform.logs:
                return
        self.fail('We did not reach any state with logs')

    @unittest.skip('failing')
    def test_verbose_trace(self):
        if False:
            print('Hello World!')
        source_code = 'contract X {}'
        self.mevm.register_plugin(VerboseTrace())
        owner = self.mevm.create_account(balance=1000, address=1003150779044222824843138452687820282225015987163)
        self.mevm.solidity_create_contract(source_code, owner=owner, gas=90000)
        files = set(os.listdir(self.mevm.workspace))
        with self.assertLogs('manticore.core.manticore', level='INFO') as cm:
            self.mevm.finalize()
            prefix = '\x1b[34mINFO:\x1b[0m:m.c.manticore'
            self.assertEqual(f'{prefix}:Results in {self.mevm.workspace}', cm.output[0])
            self.assertEqual(len(cm.output), 1)
        import re
        files = set((f for f in os.listdir(self.mevm.workspace) if re.match('[^.].*', f)))
        expected_files = {'global_X.runtime_visited', 'global_X_runtime.bytecode', 'test_00000000.verbose_trace', 'global_X.sol', 'global_X.runtime_asm', 'global_X.init_asm', 'global_X.init_visited', 'test_00000000.constraints', 'command.sh', 'global_X_init.bytecode', 'test_00000000.tx', 'test_00000000.pkl', 'manticore.yml', 'global.summary', 'test_00000000.summary', 'test_00000000.tx.json', 'test_00000000.logs', 'test_00000000.trace'}
        self.assertEqual(files, expected_files)
        result_vt_path = os.path.join(self.mevm.workspace, 'test_00000000.verbose_trace')
        expected_vt_path = os.path.join(THIS_DIR, 'data/verbose_trace_plugin_out')
        with open(result_vt_path) as res_fp, open(expected_vt_path) as exp_fp:
            res = res_fp.readlines()
            exp = exp_fp.readlines()
        self.assertEqual(len(res), len(exp))
        self.assertEqual(len(res), 204)
        for i in range(184):
            self.assertEqual(res[i], exp[i], f'Difference on line {i}')
        till = 130
        for i in range(184, 188):
            self.assertEqual(res[i][:till], exp[i][:till], f'Difference on line {i}')
        for i in range(188, 195):
            self.assertEqual(res[i], exp[i], f'Difference on line {i}')
        for i in range(195, 200):
            self.assertEqual(res[i][:till], exp[i][:till], f'Difference on line {i}')
        for i in range(200, len(res)):
            self.assertEqual(res[i], exp[i], f'Difference on line {i}')