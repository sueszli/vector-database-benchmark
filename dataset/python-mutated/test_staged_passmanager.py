"""Test the staged passmanager logic"""
from unittest.mock import patch
from ddt import data, ddt
from qiskit.transpiler import PassManager, StagedPassManager
from qiskit.transpiler.exceptions import TranspilerError
from qiskit.transpiler.passes import Optimize1qGates, Unroller, Depth, BasicSwap
from qiskit.test import QiskitTestCase

@ddt
class TestStagedPassManager(QiskitTestCase):

    def test_default_stages(self):
        if False:
            i = 10
            return i + 15
        spm = StagedPassManager()
        self.assertEqual(spm.stages, ('init', 'layout', 'routing', 'translation', 'optimization', 'scheduling'))
        with self.assertWarns(DeprecationWarning):
            spm = StagedPassManager(init=PassManager([Optimize1qGates()]), routing=PassManager([Unroller(['u', 'cx'])]), scheduling=PassManager([Depth()]))
        self.assertEqual([x.__class__.__name__ for passes in spm.passes() for x in passes['passes']], ['Optimize1qGates', 'Unroller', 'Depth'])

    def test_inplace_edit(self):
        if False:
            for i in range(10):
                print('nop')
        spm = StagedPassManager(stages=['single_stage'])
        spm.single_stage = PassManager([Optimize1qGates(), Depth()])
        self.assertEqual([x.__class__.__name__ for passes in spm.passes() for x in passes['passes']], ['Optimize1qGates', 'Depth'])
        with self.assertWarns(DeprecationWarning):
            spm.single_stage.append(Unroller(['u']))
        spm.single_stage.append(Depth())
        self.assertEqual([x.__class__.__name__ for passes in spm.passes() for x in passes['passes']], ['Optimize1qGates', 'Depth', 'Unroller', 'Depth'])

    def test_invalid_stage(self):
        if False:
            for i in range(10):
                print('nop')
        with self.assertRaises(AttributeError):
            StagedPassManager(stages=['init'], translation=PassManager())

    def test_pre_phase_is_valid_stage(self):
        if False:
            print('Hello World!')
        spm = StagedPassManager(stages=['init'], pre_init=PassManager([Depth()]))
        self.assertEqual([x.__class__.__name__ for passes in spm.passes() for x in passes['passes']], ['Depth'])

    def test_append_extend_not_implemented(self):
        if False:
            print('Hello World!')
        spm = StagedPassManager()
        with self.assertRaises(NotImplementedError):
            spm.append(Depth())
        with self.assertRaises(NotImplementedError):
            spm += PassManager()

    def test_invalid_stages(self):
        if False:
            return 10
        invalid_stages = ['two words', 'two-words', 'two+words', 'two&words', '[two_words]', '<two_words>', '{two_words}', '(two_words)', 'two^words', 'two_words!', '^two_words', '@two_words', 'two~words', 'two\\words', 'two/words']
        all_stages = invalid_stages + ['two_words', 'init']
        with self.assertRaises(ValueError) as err:
            StagedPassManager(all_stages)
        message = str(err.exception)
        for stage in invalid_stages:
            self.assertIn(stage, message)

    def test_repeated_stages(self):
        if False:
            while True:
                i = 10
        stages = ['alpha', 'omega', 'alpha']
        with self.assertWarns(DeprecationWarning):
            pre_alpha = PassManager(Unroller(['u', 'cx']))
        alpha = PassManager(Depth())
        post_alpha = PassManager(BasicSwap([[0, 1], [1, 2]]))
        omega = PassManager(Optimize1qGates())
        spm = StagedPassManager(stages, pre_alpha=pre_alpha, alpha=alpha, post_alpha=post_alpha, omega=omega)
        passes = [*pre_alpha.passes(), *alpha.passes(), *post_alpha.passes(), *omega.passes(), *pre_alpha.passes(), *alpha.passes(), *post_alpha.passes()]
        self.assertEqual(spm.passes(), passes)

    def test_edit_stages(self):
        if False:
            for i in range(10):
                print('nop')
        spm = StagedPassManager()
        with self.assertRaises(AttributeError):
            spm.stages = ['init']
        with self.assertRaises(AttributeError):
            spm.expanded_stages = ['init']

    @data(None, ['init'], ['init', 'schedule'])
    def test_expanded_stages(self, stages):
        if False:
            for i in range(10):
                print('nop')
        spm = StagedPassManager(stages=stages)
        expanded_stages = (stage for stage in spm.expanded_stages)
        for stage in spm.stages:
            self.assertEqual(next(expanded_stages), 'pre_' + stage)
            self.assertEqual(next(expanded_stages), stage)
            self.assertEqual(next(expanded_stages), 'post_' + stage)

    def test_setattr(self):
        if False:
            print('Hello World!')
        spm = StagedPassManager()
        with self.assertRaises(TranspilerError):
            spm.init = spm
        mock_target = 'qiskit.transpiler.passmanager.StagedPassManager._update_passmanager'
        with patch(mock_target, spec=True) as mock:
            spm.max_iteration = spm.max_iteration
            mock.assert_not_called()
            spm.init = None
            mock.assert_called_once()

    def test_getitem(self):
        if False:
            while True:
                i = 10
        pm = PassManager(Depth())
        spm = StagedPassManager(init=pm)
        mock_target = 'qiskit.transpiler.passmanager.StagedPassManager._update_passmanager'
        with patch(mock_target, spec=True) as mock:
            _ = spm[0]
            mock.assert_called_once()