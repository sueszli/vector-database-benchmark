"""Transpiler testing"""
import io
from logging import StreamHandler, getLogger
import unittest.mock
import sys
from qiskit import QuantumRegister, QuantumCircuit
from qiskit.transpiler import PassManager, TranspilerError
from qiskit.transpiler.runningpassmanager import DoWhileController, ConditionalController, FlowController
from qiskit.test import QiskitTestCase
from ._dummy_passes import PassA_TP_NR_NP, PassB_TP_RA_PA, PassC_TP_RA_PA, PassD_TP_NR_NP, PassE_AP_NR_NP, PassF_reduce_dag_property, PassJ_Bad_NoReturn, PassK_check_fixed_point_property, PassM_AP_NR_NP

class SchedulerTestCase(QiskitTestCase):
    """Asserts for the scheduler."""

    def assertScheduler(self, circuit, passmanager, expected):
        if False:
            while True:
                i = 10
        '\n        Run `transpile(circuit, passmanager)` and check\n        if the passes run as expected.\n\n        Args:\n            circuit (QuantumCircuit): Circuit to transform via transpilation.\n            passmanager (PassManager): pass manager instance for the transpilation process\n            expected (list): List of things the passes are logging\n        '
        logger = 'LocalLogger'
        with self.assertLogs(logger, level='INFO') as cm:
            out = passmanager.run(circuit)
        self.assertIsInstance(out, QuantumCircuit)
        self.assertEqual([record.message for record in cm.records], expected)

    def assertSchedulerRaises(self, circuit, passmanager, expected, exception_type):
        if False:
            while True:
                i = 10
        '\n        Run `transpile(circuit, passmanager)` and check\n        if the passes run as expected until exception_type is raised.\n\n        Args:\n            circuit (QuantumCircuit): Circuit to transform via transpilation\n            passmanager (PassManager): pass manager instance for the transpilation process\n            expected (list): List of things the passes are logging\n            exception_type (Exception): Exception that is expected to be raised.\n        '
        logger = 'LocalLogger'
        with self.assertLogs(logger, level='INFO') as cm:
            self.assertRaises(exception_type, passmanager.run, circuit)
        self.assertEqual([record.message for record in cm.records], expected)

class TestPassManagerInit(SchedulerTestCase):
    """The pass manager sets things at init time."""

    def test_passes(self):
        if False:
            return 10
        'A single chain of passes, with Requests and Preserves, at __init__ time'
        circuit = QuantumCircuit(QuantumRegister(1))
        passmanager = PassManager(passes=[PassC_TP_RA_PA(), PassB_TP_RA_PA(), PassD_TP_NR_NP(argument1=[1, 2]), PassB_TP_RA_PA()])
        self.assertScheduler(circuit, passmanager, ['run transformation pass PassA_TP_NR_NP', 'run transformation pass PassC_TP_RA_PA', 'run transformation pass PassB_TP_RA_PA', 'run transformation pass PassD_TP_NR_NP', 'argument [1, 2]', 'run transformation pass PassA_TP_NR_NP', 'run transformation pass PassB_TP_RA_PA'])

class TestUseCases(SchedulerTestCase):
    """Combine passes in different ways and checks that passes are run
    in the right order."""

    def setUp(self):
        if False:
            while True:
                i = 10
        super().setUp()
        self.circuit = QuantumCircuit(QuantumRegister(1))
        self.passmanager = PassManager()

    def test_chain(self):
        if False:
            i = 10
            return i + 15
        'A single chain of passes, with Requires and Preserves.'
        self.passmanager.append(PassC_TP_RA_PA())
        self.passmanager.append(PassB_TP_RA_PA())
        self.passmanager.append(PassD_TP_NR_NP(argument1=[1, 2]))
        self.passmanager.append(PassB_TP_RA_PA())
        self.assertScheduler(self.circuit, self.passmanager, ['run transformation pass PassA_TP_NR_NP', 'run transformation pass PassC_TP_RA_PA', 'run transformation pass PassB_TP_RA_PA', 'run transformation pass PassD_TP_NR_NP', 'argument [1, 2]', 'run transformation pass PassA_TP_NR_NP', 'run transformation pass PassB_TP_RA_PA'])

    def test_conditional_passes_true(self):
        if False:
            i = 10
            return i + 15
        'A pass set with a conditional parameter. The callable is True.'
        self.passmanager.append(PassE_AP_NR_NP(True))
        self.passmanager.append(PassA_TP_NR_NP(), condition=lambda property_set: property_set['property'])
        self.assertScheduler(self.circuit, self.passmanager, ['run analysis pass PassE_AP_NR_NP', 'set property as True', 'run transformation pass PassA_TP_NR_NP'])

    def test_conditional_passes_true_fc(self):
        if False:
            return 10
        'A pass set with a conditional parameter (with FlowController). The callable is True.'
        self.passmanager.append(PassE_AP_NR_NP(True))
        self.passmanager.append(ConditionalController([PassA_TP_NR_NP()], condition=lambda property_set: property_set['property']))
        self.assertScheduler(self.circuit, self.passmanager, ['run analysis pass PassE_AP_NR_NP', 'set property as True', 'run transformation pass PassA_TP_NR_NP'])

    def test_conditional_passes_false(self):
        if False:
            for i in range(10):
                print('nop')
        'A pass set with a conditional parameter. The callable is False.'
        self.passmanager.append(PassE_AP_NR_NP(False))
        self.passmanager.append(PassA_TP_NR_NP(), condition=lambda property_set: property_set['property'])
        self.assertScheduler(self.circuit, self.passmanager, ['run analysis pass PassE_AP_NR_NP', 'set property as False'])

    def test_conditional_and_loop(self):
        if False:
            print('Hello World!')
        'Run a conditional first, then a loop.'
        self.passmanager.append(PassE_AP_NR_NP(True))
        self.passmanager.append([PassK_check_fixed_point_property(), PassA_TP_NR_NP(), PassF_reduce_dag_property()], do_while=lambda property_set: not property_set['property_fixed_point'], condition=lambda property_set: property_set['property'])
        self.assertScheduler(self.circuit, self.passmanager, ['run analysis pass PassE_AP_NR_NP', 'set property as True', 'run analysis pass PassG_calculates_dag_property', 'set property as 8 (from dag.property)', 'run analysis pass PassK_check_fixed_point_property', 'run transformation pass PassA_TP_NR_NP', 'run transformation pass PassF_reduce_dag_property', 'dag property = 6', 'run analysis pass PassG_calculates_dag_property', 'set property as 6 (from dag.property)', 'run analysis pass PassK_check_fixed_point_property', 'run transformation pass PassA_TP_NR_NP', 'run transformation pass PassF_reduce_dag_property', 'dag property = 5', 'run analysis pass PassG_calculates_dag_property', 'set property as 5 (from dag.property)', 'run analysis pass PassK_check_fixed_point_property', 'run transformation pass PassA_TP_NR_NP', 'run transformation pass PassF_reduce_dag_property', 'dag property = 4', 'run analysis pass PassG_calculates_dag_property', 'set property as 4 (from dag.property)', 'run analysis pass PassK_check_fixed_point_property', 'run transformation pass PassA_TP_NR_NP', 'run transformation pass PassF_reduce_dag_property', 'dag property = 3', 'run analysis pass PassG_calculates_dag_property', 'set property as 3 (from dag.property)', 'run analysis pass PassK_check_fixed_point_property', 'run transformation pass PassA_TP_NR_NP', 'run transformation pass PassF_reduce_dag_property', 'dag property = 2', 'run analysis pass PassG_calculates_dag_property', 'set property as 2 (from dag.property)', 'run analysis pass PassK_check_fixed_point_property', 'run transformation pass PassA_TP_NR_NP', 'run transformation pass PassF_reduce_dag_property', 'dag property = 2', 'run analysis pass PassG_calculates_dag_property', 'set property as 2 (from dag.property)', 'run analysis pass PassK_check_fixed_point_property', 'run transformation pass PassA_TP_NR_NP', 'run transformation pass PassF_reduce_dag_property', 'dag property = 2'])

    def test_loop_and_conditional(self):
        if False:
            while True:
                i = 10
        'Run a loop first, then a conditional.'
        with self.assertWarns(DeprecationWarning):
            FlowController.remove_flow_controller('condition')
            FlowController.add_flow_controller('condition', ConditionalController)
        self.passmanager.append(PassK_check_fixed_point_property())
        self.passmanager.append([PassK_check_fixed_point_property(), PassA_TP_NR_NP(), PassF_reduce_dag_property()], do_while=lambda property_set: not property_set['property_fixed_point'], condition=lambda property_set: not property_set['property_fixed_point'])
        self.assertScheduler(self.circuit, self.passmanager, ['run analysis pass PassG_calculates_dag_property', 'set property as 8 (from dag.property)', 'run analysis pass PassK_check_fixed_point_property', 'run transformation pass PassA_TP_NR_NP', 'run transformation pass PassF_reduce_dag_property', 'dag property = 6', 'run analysis pass PassG_calculates_dag_property', 'set property as 6 (from dag.property)', 'run analysis pass PassK_check_fixed_point_property', 'run transformation pass PassA_TP_NR_NP', 'run transformation pass PassF_reduce_dag_property', 'dag property = 5', 'run analysis pass PassG_calculates_dag_property', 'set property as 5 (from dag.property)', 'run analysis pass PassK_check_fixed_point_property', 'run transformation pass PassA_TP_NR_NP', 'run transformation pass PassF_reduce_dag_property', 'dag property = 4', 'run analysis pass PassG_calculates_dag_property', 'set property as 4 (from dag.property)', 'run analysis pass PassK_check_fixed_point_property', 'run transformation pass PassA_TP_NR_NP', 'run transformation pass PassF_reduce_dag_property', 'dag property = 3', 'run analysis pass PassG_calculates_dag_property', 'set property as 3 (from dag.property)', 'run analysis pass PassK_check_fixed_point_property', 'run transformation pass PassA_TP_NR_NP', 'run transformation pass PassF_reduce_dag_property', 'dag property = 2', 'run analysis pass PassG_calculates_dag_property', 'set property as 2 (from dag.property)', 'run analysis pass PassK_check_fixed_point_property', 'run transformation pass PassA_TP_NR_NP', 'run transformation pass PassF_reduce_dag_property', 'dag property = 2', 'run analysis pass PassG_calculates_dag_property', 'set property as 2 (from dag.property)', 'run analysis pass PassK_check_fixed_point_property', 'run transformation pass PassA_TP_NR_NP', 'run transformation pass PassF_reduce_dag_property', 'dag property = 2'])

    def test_do_not_repeat_based_on_preservation(self):
        if False:
            for i in range(10):
                print('nop')
        'When a pass is still a valid pass (because the following passes\n        preserved it), it should not run again.'
        self.passmanager.append([PassB_TP_RA_PA(), PassA_TP_NR_NP(), PassB_TP_RA_PA()])
        self.assertScheduler(self.circuit, self.passmanager, ['run transformation pass PassA_TP_NR_NP', 'run transformation pass PassB_TP_RA_PA'])

    def test_do_not_repeat_based_on_idempotence(self):
        if False:
            while True:
                i = 10
        'Repetition can be optimized to a single execution when\n        the pass is idempotent.'
        self.passmanager.append(PassA_TP_NR_NP())
        self.passmanager.append([PassA_TP_NR_NP(), PassA_TP_NR_NP()])
        self.passmanager.append(PassA_TP_NR_NP())
        self.assertScheduler(self.circuit, self.passmanager, ['run transformation pass PassA_TP_NR_NP'])

    def test_non_idempotent_pass(self):
        if False:
            for i in range(10):
                print('nop')
        'Two or more runs of a non-idempotent pass cannot be optimized.'
        self.passmanager.append(PassF_reduce_dag_property())
        self.passmanager.append([PassF_reduce_dag_property(), PassF_reduce_dag_property()])
        self.passmanager.append(PassF_reduce_dag_property())
        self.assertScheduler(self.circuit, self.passmanager, ['run transformation pass PassF_reduce_dag_property', 'dag property = 6', 'run transformation pass PassF_reduce_dag_property', 'dag property = 5', 'run transformation pass PassF_reduce_dag_property', 'dag property = 4', 'run transformation pass PassF_reduce_dag_property', 'dag property = 3'])

    def test_analysis_pass_is_idempotent(self):
        if False:
            print('Hello World!')
        'Analysis passes are idempotent.'
        passmanager = PassManager()
        passmanager.append(PassE_AP_NR_NP(argument1=1))
        passmanager.append(PassE_AP_NR_NP(argument1=1))
        self.assertScheduler(self.circuit, passmanager, ['run analysis pass PassE_AP_NR_NP', 'set property as 1'])

    def test_ap_before_and_after_a_tp(self):
        if False:
            print('Hello World!')
        'A default transformation does not preserves anything\n        and analysis passes need to be re-run'
        passmanager = PassManager()
        passmanager.append(PassE_AP_NR_NP(argument1=1))
        passmanager.append(PassA_TP_NR_NP())
        passmanager.append(PassE_AP_NR_NP(argument1=1))
        self.assertScheduler(self.circuit, passmanager, ['run analysis pass PassE_AP_NR_NP', 'set property as 1', 'run transformation pass PassA_TP_NR_NP', 'run analysis pass PassE_AP_NR_NP', 'set property as 1'])

    def test_pass_no_return(self):
        if False:
            print('Hello World!')
        "Transformation passes that don't return a DAG raise error."
        self.passmanager.append(PassJ_Bad_NoReturn())
        self.assertSchedulerRaises(self.circuit, self.passmanager, ['run transformation pass PassJ_Bad_NoReturn'], TranspilerError)

    def test_fixed_point_pass(self):
        if False:
            for i in range(10):
                print('nop')
        'A pass set with a do_while parameter that checks for a fixed point.'
        self.passmanager.append([PassK_check_fixed_point_property(), PassA_TP_NR_NP(), PassF_reduce_dag_property()], do_while=lambda property_set: not property_set['property_fixed_point'])
        self.assertScheduler(self.circuit, self.passmanager, ['run analysis pass PassG_calculates_dag_property', 'set property as 8 (from dag.property)', 'run analysis pass PassK_check_fixed_point_property', 'run transformation pass PassA_TP_NR_NP', 'run transformation pass PassF_reduce_dag_property', 'dag property = 6', 'run analysis pass PassG_calculates_dag_property', 'set property as 6 (from dag.property)', 'run analysis pass PassK_check_fixed_point_property', 'run transformation pass PassA_TP_NR_NP', 'run transformation pass PassF_reduce_dag_property', 'dag property = 5', 'run analysis pass PassG_calculates_dag_property', 'set property as 5 (from dag.property)', 'run analysis pass PassK_check_fixed_point_property', 'run transformation pass PassA_TP_NR_NP', 'run transformation pass PassF_reduce_dag_property', 'dag property = 4', 'run analysis pass PassG_calculates_dag_property', 'set property as 4 (from dag.property)', 'run analysis pass PassK_check_fixed_point_property', 'run transformation pass PassA_TP_NR_NP', 'run transformation pass PassF_reduce_dag_property', 'dag property = 3', 'run analysis pass PassG_calculates_dag_property', 'set property as 3 (from dag.property)', 'run analysis pass PassK_check_fixed_point_property', 'run transformation pass PassA_TP_NR_NP', 'run transformation pass PassF_reduce_dag_property', 'dag property = 2', 'run analysis pass PassG_calculates_dag_property', 'set property as 2 (from dag.property)', 'run analysis pass PassK_check_fixed_point_property', 'run transformation pass PassA_TP_NR_NP', 'run transformation pass PassF_reduce_dag_property', 'dag property = 2', 'run analysis pass PassG_calculates_dag_property', 'set property as 2 (from dag.property)', 'run analysis pass PassK_check_fixed_point_property', 'run transformation pass PassA_TP_NR_NP', 'run transformation pass PassF_reduce_dag_property', 'dag property = 2'])

    def test_fixed_point_fc(self):
        if False:
            return 10
        'A fixed point scheduler with flow control.'
        self.passmanager.append(DoWhileController([PassK_check_fixed_point_property(), PassA_TP_NR_NP(), PassF_reduce_dag_property()], do_while=lambda property_set: not property_set['property_fixed_point']))
        expected = ['run analysis pass PassG_calculates_dag_property', 'set property as 8 (from dag.property)', 'run analysis pass PassK_check_fixed_point_property', 'run transformation pass PassA_TP_NR_NP', 'run transformation pass PassF_reduce_dag_property', 'dag property = 6', 'run analysis pass PassG_calculates_dag_property', 'set property as 6 (from dag.property)', 'run analysis pass PassK_check_fixed_point_property', 'run transformation pass PassA_TP_NR_NP', 'run transformation pass PassF_reduce_dag_property', 'dag property = 5', 'run analysis pass PassG_calculates_dag_property', 'set property as 5 (from dag.property)', 'run analysis pass PassK_check_fixed_point_property', 'run transformation pass PassA_TP_NR_NP', 'run transformation pass PassF_reduce_dag_property', 'dag property = 4', 'run analysis pass PassG_calculates_dag_property', 'set property as 4 (from dag.property)', 'run analysis pass PassK_check_fixed_point_property', 'run transformation pass PassA_TP_NR_NP', 'run transformation pass PassF_reduce_dag_property', 'dag property = 3', 'run analysis pass PassG_calculates_dag_property', 'set property as 3 (from dag.property)', 'run analysis pass PassK_check_fixed_point_property', 'run transformation pass PassA_TP_NR_NP', 'run transformation pass PassF_reduce_dag_property', 'dag property = 2', 'run analysis pass PassG_calculates_dag_property', 'set property as 2 (from dag.property)', 'run analysis pass PassK_check_fixed_point_property', 'run transformation pass PassA_TP_NR_NP', 'run transformation pass PassF_reduce_dag_property', 'dag property = 2', 'run analysis pass PassG_calculates_dag_property', 'set property as 2 (from dag.property)', 'run analysis pass PassK_check_fixed_point_property', 'run transformation pass PassA_TP_NR_NP', 'run transformation pass PassF_reduce_dag_property', 'dag property = 2']
        self.assertScheduler(self.circuit, self.passmanager, expected)

    def test_fixed_point_pass_max_iteration(self):
        if False:
            while True:
                i = 10
        'A pass set with a do_while parameter that checks that\n        the max_iteration is raised.'
        self.passmanager.append([PassK_check_fixed_point_property(), PassA_TP_NR_NP(), PassF_reduce_dag_property()], do_while=lambda property_set: not property_set['property_fixed_point'], max_iteration=2)
        self.assertSchedulerRaises(self.circuit, self.passmanager, ['run analysis pass PassG_calculates_dag_property', 'set property as 8 (from dag.property)', 'run analysis pass PassK_check_fixed_point_property', 'run transformation pass PassA_TP_NR_NP', 'run transformation pass PassF_reduce_dag_property', 'dag property = 6', 'run analysis pass PassG_calculates_dag_property', 'set property as 6 (from dag.property)', 'run analysis pass PassK_check_fixed_point_property', 'run transformation pass PassA_TP_NR_NP', 'run transformation pass PassF_reduce_dag_property', 'dag property = 5'], TranspilerError)

    def test_fresh_initial_state(self):
        if False:
            while True:
                i = 10
        'New construction gives fresh instance.'
        self.passmanager.append(PassM_AP_NR_NP(argument1=1))
        self.passmanager.append(PassA_TP_NR_NP())
        self.passmanager.append(PassM_AP_NR_NP(argument1=1))
        self.assertScheduler(self.circuit, self.passmanager, ['run analysis pass PassM_AP_NR_NP', 'self.argument1 = 2', 'run transformation pass PassA_TP_NR_NP', 'run analysis pass PassM_AP_NR_NP', 'self.argument1 = 2'])

    def test_nested_conditional_in_loop(self):
        if False:
            i = 10
            return i + 15
        'Run a loop with a nested conditional.'
        nested_conditional = [ConditionalController([PassA_TP_NR_NP()], condition=lambda property_set: property_set['property'] >= 5)]
        self.passmanager.append([PassK_check_fixed_point_property()] + nested_conditional + [PassF_reduce_dag_property()], do_while=lambda property_set: not property_set['property_fixed_point'])
        expected = ['run analysis pass PassG_calculates_dag_property', 'set property as 8 (from dag.property)', 'run analysis pass PassK_check_fixed_point_property', 'run transformation pass PassA_TP_NR_NP', 'run transformation pass PassF_reduce_dag_property', 'dag property = 6', 'run analysis pass PassG_calculates_dag_property', 'set property as 6 (from dag.property)', 'run analysis pass PassK_check_fixed_point_property', 'run transformation pass PassA_TP_NR_NP', 'run transformation pass PassF_reduce_dag_property', 'dag property = 5', 'run analysis pass PassG_calculates_dag_property', 'set property as 5 (from dag.property)', 'run analysis pass PassK_check_fixed_point_property', 'run transformation pass PassA_TP_NR_NP', 'run transformation pass PassF_reduce_dag_property', 'dag property = 4', 'run analysis pass PassG_calculates_dag_property', 'set property as 4 (from dag.property)', 'run analysis pass PassK_check_fixed_point_property', 'run transformation pass PassF_reduce_dag_property', 'dag property = 3', 'run analysis pass PassG_calculates_dag_property', 'set property as 3 (from dag.property)', 'run analysis pass PassK_check_fixed_point_property', 'run transformation pass PassF_reduce_dag_property', 'dag property = 2', 'run analysis pass PassG_calculates_dag_property', 'set property as 2 (from dag.property)', 'run analysis pass PassK_check_fixed_point_property', 'run transformation pass PassF_reduce_dag_property', 'dag property = 2', 'run analysis pass PassG_calculates_dag_property', 'set property as 2 (from dag.property)', 'run analysis pass PassK_check_fixed_point_property', 'run transformation pass PassF_reduce_dag_property', 'dag property = 2']
        self.assertScheduler(self.circuit, self.passmanager, expected)

class DoXTimesController(FlowController):
    """A control-flow plugin for running a set of passes an X amount of times."""

    def __init__(self, passes, options, do_x_times, **_):
        if False:
            while True:
                i = 10
        super().__init__(options)
        self.passes = passes
        self.do_x_times = do_x_times

    def iter_tasks(self, metadata):
        if False:
            return 10
        for _ in range(self.do_x_times(metadata.property_set)):
            for pass_ in self.passes:
                metadata = (yield pass_)

class TestControlFlowPlugin(SchedulerTestCase):
    """Testing the control flow plugin system."""

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        super().setUp()
        self.passmanager = PassManager()
        self.circuit = QuantumCircuit(QuantumRegister(1))

    def test_control_flow_plugin(self):
        if False:
            i = 10
            return i + 15
        'Adds a control flow plugin with a single parameter and runs it.'
        with self.assertWarns(DeprecationWarning):
            FlowController.add_flow_controller('do_x_times', DoXTimesController)
        self.passmanager.append([PassB_TP_RA_PA(), PassC_TP_RA_PA()], do_x_times=lambda x: 3)
        self.assertScheduler(self.circuit, self.passmanager, ['run transformation pass PassA_TP_NR_NP', 'run transformation pass PassB_TP_RA_PA', 'run transformation pass PassC_TP_RA_PA', 'run transformation pass PassB_TP_RA_PA', 'run transformation pass PassC_TP_RA_PA', 'run transformation pass PassB_TP_RA_PA', 'run transformation pass PassC_TP_RA_PA'])

    def test_callable_control_flow_plugin(self):
        if False:
            return 10
        'Removes do_while, then adds it back. Checks max_iteration still working.'
        controllers_length = len(FlowController.registered_controllers)
        with self.assertWarns(DeprecationWarning):
            FlowController.remove_flow_controller('do_while')
        self.assertEqual(controllers_length - 1, len(FlowController.registered_controllers))
        with self.assertWarns(DeprecationWarning):
            FlowController.add_flow_controller('do_while', DoWhileController)
        self.assertEqual(controllers_length, len(FlowController.registered_controllers))
        self.passmanager.append([PassB_TP_RA_PA(), PassC_TP_RA_PA()], do_while=lambda property_set: True, max_iteration=2)
        self.assertSchedulerRaises(self.circuit, self.passmanager, ['run transformation pass PassA_TP_NR_NP', 'run transformation pass PassB_TP_RA_PA', 'run transformation pass PassC_TP_RA_PA', 'run transformation pass PassB_TP_RA_PA', 'run transformation pass PassC_TP_RA_PA'], TranspilerError)

    def test_remove_nonexistent_plugin(self):
        if False:
            while True:
                i = 10
        'Tries to remove a plugin that does not exist.'
        with self.assertRaises(KeyError):
            with self.assertWarns(DeprecationWarning):
                FlowController.remove_flow_controller('foo')

class TestDumpPasses(SchedulerTestCase):
    """Testing the passes method."""

    def test_passes(self):
        if False:
            print('Hello World!')
        'Dump passes in different FlowControllerLinear'
        passmanager = PassManager()
        passmanager.append(PassC_TP_RA_PA())
        passmanager.append(PassB_TP_RA_PA())
        expected = [{'flow_controllers': {}, 'passes': [PassC_TP_RA_PA()]}, {'flow_controllers': {}, 'passes': [PassB_TP_RA_PA()]}]
        self.assertEqual(expected, passmanager.passes())

    def test_passes_in_linear(self):
        if False:
            print('Hello World!')
        'Dump passes in the same FlowControllerLinear'
        passmanager = PassManager(passes=[PassC_TP_RA_PA(), PassB_TP_RA_PA(), PassD_TP_NR_NP(argument1=[1, 2]), PassB_TP_RA_PA()])
        expected = [{'flow_controllers': {}, 'passes': [PassC_TP_RA_PA(), PassB_TP_RA_PA(), PassD_TP_NR_NP(argument1=[1, 2]), PassB_TP_RA_PA()]}]
        self.assertEqual(expected, passmanager.passes())

    def test_control_flow_plugin(self):
        if False:
            print('Hello World!')
        'Dump passes in a custom flow controller.'
        passmanager = PassManager()
        with self.assertWarns(DeprecationWarning):
            FlowController.add_flow_controller('do_x_times', DoXTimesController)
        passmanager.append([PassB_TP_RA_PA(), PassC_TP_RA_PA()], do_x_times=lambda x: 3)
        expected = [{'passes': [PassB_TP_RA_PA(), PassC_TP_RA_PA()], 'flow_controllers': {'do_x_times'}}]
        self.assertEqual(expected, passmanager.passes())

    def test_conditional_and_loop(self):
        if False:
            for i in range(10):
                print('nop')
        'Dump passes with a conditional and a loop.'
        passmanager = PassManager()
        passmanager.append(PassE_AP_NR_NP(True))
        passmanager.append([PassK_check_fixed_point_property(), PassA_TP_NR_NP(), PassF_reduce_dag_property()], do_while=lambda property_set: not property_set['property_fixed_point'], condition=lambda property_set: property_set['property_fixed_point'])
        expected = [{'passes': [PassE_AP_NR_NP(True)], 'flow_controllers': {}}, {'passes': [PassK_check_fixed_point_property(), PassA_TP_NR_NP(), PassF_reduce_dag_property()], 'flow_controllers': {'condition', 'do_while'}}]
        self.assertEqual(expected, passmanager.passes())

class StreamHandlerRaiseException(StreamHandler):
    """Handler class that will raise an exception on formatting errors."""

    def handleError(self, record):
        if False:
            i = 10
            return i + 15
        raise sys.exc_info()

class TestLogPasses(QiskitTestCase):
    """Testing the log_passes option."""

    def setUp(self):
        if False:
            return 10
        super().setUp()
        logger = getLogger()
        self.addCleanup(logger.setLevel, logger.level)
        logger.setLevel('DEBUG')
        self.output = io.StringIO()
        logger.addHandler(StreamHandlerRaiseException(self.output))
        self.circuit = QuantumCircuit(QuantumRegister(1))

    def assertPassLog(self, passmanager, list_of_passes):
        if False:
            for i in range(10):
                print('nop')
        "Runs the passmanager and checks that the elements in\n        passmanager.property_set['pass_log'] match list_of_passes (the names)."
        passmanager.run(self.circuit)
        self.output.seek(0)
        output_lines = self.output.readlines()
        pass_log_lines = [x for x in output_lines if x.startswith('Pass:')]
        for (index, pass_name) in enumerate(list_of_passes):
            self.assertTrue(pass_log_lines[index].startswith('Pass: %s -' % pass_name))

    def test_passes(self):
        if False:
            for i in range(10):
                print('nop')
        'Dump passes in different FlowControllerLinear'
        passmanager = PassManager()
        passmanager.append(PassC_TP_RA_PA())
        passmanager.append(PassB_TP_RA_PA())
        self.assertPassLog(passmanager, ['PassA_TP_NR_NP', 'PassC_TP_RA_PA', 'PassB_TP_RA_PA'])

    def test_passes_in_linear(self):
        if False:
            while True:
                i = 10
        'Dump passes in the same FlowControllerLinear'
        passmanager = PassManager(passes=[PassC_TP_RA_PA(), PassB_TP_RA_PA(), PassD_TP_NR_NP(argument1=[1, 2]), PassB_TP_RA_PA()])
        self.assertPassLog(passmanager, ['PassA_TP_NR_NP', 'PassC_TP_RA_PA', 'PassB_TP_RA_PA', 'PassD_TP_NR_NP', 'PassA_TP_NR_NP', 'PassB_TP_RA_PA'])

    def test_control_flow_plugin(self):
        if False:
            return 10
        'Dump passes in a custom flow controller.'
        passmanager = PassManager()
        with self.assertWarns(DeprecationWarning):
            FlowController.add_flow_controller('do_x_times', DoXTimesController)
        passmanager.append([PassB_TP_RA_PA(), PassC_TP_RA_PA()], do_x_times=lambda x: 3)
        self.assertPassLog(passmanager, ['PassA_TP_NR_NP', 'PassB_TP_RA_PA', 'PassC_TP_RA_PA', 'PassB_TP_RA_PA', 'PassC_TP_RA_PA', 'PassB_TP_RA_PA', 'PassC_TP_RA_PA'])

    def test_conditional_and_loop(self):
        if False:
            return 10
        'Dump passes with a conditional and a loop'
        passmanager = PassManager()
        passmanager.append(PassE_AP_NR_NP(True))
        passmanager.append([PassK_check_fixed_point_property(), PassA_TP_NR_NP(), PassF_reduce_dag_property()], do_while=lambda property_set: not property_set['property_fixed_point'], condition=lambda property_set: property_set['property_fixed_point'])
        self.assertPassLog(passmanager, ['PassE_AP_NR_NP'])

class TestPassManagerReuse(SchedulerTestCase):
    """The PassManager instance should be reusable."""

    def setUp(self):
        if False:
            while True:
                i = 10
        super().setUp()
        self.passmanager = PassManager()
        self.circuit = QuantumCircuit(QuantumRegister(1))

    def test_chain_twice(self):
        if False:
            i = 10
            return i + 15
        'Run a chain twice.'
        self.passmanager.append(PassC_TP_RA_PA())
        self.passmanager.append(PassB_TP_RA_PA())
        expected = ['run transformation pass PassA_TP_NR_NP', 'run transformation pass PassC_TP_RA_PA', 'run transformation pass PassB_TP_RA_PA']
        self.assertScheduler(self.circuit, self.passmanager, expected)
        self.assertScheduler(self.circuit, self.passmanager, expected)

    def test_conditional_twice(self):
        if False:
            return 10
        'Run a conditional twice.'
        self.passmanager.append(PassE_AP_NR_NP(True))
        self.passmanager.append(PassA_TP_NR_NP(), condition=lambda property_set: property_set['property'])
        expected = ['run analysis pass PassE_AP_NR_NP', 'set property as True', 'run transformation pass PassA_TP_NR_NP']
        self.assertScheduler(self.circuit, self.passmanager, expected)
        self.assertScheduler(self.circuit, self.passmanager, expected)

    def test_fixed_point_twice(self):
        if False:
            return 10
        'A fixed point scheduler, twice.'
        self.passmanager.append([PassK_check_fixed_point_property(), PassA_TP_NR_NP(), PassF_reduce_dag_property()], do_while=lambda property_set: not property_set['property_fixed_point'])
        expected = ['run analysis pass PassG_calculates_dag_property', 'set property as 8 (from dag.property)', 'run analysis pass PassK_check_fixed_point_property', 'run transformation pass PassA_TP_NR_NP', 'run transformation pass PassF_reduce_dag_property', 'dag property = 6', 'run analysis pass PassG_calculates_dag_property', 'set property as 6 (from dag.property)', 'run analysis pass PassK_check_fixed_point_property', 'run transformation pass PassA_TP_NR_NP', 'run transformation pass PassF_reduce_dag_property', 'dag property = 5', 'run analysis pass PassG_calculates_dag_property', 'set property as 5 (from dag.property)', 'run analysis pass PassK_check_fixed_point_property', 'run transformation pass PassA_TP_NR_NP', 'run transformation pass PassF_reduce_dag_property', 'dag property = 4', 'run analysis pass PassG_calculates_dag_property', 'set property as 4 (from dag.property)', 'run analysis pass PassK_check_fixed_point_property', 'run transformation pass PassA_TP_NR_NP', 'run transformation pass PassF_reduce_dag_property', 'dag property = 3', 'run analysis pass PassG_calculates_dag_property', 'set property as 3 (from dag.property)', 'run analysis pass PassK_check_fixed_point_property', 'run transformation pass PassA_TP_NR_NP', 'run transformation pass PassF_reduce_dag_property', 'dag property = 2', 'run analysis pass PassG_calculates_dag_property', 'set property as 2 (from dag.property)', 'run analysis pass PassK_check_fixed_point_property', 'run transformation pass PassA_TP_NR_NP', 'run transformation pass PassF_reduce_dag_property', 'dag property = 2', 'run analysis pass PassG_calculates_dag_property', 'set property as 2 (from dag.property)', 'run analysis pass PassK_check_fixed_point_property', 'run transformation pass PassA_TP_NR_NP', 'run transformation pass PassF_reduce_dag_property', 'dag property = 2']
        self.assertScheduler(self.circuit, self.passmanager, expected)
        self.assertScheduler(self.circuit, self.passmanager, expected)

class TestPassManagerChanges(SchedulerTestCase):
    """Test PassManager manipulation with changes"""

    def setUp(self):
        if False:
            i = 10
            return i + 15
        super().setUp()
        self.passmanager = PassManager()
        self.circuit = QuantumCircuit(QuantumRegister(1))

    def test_replace0(self):
        if False:
            for i in range(10):
                print('nop')
        'Test passmanager.replace(0, ...).'
        self.passmanager.append(PassC_TP_RA_PA())
        self.passmanager.append(PassB_TP_RA_PA())
        self.passmanager.replace(0, PassB_TP_RA_PA())
        expected = ['run transformation pass PassA_TP_NR_NP', 'run transformation pass PassB_TP_RA_PA']
        self.assertScheduler(self.circuit, self.passmanager, expected)

    def test_replace1(self):
        if False:
            for i in range(10):
                print('nop')
        'Test passmanager.replace(1, ...).'
        self.passmanager.append(PassC_TP_RA_PA())
        self.passmanager.append(PassB_TP_RA_PA())
        self.passmanager.replace(1, PassC_TP_RA_PA())
        expected = ['run transformation pass PassA_TP_NR_NP', 'run transformation pass PassC_TP_RA_PA']
        self.assertScheduler(self.circuit, self.passmanager, expected)

    def test_remove0(self):
        if False:
            return 10
        'Test passmanager.remove(0).'
        self.passmanager.append(PassC_TP_RA_PA())
        self.passmanager.append(PassB_TP_RA_PA())
        self.passmanager.remove(0)
        expected = ['run transformation pass PassA_TP_NR_NP', 'run transformation pass PassB_TP_RA_PA']
        self.assertScheduler(self.circuit, self.passmanager, expected)

    def test_remove1(self):
        if False:
            while True:
                i = 10
        'Test passmanager.remove(1).'
        self.passmanager.append(PassC_TP_RA_PA())
        self.passmanager.append(PassB_TP_RA_PA())
        self.passmanager.remove(1)
        expected = ['run transformation pass PassA_TP_NR_NP', 'run transformation pass PassC_TP_RA_PA']
        self.assertScheduler(self.circuit, self.passmanager, expected)

    def test_remove_minus_1(self):
        if False:
            i = 10
            return i + 15
        'Test passmanager.remove(-1).'
        self.passmanager.append(PassA_TP_NR_NP())
        self.passmanager.append(PassB_TP_RA_PA())
        self.passmanager.remove(-1)
        expected = ['run transformation pass PassA_TP_NR_NP']
        self.assertScheduler(self.circuit, self.passmanager, expected)

    def test_setitem(self):
        if False:
            print('Hello World!')
        'Test passmanager[1] = ...'
        self.passmanager.append(PassC_TP_RA_PA())
        self.passmanager.append(PassB_TP_RA_PA())
        self.passmanager[1] = PassC_TP_RA_PA()
        expected = ['run transformation pass PassA_TP_NR_NP', 'run transformation pass PassC_TP_RA_PA']
        self.assertScheduler(self.circuit, self.passmanager, expected)

    def test_replace_with_conditional(self):
        if False:
            i = 10
            return i + 15
        'Replace a pass with a conditional pass.'
        self.passmanager.append(PassE_AP_NR_NP(False))
        self.passmanager.append(PassB_TP_RA_PA())
        self.passmanager.replace(1, PassA_TP_NR_NP(), condition=lambda property_set: property_set['property'])
        expected = ['run analysis pass PassE_AP_NR_NP', 'set property as False']
        self.assertScheduler(self.circuit, self.passmanager, expected)

    def test_replace_error(self):
        if False:
            for i in range(10):
                print('nop')
        'Replace a non-existing index.'
        self.passmanager.append(PassB_TP_RA_PA())
        with self.assertRaises(TranspilerError):
            self.passmanager.replace(99, PassA_TP_NR_NP())

class TestPassManagerSlicing(SchedulerTestCase):
    """test PassManager slicing."""

    def setUp(self):
        if False:
            return 10
        super().setUp()
        self.passmanager = PassManager()
        self.circuit = QuantumCircuit(QuantumRegister(1))

    def test_empty_passmanager_length(self):
        if False:
            return 10
        'test len(PassManager) when PassManager is empty'
        length = len(self.passmanager)
        expected_length = 0
        self.assertEqual(length, expected_length)

    def test_passmanager_length(self):
        if False:
            for i in range(10):
                print('nop')
        'test len(PassManager) when PassManager is not empty'
        self.passmanager.append(PassA_TP_NR_NP())
        self.passmanager.append(PassA_TP_NR_NP())
        length = len(self.passmanager)
        expected_length = 2
        self.assertEqual(length, expected_length)

    def test_accessing_passmanager_by_index(self):
        if False:
            print('Hello World!')
        "test accessing PassManager's passes by index"
        self.passmanager.append(PassB_TP_RA_PA())
        self.passmanager.append(PassC_TP_RA_PA())
        new_passmanager = self.passmanager[1]
        expected = ['run transformation pass PassA_TP_NR_NP', 'run transformation pass PassC_TP_RA_PA']
        self.assertScheduler(self.circuit, new_passmanager, expected)

    def test_accessing_passmanager_by_index_with_condition(self):
        if False:
            print('Hello World!')
        "test accessing PassManager's conditioned passes by index"
        self.passmanager.append(PassF_reduce_dag_property())
        self.passmanager.append([PassK_check_fixed_point_property(), PassA_TP_NR_NP(), PassF_reduce_dag_property()], condition=lambda property_set: True, do_while=lambda property_set: not property_set['property_fixed_point'])
        new_passmanager = self.passmanager[1]
        expected = ['run analysis pass PassG_calculates_dag_property', 'set property as 8 (from dag.property)', 'run analysis pass PassK_check_fixed_point_property', 'run transformation pass PassA_TP_NR_NP', 'run transformation pass PassF_reduce_dag_property', 'dag property = 6', 'run analysis pass PassG_calculates_dag_property', 'set property as 6 (from dag.property)', 'run analysis pass PassK_check_fixed_point_property', 'run transformation pass PassA_TP_NR_NP', 'run transformation pass PassF_reduce_dag_property', 'dag property = 5', 'run analysis pass PassG_calculates_dag_property', 'set property as 5 (from dag.property)', 'run analysis pass PassK_check_fixed_point_property', 'run transformation pass PassA_TP_NR_NP', 'run transformation pass PassF_reduce_dag_property', 'dag property = 4', 'run analysis pass PassG_calculates_dag_property', 'set property as 4 (from dag.property)', 'run analysis pass PassK_check_fixed_point_property', 'run transformation pass PassA_TP_NR_NP', 'run transformation pass PassF_reduce_dag_property', 'dag property = 3', 'run analysis pass PassG_calculates_dag_property', 'set property as 3 (from dag.property)', 'run analysis pass PassK_check_fixed_point_property', 'run transformation pass PassA_TP_NR_NP', 'run transformation pass PassF_reduce_dag_property', 'dag property = 2', 'run analysis pass PassG_calculates_dag_property', 'set property as 2 (from dag.property)', 'run analysis pass PassK_check_fixed_point_property', 'run transformation pass PassA_TP_NR_NP', 'run transformation pass PassF_reduce_dag_property', 'dag property = 2', 'run analysis pass PassG_calculates_dag_property', 'set property as 2 (from dag.property)', 'run analysis pass PassK_check_fixed_point_property', 'run transformation pass PassA_TP_NR_NP', 'run transformation pass PassF_reduce_dag_property', 'dag property = 2']
        self.assertScheduler(self.circuit, new_passmanager, expected)

    def test_accessing_passmanager_by_range(self):
        if False:
            while True:
                i = 10
        "test accessing PassManager's passes by range"
        self.passmanager.append(PassC_TP_RA_PA())
        self.passmanager.append(PassB_TP_RA_PA())
        self.passmanager.append(PassC_TP_RA_PA())
        self.passmanager.append(PassD_TP_NR_NP())
        new_passmanager = self.passmanager[1:3]
        expected = ['run transformation pass PassA_TP_NR_NP', 'run transformation pass PassB_TP_RA_PA', 'run transformation pass PassC_TP_RA_PA']
        self.assertScheduler(self.circuit, new_passmanager, expected)

    def test_accessing_passmanager_by_range_with_condition(self):
        if False:
            print('Hello World!')
        "test accessing PassManager's passes by range with condition"
        self.passmanager.append(PassB_TP_RA_PA())
        self.passmanager.append(PassE_AP_NR_NP(True))
        self.passmanager.append(PassA_TP_NR_NP(), condition=lambda property_set: property_set['property'])
        self.passmanager.append(PassB_TP_RA_PA())
        new_passmanager = self.passmanager[1:3]
        expected = ['run analysis pass PassE_AP_NR_NP', 'set property as True', 'run transformation pass PassA_TP_NR_NP']
        self.assertScheduler(self.circuit, new_passmanager, expected)

    def test_accessing_passmanager_error(self):
        if False:
            print('Hello World!')
        'testing accessing a pass item not in list'
        self.passmanager.append(PassB_TP_RA_PA())
        with self.assertRaises(IndexError):
            self.passmanager = self.passmanager[99]

class TestPassManagerConcatenation(SchedulerTestCase):
    """test PassManager concatenation by + operator."""

    def setUp(self):
        if False:
            print('Hello World!')
        super().setUp()
        self.passmanager1 = PassManager()
        self.passmanager2 = PassManager()
        self.circuit = QuantumCircuit(QuantumRegister(1))

    def test_concatenating_passmanagers(self):
        if False:
            print('Hello World!')
        'test adding two PassManagers together'
        self.passmanager1.append(PassB_TP_RA_PA())
        self.passmanager2.append(PassC_TP_RA_PA())
        new_passmanager = self.passmanager1 + self.passmanager2
        expected = ['run transformation pass PassA_TP_NR_NP', 'run transformation pass PassB_TP_RA_PA', 'run transformation pass PassC_TP_RA_PA']
        self.assertScheduler(self.circuit, new_passmanager, expected)

    def test_concatenating_passmanagers_with_condition(self):
        if False:
            print('Hello World!')
        'test adding two pass managers with condition'
        self.passmanager1.append(PassE_AP_NR_NP(True))
        self.passmanager1.append(PassB_TP_RA_PA())
        self.passmanager2.append(PassC_TP_RA_PA(), condition=lambda property_set: property_set['property'])
        self.passmanager2.append(PassB_TP_RA_PA())
        new_passmanager = self.passmanager1 + self.passmanager2
        expected = ['run analysis pass PassE_AP_NR_NP', 'set property as True', 'run transformation pass PassA_TP_NR_NP', 'run transformation pass PassB_TP_RA_PA', 'run transformation pass PassC_TP_RA_PA', 'run transformation pass PassB_TP_RA_PA']
        self.assertScheduler(self.circuit, new_passmanager, expected)

    def test_adding_pass_to_passmanager(self):
        if False:
            for i in range(10):
                print('nop')
        'test adding a pass to PassManager'
        self.passmanager1.append(PassE_AP_NR_NP(argument1=1))
        self.passmanager1.append(PassB_TP_RA_PA())
        self.passmanager1 += PassC_TP_RA_PA()
        expected = ['run analysis pass PassE_AP_NR_NP', 'set property as 1', 'run transformation pass PassA_TP_NR_NP', 'run transformation pass PassB_TP_RA_PA', 'run transformation pass PassC_TP_RA_PA']
        self.assertScheduler(self.circuit, self.passmanager1, expected)

    def test_adding_list_of_passes_to_passmanager(self):
        if False:
            return 10
        'test adding a list of passes to PassManager'
        self.passmanager1.append(PassE_AP_NR_NP(argument1=1))
        self.passmanager1.append(PassB_TP_RA_PA())
        self.passmanager1 += [PassC_TP_RA_PA(), PassB_TP_RA_PA()]
        expected = ['run analysis pass PassE_AP_NR_NP', 'set property as 1', 'run transformation pass PassA_TP_NR_NP', 'run transformation pass PassB_TP_RA_PA', 'run transformation pass PassC_TP_RA_PA', 'run transformation pass PassB_TP_RA_PA']
        self.assertScheduler(self.circuit, self.passmanager1, expected)

    def test_adding_list_of_passes_to_passmanager_with_condition(self):
        if False:
            i = 10
            return i + 15
        'test adding a list of passes to a PassManager that have conditions'
        self.passmanager1.append(PassE_AP_NR_NP(False))
        self.passmanager1.append(PassB_TP_RA_PA(), condition=lambda property_set: property_set['property'])
        self.passmanager1 += PassC_TP_RA_PA()
        expected = ['run analysis pass PassE_AP_NR_NP', 'set property as False', 'run transformation pass PassA_TP_NR_NP', 'run transformation pass PassC_TP_RA_PA']
        self.assertScheduler(self.circuit, self.passmanager1, expected)

    def test_adding_pass_to_passmanager_error(self):
        if False:
            while True:
                i = 10
        'testing adding a non-pass item to PassManager'
        with self.assertRaises(TypeError):
            self.passmanager1 += 'not a pass'

    def test_adding_list_to_passmanager_error(self):
        if False:
            return 10
        'testing adding a list having a non-pass item to PassManager'
        with self.assertRaises(TypeError):
            self.passmanager1 += [PassB_TP_RA_PA(), 'not a pass']
if __name__ == '__main__':
    unittest.main()