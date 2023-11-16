"""Test the InstructionScheduleMap."""
import copy
import pickle
import numpy as np
from qiskit.pulse import library
from qiskit.circuit.library.standard_gates import U1Gate, U3Gate, CXGate, XGate
from qiskit.circuit.parameter import Parameter
from qiskit.circuit.parameterexpression import ParameterExpression
from qiskit.pulse import InstructionScheduleMap, Play, PulseError, Schedule, ScheduleBlock, Waveform, ShiftPhase, Constant
from qiskit.pulse.calibration_entries import CalibrationPublisher
from qiskit.pulse.channels import DriveChannel
from qiskit.qobj import PulseQobjInstruction
from qiskit.qobj.converters import QobjToInstructionConverter
from qiskit.test import QiskitTestCase
from qiskit.providers.fake_provider import FakeOpenPulse2Q, FakeAthens

class TestInstructionScheduleMap(QiskitTestCase):
    """Test the InstructionScheduleMap."""

    def test_add(self):
        if False:
            print('Hello World!')
        'Test add, and that errors are raised when expected.'
        sched = Schedule()
        sched.append(Play(Waveform(np.ones(5)), DriveChannel(0)), inplace=True)
        inst_map = InstructionScheduleMap()
        inst_map.add('u1', 1, sched)
        inst_map.add('u1', 0, sched)
        self.assertIn('u1', inst_map.instructions)
        self.assertEqual(inst_map.qubits_with_instruction('u1'), [0, 1])
        self.assertTrue('u1' in inst_map.qubit_instructions(0))
        with self.assertRaises(PulseError):
            inst_map.add('u1', (), sched)
        with self.assertRaises(PulseError):
            inst_map.add('u1', 1, 'not a schedule')

    def test_add_block(self):
        if False:
            print('Hello World!')
        'Test add block, and that errors are raised when expected.'
        sched = ScheduleBlock()
        sched.append(Play(Waveform(np.ones(5)), DriveChannel(0)), inplace=True)
        inst_map = InstructionScheduleMap()
        inst_map.add('u1', 1, sched)
        inst_map.add('u1', 0, sched)
        self.assertIn('u1', inst_map.instructions)
        self.assertEqual(inst_map.qubits_with_instruction('u1'), [0, 1])
        self.assertTrue('u1' in inst_map.qubit_instructions(0))

    def test_instructions(self):
        if False:
            return 10
        'Test `instructions`.'
        sched = Schedule()
        inst_map = InstructionScheduleMap()
        inst_map.add('u1', 1, sched)
        inst_map.add('u3', 0, sched)
        instructions = inst_map.instructions
        for inst in ['u1', 'u3']:
            self.assertTrue(inst in instructions)

    def test_has(self):
        if False:
            return 10
        'Test `has` and `assert_has`.'
        sched = Schedule()
        inst_map = InstructionScheduleMap()
        inst_map.add('u1', (0,), sched)
        inst_map.add('cx', [0, 1], sched)
        self.assertTrue(inst_map.has('u1', [0]))
        self.assertTrue(inst_map.has('cx', (0, 1)))
        with self.assertRaises(PulseError):
            inst_map.assert_has('dne', [0])
        with self.assertRaises(PulseError):
            inst_map.assert_has('cx', 100)

    def test_has_from_mock(self):
        if False:
            i = 10
            return i + 15
        'Test `has` and `assert_has` from mock data.'
        inst_map = FakeOpenPulse2Q().defaults().instruction_schedule_map
        self.assertTrue(inst_map.has('u1', [0]))
        self.assertTrue(inst_map.has('cx', (0, 1)))
        self.assertTrue(inst_map.has('u3', 0))
        self.assertTrue(inst_map.has('measure', [0, 1]))
        self.assertFalse(inst_map.has('u1', [0, 1]))
        with self.assertRaises(PulseError):
            inst_map.assert_has('dne', [0])
        with self.assertRaises(PulseError):
            inst_map.assert_has('cx', 100)

    def test_qubits_with_instruction(self):
        if False:
            for i in range(10):
                print('nop')
        'Test `qubits_with_instruction`.'
        sched = Schedule()
        inst_map = InstructionScheduleMap()
        inst_map.add('u1', (0,), sched)
        inst_map.add('u1', (1,), sched)
        inst_map.add('cx', [0, 1], sched)
        self.assertEqual(inst_map.qubits_with_instruction('u1'), [0, 1])
        self.assertEqual(inst_map.qubits_with_instruction('cx'), [(0, 1)])
        self.assertEqual(inst_map.qubits_with_instruction('none'), [])

    def test_qubit_instructions(self):
        if False:
            i = 10
            return i + 15
        'Test `qubit_instructions`.'
        sched = Schedule()
        inst_map = InstructionScheduleMap()
        inst_map.add('u1', (0,), sched)
        inst_map.add('u1', (1,), sched)
        inst_map.add('cx', [0, 1], sched)
        self.assertEqual(inst_map.qubit_instructions(0), ['u1'])
        self.assertEqual(inst_map.qubit_instructions(1), ['u1'])
        self.assertEqual(inst_map.qubit_instructions((0, 1)), ['cx'])
        self.assertEqual(inst_map.qubit_instructions(10), [])

    def test_get(self):
        if False:
            return 10
        'Test `get`.'
        sched = Schedule()
        sched.append(Play(Waveform(np.ones(5)), DriveChannel(0)), inplace=True)
        inst_map = InstructionScheduleMap()
        inst_map.add('x', 0, sched)
        self.assertEqual(sched, inst_map.get('x', (0,)))

    def test_get_block(self):
        if False:
            while True:
                i = 10
        'Test `get` block.'
        sched = ScheduleBlock()
        sched.append(Play(Waveform(np.ones(5)), DriveChannel(0)), inplace=True)
        inst_map = InstructionScheduleMap()
        inst_map.add('x', 0, sched)
        self.assertEqual(sched, inst_map.get('x', (0,)))

    def test_remove(self):
        if False:
            print('Hello World!')
        'Test removing a defined operation and removing an undefined operation.'
        sched = Schedule()
        inst_map = InstructionScheduleMap()
        inst_map.add('tmp', 0, sched)
        inst_map.remove('tmp', 0)
        self.assertFalse(inst_map.has('tmp', 0))
        with self.assertRaises(PulseError):
            inst_map.remove('not_there', (0,))
        self.assertFalse('tmp' in inst_map.qubit_instructions(0))

    def test_pop(self):
        if False:
            i = 10
            return i + 15
        'Test pop with default.'
        sched = Schedule()
        inst_map = InstructionScheduleMap()
        inst_map.add('tmp', 100, sched)
        self.assertEqual(inst_map.pop('tmp', 100), sched)
        self.assertFalse(inst_map.has('tmp', 100))
        self.assertEqual(inst_map.qubit_instructions(100), [])
        self.assertEqual(inst_map.qubits_with_instruction('tmp'), [])
        with self.assertRaises(PulseError):
            inst_map.pop('not_there', (0,))

    def test_add_gate(self):
        if False:
            return 10
        'Test add, and that errors are raised when expected.'
        sched = Schedule()
        sched.append(Play(Waveform(np.ones(5)), DriveChannel(0)))
        inst_map = InstructionScheduleMap()
        inst_map.add(U1Gate(0), 1, sched)
        inst_map.add(U1Gate(0), 0, sched)
        self.assertIn('u1', inst_map.instructions)
        self.assertEqual(inst_map.qubits_with_instruction(U1Gate(0)), [0, 1])
        self.assertTrue('u1' in inst_map.qubit_instructions(0))
        with self.assertRaises(PulseError):
            inst_map.add(U1Gate(0), (), sched)
        with self.assertRaises(PulseError):
            inst_map.add(U1Gate(0), 1, 'not a schedule')

    def test_instructions_gate(self):
        if False:
            i = 10
            return i + 15
        'Test `instructions`.'
        sched = Schedule()
        inst_map = InstructionScheduleMap()
        inst_map.add(U1Gate(0), 1, sched)
        inst_map.add(U3Gate(0, 0, 0), 0, sched)
        instructions = inst_map.instructions
        for inst in ['u1', 'u3']:
            self.assertTrue(inst in instructions)

    def test_has_gate(self):
        if False:
            for i in range(10):
                print('nop')
        'Test `has` and `assert_has`.'
        sched = Schedule()
        inst_map = InstructionScheduleMap()
        inst_map.add(U1Gate(0), (0,), sched)
        inst_map.add(CXGate(), [0, 1], sched)
        self.assertTrue(inst_map.has(U1Gate(0), [0]))
        self.assertTrue(inst_map.has(CXGate(), (0, 1)))
        with self.assertRaises(PulseError):
            inst_map.assert_has('dne', [0])
        with self.assertRaises(PulseError):
            inst_map.assert_has(CXGate(), 100)

    def test_has_from_mock_gate(self):
        if False:
            while True:
                i = 10
        'Test `has` and `assert_has` from mock data.'
        inst_map = FakeOpenPulse2Q().defaults().instruction_schedule_map
        self.assertTrue(inst_map.has(U1Gate(0), [0]))
        self.assertTrue(inst_map.has(CXGate(), (0, 1)))
        self.assertTrue(inst_map.has(U3Gate(0, 0, 0), 0))
        self.assertTrue(inst_map.has('measure', [0, 1]))
        self.assertFalse(inst_map.has(U1Gate(0), [0, 1]))
        with self.assertRaises(PulseError):
            inst_map.assert_has('dne', [0])
        with self.assertRaises(PulseError):
            inst_map.assert_has(CXGate(), 100)

    def test_qubits_with_instruction_gate(self):
        if False:
            i = 10
            return i + 15
        'Test `qubits_with_instruction`.'
        sched = Schedule()
        inst_map = InstructionScheduleMap()
        inst_map.add(U1Gate(0), (0,), sched)
        inst_map.add(U1Gate(0), (1,), sched)
        inst_map.add(CXGate(), [0, 1], sched)
        self.assertEqual(inst_map.qubits_with_instruction(U1Gate(0)), [0, 1])
        self.assertEqual(inst_map.qubits_with_instruction(CXGate()), [(0, 1)])
        self.assertEqual(inst_map.qubits_with_instruction('none'), [])

    def test_qubit_instructions_gate(self):
        if False:
            return 10
        'Test `qubit_instructions`.'
        sched = Schedule()
        inst_map = InstructionScheduleMap()
        inst_map.add(U1Gate(0), (0,), sched)
        inst_map.add(U1Gate(0), (1,), sched)
        inst_map.add(CXGate(), [0, 1], sched)
        self.assertEqual(inst_map.qubit_instructions(0), ['u1'])
        self.assertEqual(inst_map.qubit_instructions(1), ['u1'])
        self.assertEqual(inst_map.qubit_instructions((0, 1)), ['cx'])
        self.assertEqual(inst_map.qubit_instructions(10), [])

    def test_get_gate(self):
        if False:
            return 10
        'Test `get`.'
        sched = Schedule()
        sched.append(Play(Waveform(np.ones(5)), DriveChannel(0)))
        inst_map = InstructionScheduleMap()
        inst_map.add(XGate(), 0, sched)
        self.assertEqual(sched, inst_map.get(XGate(), (0,)))

    def test_remove_gate(self):
        if False:
            i = 10
            return i + 15
        'Test removing a defined operation and removing an undefined operation.'
        sched = Schedule()
        inst_map = InstructionScheduleMap()
        inst_map.add('tmp', 0, sched)
        inst_map.remove('tmp', 0)
        self.assertFalse(inst_map.has('tmp', 0))
        with self.assertRaises(PulseError):
            inst_map.remove('not_there', (0,))
        self.assertFalse('tmp' in inst_map.qubit_instructions(0))

    def test_pop_gate(self):
        if False:
            while True:
                i = 10
        'Test pop with default.'
        sched = Schedule()
        inst_map = InstructionScheduleMap()
        inst_map.add(XGate(), 100, sched)
        self.assertEqual(inst_map.pop(XGate(), 100), sched)
        self.assertFalse(inst_map.has(XGate(), 100))
        self.assertEqual(inst_map.qubit_instructions(100), [])
        self.assertEqual(inst_map.qubits_with_instruction(XGate()), [])
        with self.assertRaises(PulseError):
            inst_map.pop('not_there', (0,))

    def test_sequenced_parameterized_schedule(self):
        if False:
            return 10
        'Test parameterized schedule consists of multiple instruction.'
        converter = QobjToInstructionConverter([], buffer=0)
        qobjs = [PulseQobjInstruction(name='fc', ch='d0', t0=10, phase='P1'), PulseQobjInstruction(name='fc', ch='d0', t0=20, phase='P2'), PulseQobjInstruction(name='fc', ch='d0', t0=30, phase='P3')]
        converted_instruction = [converter(qobj) for qobj in qobjs]
        inst_map = InstructionScheduleMap()
        inst_map.add('inst_seq', 0, Schedule(*converted_instruction, name='inst_seq'))
        with self.assertRaises(PulseError):
            inst_map.get('inst_seq', 0, P1=1, P2=2, P3=3, P4=4, P5=5)
        with self.assertRaises(PulseError):
            inst_map.get('inst_seq', 0, 1, 2, 3, 4, 5, 6, 7, 8)
        p3_expr = Parameter('p3')
        p3_expr = p3_expr.bind({p3_expr: 3})
        sched = inst_map.get('inst_seq', 0, 1, 2, p3_expr)
        self.assertEqual(sched.instructions[0][-1].phase, 1)
        self.assertEqual(sched.instructions[1][-1].phase, 2)
        self.assertEqual(sched.instructions[2][-1].phase, 3)
        sched = inst_map.get('inst_seq', 0, P1=1, P2=2, P3=p3_expr)
        self.assertEqual(sched.instructions[0][-1].phase, 1)
        self.assertEqual(sched.instructions[1][-1].phase, 2)
        self.assertEqual(sched.instructions[2][-1].phase, 3)
        sched = inst_map.get('inst_seq', 0, 1, 2, P3=p3_expr)
        self.assertEqual(sched.instructions[0][-1].phase, 1)
        self.assertEqual(sched.instructions[1][-1].phase, 2)
        self.assertEqual(sched.instructions[2][-1].phase, 3)

    def test_schedule_generator(self):
        if False:
            while True:
                i = 10
        'Test schedule generator functionalty.'
        dur_val = 10
        amp = 1.0

        def test_func(dur: int):
            if False:
                for i in range(10):
                    print('nop')
            sched = Schedule()
            with self.assertWarns(DeprecationWarning):
                waveform = library.constant(int(dur), amp)
            sched += Play(waveform, DriveChannel(0))
            return sched
        expected_sched = Schedule()
        with self.assertWarns(DeprecationWarning):
            cons_waveform = library.constant(dur_val, amp)
        expected_sched += Play(cons_waveform, DriveChannel(0))
        inst_map = InstructionScheduleMap()
        inst_map.add('f', (0,), test_func)
        self.assertEqual(inst_map.get('f', (0,), dur_val), expected_sched)
        self.assertEqual(inst_map.get_parameters('f', (0,)), ('dur',))

    def test_schedule_generator_supports_parameter_expressions(self):
        if False:
            return 10
        'Test expression-based schedule generator functionalty.'
        t_param = Parameter('t')
        amp = 1.0

        def test_func(dur: ParameterExpression, t_val: int):
            if False:
                i = 10
                return i + 15
            dur_bound = dur.bind({t_param: t_val})
            sched = Schedule()
            with self.assertWarns(DeprecationWarning):
                waveform = library.constant(int(float(dur_bound)), amp)
            sched += Play(waveform, DriveChannel(0))
            return sched
        expected_sched = Schedule()
        with self.assertWarns(DeprecationWarning):
            cons_waveform = library.constant(10, amp)
        expected_sched += Play(cons_waveform, DriveChannel(0))
        inst_map = InstructionScheduleMap()
        inst_map.add('f', (0,), test_func)
        self.assertEqual(inst_map.get('f', (0,), dur=2 * t_param, t_val=5), expected_sched)
        self.assertEqual(inst_map.get_parameters('f', (0,)), ('dur', 't_val'))

    def test_schedule_with_non_alphanumeric_ordering(self):
        if False:
            for i in range(10):
                print('nop')
        'Test adding and getting schedule with non obvious parameter ordering.'
        theta = Parameter('theta')
        phi = Parameter('phi')
        lamb = Parameter('lam')
        target_sched = Schedule()
        target_sched.insert(0, ShiftPhase(theta, DriveChannel(0)), inplace=True)
        target_sched.insert(10, ShiftPhase(phi, DriveChannel(0)), inplace=True)
        target_sched.insert(20, ShiftPhase(lamb, DriveChannel(0)), inplace=True)
        inst_map = InstructionScheduleMap()
        inst_map.add('target_sched', (0,), target_sched, arguments=['theta', 'phi', 'lam'])
        ref_sched = Schedule()
        ref_sched.insert(0, ShiftPhase(0, DriveChannel(0)), inplace=True)
        ref_sched.insert(10, ShiftPhase(1, DriveChannel(0)), inplace=True)
        ref_sched.insert(20, ShiftPhase(2, DriveChannel(0)), inplace=True)
        test_sched = inst_map.get('target_sched', (0,), 0, 1, 2)
        for (test_inst, ref_inst) in zip(test_sched.instructions, ref_sched.instructions):
            self.assertEqual(test_inst[0], ref_inst[0])
            self.assertEqual(test_inst[1], ref_inst[1])

    def test_binding_too_many_parameters(self):
        if False:
            print('Hello World!')
        'Test getting schedule with too many parameter binding.'
        param = Parameter('param')
        target_sched = Schedule()
        target_sched.insert(0, ShiftPhase(param, DriveChannel(0)), inplace=True)
        inst_map = InstructionScheduleMap()
        inst_map.add('target_sched', (0,), target_sched)
        with self.assertRaises(PulseError):
            inst_map.get('target_sched', (0,), 0, 1, 2, 3)

    def test_binding_unassigned_parameters(self):
        if False:
            for i in range(10):
                print('nop')
        'Test getting schedule with unassigned parameter binding.'
        param = Parameter('param')
        target_sched = Schedule()
        target_sched.insert(0, ShiftPhase(param, DriveChannel(0)), inplace=True)
        inst_map = InstructionScheduleMap()
        inst_map.add('target_sched', (0,), target_sched)
        with self.assertRaises(PulseError):
            inst_map.get('target_sched', (0,), P0=0)

    def test_schedule_with_multiple_parameters_under_same_name(self):
        if False:
            return 10
        'Test getting schedule with parameters that have the same name.'
        param1 = Parameter('param')
        param2 = Parameter('param')
        param3 = Parameter('param')
        target_sched = Schedule()
        target_sched.insert(0, ShiftPhase(param1, DriveChannel(0)), inplace=True)
        target_sched.insert(10, ShiftPhase(param2, DriveChannel(0)), inplace=True)
        target_sched.insert(20, ShiftPhase(param3, DriveChannel(0)), inplace=True)
        inst_map = InstructionScheduleMap()
        inst_map.add('target_sched', (0,), target_sched)
        ref_sched = Schedule()
        ref_sched.insert(0, ShiftPhase(1.23, DriveChannel(0)), inplace=True)
        ref_sched.insert(10, ShiftPhase(1.23, DriveChannel(0)), inplace=True)
        ref_sched.insert(20, ShiftPhase(1.23, DriveChannel(0)), inplace=True)
        test_sched = inst_map.get('target_sched', (0,), param=1.23)
        for (test_inst, ref_inst) in zip(test_sched.instructions, ref_sched.instructions):
            self.assertEqual(test_inst[0], ref_inst[0])
            self.assertAlmostEqual(test_inst[1], ref_inst[1])

    def test_get_schedule_with_unbound_parameter(self):
        if False:
            while True:
                i = 10
        'Test get schedule with partial binding.'
        param1 = Parameter('param1')
        param2 = Parameter('param2')
        target_sched = Schedule()
        target_sched.insert(0, ShiftPhase(param1, DriveChannel(0)), inplace=True)
        target_sched.insert(10, ShiftPhase(param2, DriveChannel(0)), inplace=True)
        inst_map = InstructionScheduleMap()
        inst_map.add('target_sched', (0,), target_sched)
        ref_sched = Schedule()
        ref_sched.insert(0, ShiftPhase(param1, DriveChannel(0)), inplace=True)
        ref_sched.insert(10, ShiftPhase(1.23, DriveChannel(0)), inplace=True)
        test_sched = inst_map.get('target_sched', (0,), param2=1.23)
        for (test_inst, ref_inst) in zip(test_sched.instructions, ref_sched.instructions):
            self.assertEqual(test_inst[0], ref_inst[0])
            self.assertAlmostEqual(test_inst[1], ref_inst[1])

    def test_partially_bound_callable(self):
        if False:
            print('Hello World!')
        'Test register partial function.'
        import functools

        def callable_schedule(par_b, par_a):
            if False:
                while True:
                    i = 10
            sched = Schedule()
            sched.insert(10, Play(Constant(10, par_b), DriveChannel(0)), inplace=True)
            sched.insert(20, Play(Constant(10, par_a), DriveChannel(0)), inplace=True)
            return sched
        ref_sched = Schedule()
        ref_sched.insert(10, Play(Constant(10, 0.1), DriveChannel(0)), inplace=True)
        ref_sched.insert(20, Play(Constant(10, 0.2), DriveChannel(0)), inplace=True)
        inst_map = InstructionScheduleMap()

        def test_callable_sched1(par_b):
            if False:
                while True:
                    i = 10
            return callable_schedule(par_b, 0.2)
        inst_map.add('my_gate1', (0,), test_callable_sched1, ['par_b'])
        ret_sched = inst_map.get('my_gate1', (0,), par_b=0.1)
        self.assertEqual(ret_sched, ref_sched)
        test_callable_sched2 = functools.partial(callable_schedule, par_a=0.2)
        inst_map.add('my_gate2', (0,), test_callable_sched2, ['par_b'])
        ret_sched = inst_map.get('my_gate2', (0,), par_b=0.1)
        self.assertEqual(ret_sched, ref_sched)

    def test_two_instmaps_equal(self):
        if False:
            for i in range(10):
                print('nop')
        'Test eq method when two instmaps are identical.'
        instmap1 = FakeAthens().defaults().instruction_schedule_map
        instmap2 = copy.deepcopy(instmap1)
        self.assertEqual(instmap1, instmap2)

    def test_two_instmaps_different(self):
        if False:
            i = 10
            return i + 15
        'Test eq method when two instmaps are not identical.'
        instmap1 = FakeAthens().defaults().instruction_schedule_map
        instmap2 = copy.deepcopy(instmap1)
        instmap2.add('sx', (0,), Schedule())
        self.assertNotEqual(instmap1, instmap2)

    def test_instmap_picklable(self):
        if False:
            print('Hello World!')
        'Test if instmap can be pickled.'
        instmap = FakeAthens().defaults().instruction_schedule_map
        ser_obj = pickle.dumps(instmap)
        deser_instmap = pickle.loads(ser_obj)
        self.assertEqual(instmap, deser_instmap)

    def test_instmap_picklable_with_arguments(self):
        if False:
            i = 10
            return i + 15
        'Test instmap pickling with an edge case.\n\n        This test attempts to pickle instmap with custom entry,\n        in which arguments are provided by users in the form of\n        python dict key object that is not picklable.\n        '
        instmap = FakeAthens().defaults().instruction_schedule_map
        param1 = Parameter('P1')
        param2 = Parameter('P2')
        sched = Schedule()
        sched.insert(0, Play(Constant(100, param1), DriveChannel(0)), inplace=True)
        sched.insert(0, Play(Constant(100, param2), DriveChannel(1)), inplace=True)
        to_assign = {'P1': 0.1, 'P2': 0.2}
        instmap.add('custom', (0, 1), sched, arguments=to_assign.keys())
        ser_obj = pickle.dumps(instmap)
        deser_instmap = pickle.loads(ser_obj)
        self.assertEqual(instmap, deser_instmap)

    def test_check_backend_provider_cals(self):
        if False:
            for i in range(10):
                print('nop')
        'Test if schedules provided by backend provider is distinguishable.'
        instmap = FakeOpenPulse2Q().defaults().instruction_schedule_map
        publisher = instmap.get('u1', (0,), P0=0).metadata['publisher']
        self.assertEqual(publisher, CalibrationPublisher.BACKEND_PROVIDER)

    def test_check_user_cals(self):
        if False:
            print('Hello World!')
        'Test if schedules provided by user is distinguishable.'
        instmap = FakeOpenPulse2Q().defaults().instruction_schedule_map
        test_u1 = Schedule()
        test_u1 += ShiftPhase(Parameter('P0'), DriveChannel(0))
        instmap.add('u1', (0,), test_u1, arguments=['P0'])
        publisher = instmap.get('u1', (0,), P0=0).metadata['publisher']
        self.assertEqual(publisher, CalibrationPublisher.QISKIT)

    def test_has_custom_gate(self):
        if False:
            return 10
        'Test method to check custom gate.'
        backend = FakeOpenPulse2Q()
        instmap = backend.defaults().instruction_schedule_map
        self.assertFalse(instmap.has_custom_gate())
        some_sched = Schedule()
        instmap.add('u3', (0,), some_sched)
        self.assertTrue(instmap.has_custom_gate())
        instmap.remove('u3', (0,))
        self.assertFalse(instmap.has_custom_gate())