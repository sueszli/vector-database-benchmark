"""Test scheduled circuit (quantum circuit with duration)."""
from ddt import ddt, data
from qiskit import QuantumCircuit, QiskitError
from qiskit import transpile, assemble, BasicAer
from qiskit.circuit import Parameter
from qiskit.providers.fake_provider import FakeParis
from qiskit.transpiler.exceptions import TranspilerError
from qiskit.transpiler.instruction_durations import InstructionDurations
from qiskit.test.base import QiskitTestCase

@ddt
class TestScheduledCircuit(QiskitTestCase):
    """Test scheduled circuit (quantum circuit with duration)."""

    def setUp(self):
        if False:
            return 10
        super().setUp()
        self.backend_with_dt = FakeParis()
        self.backend_without_dt = FakeParis()
        delattr(self.backend_without_dt.configuration(), 'dt')
        self.dt = 2.2222222222222221e-10
        self.simulator_backend = BasicAer.get_backend('qasm_simulator')

    def test_schedule_circuit_when_backend_tells_dt(self):
        if False:
            for i in range(10):
                print('nop')
        'dt is known to transpiler by backend'
        qc = QuantumCircuit(2)
        qc.delay(0.1, 0, unit='ms')
        qc.delay(100, 0, unit='ns')
        qc.h(0)
        qc.h(1)
        sc = transpile(qc, self.backend_with_dt, scheduling_method='alap', layout_method='trivial')
        self.assertEqual(sc.duration, 450610)
        self.assertEqual(sc.unit, 'dt')
        self.assertEqual(sc.data[0].operation.name, 'delay')
        self.assertEqual(sc.data[0].operation.duration, 450450)
        self.assertEqual(sc.data[0].operation.unit, 'dt')
        self.assertEqual(sc.data[1].operation.name, 'rz')
        self.assertEqual(sc.data[1].operation.duration, 0)
        self.assertEqual(sc.data[1].operation.unit, 'dt')
        self.assertEqual(sc.data[4].operation.name, 'delay')
        self.assertEqual(sc.data[4].operation.duration, 450450)
        self.assertEqual(sc.data[4].operation.unit, 'dt')
        qobj = assemble(sc, self.backend_with_dt)
        self.assertEqual(qobj.experiments[0].instructions[0].name, 'delay')
        self.assertEqual(qobj.experiments[0].instructions[0].params[0], 450450)
        self.assertEqual(qobj.experiments[0].instructions[4].name, 'delay')
        self.assertEqual(qobj.experiments[0].instructions[4].params[0], 450450)

    def test_schedule_circuit_when_transpile_option_tells_dt(self):
        if False:
            return 10
        'dt is known to transpiler by transpile option'
        qc = QuantumCircuit(2)
        qc.delay(0.1, 0, unit='ms')
        qc.delay(100, 0, unit='ns')
        qc.h(0)
        qc.h(1)
        sc = transpile(qc, self.backend_without_dt, scheduling_method='alap', dt=self.dt, layout_method='trivial')
        self.assertEqual(sc.duration, 450610)
        self.assertEqual(sc.unit, 'dt')
        self.assertEqual(sc.data[0].operation.name, 'delay')
        self.assertEqual(sc.data[0].operation.duration, 450450)
        self.assertEqual(sc.data[0].operation.unit, 'dt')
        self.assertEqual(sc.data[1].operation.name, 'rz')
        self.assertEqual(sc.data[1].operation.duration, 0)
        self.assertEqual(sc.data[1].operation.unit, 'dt')
        self.assertEqual(sc.data[4].operation.name, 'delay')
        self.assertEqual(sc.data[4].operation.duration, 450450)
        self.assertEqual(sc.data[4].operation.unit, 'dt')

    def test_schedule_circuit_in_sec_when_no_one_tells_dt(self):
        if False:
            while True:
                i = 10
        'dt is unknown and all delays and gate times are in SI'
        qc = QuantumCircuit(2)
        qc.delay(0.1, 0, unit='ms')
        qc.delay(100, 0, unit='ns')
        qc.h(0)
        qc.h(1)
        sc = transpile(qc, self.backend_without_dt, scheduling_method='alap', layout_method='trivial')
        self.assertAlmostEqual(sc.duration, 450610 * self.dt)
        self.assertEqual(sc.unit, 's')
        self.assertEqual(sc.data[0].operation.name, 'delay')
        self.assertAlmostEqual(sc.data[0].operation.duration, 0.0001 + 1e-07)
        self.assertEqual(sc.data[0].operation.unit, 's')
        self.assertEqual(sc.data[1].operation.name, 'rz')
        self.assertAlmostEqual(sc.data[1].operation.duration, 160 * self.dt)
        self.assertEqual(sc.data[1].operation.unit, 's')
        self.assertEqual(sc.data[4].operation.name, 'delay')
        self.assertAlmostEqual(sc.data[4].operation.duration, 0.0001 + 1e-07)
        self.assertEqual(sc.data[4].operation.unit, 's')
        with self.assertRaises(QiskitError):
            assemble(sc, self.backend_without_dt)

    def test_cannot_schedule_circuit_with_mixed_SI_and_dt_when_no_one_tells_dt(self):
        if False:
            for i in range(10):
                print('nop')
        'dt is unknown but delays and gate times have a mix of SI and dt'
        qc = QuantumCircuit(2)
        qc.delay(100, 0, unit='ns')
        qc.delay(30, 0, unit='dt')
        qc.h(0)
        qc.h(1)
        with self.assertRaises(QiskitError):
            transpile(qc, self.backend_without_dt, scheduling_method='alap')

    def test_transpile_single_delay_circuit(self):
        if False:
            return 10
        qc = QuantumCircuit(1)
        qc.delay(1234, 0)
        sc = transpile(qc, backend=self.backend_with_dt, scheduling_method='alap')
        self.assertEqual(sc.duration, 1234)
        self.assertEqual(sc.data[0].operation.name, 'delay')
        self.assertEqual(sc.data[0].operation.duration, 1234)
        self.assertEqual(sc.data[0].operation.unit, 'dt')

    def test_transpile_t1_circuit(self):
        if False:
            print('Hello World!')
        qc = QuantumCircuit(1)
        qc.x(0)
        qc.delay(1000, 0, unit='ns')
        qc.measure_all()
        scheduled = transpile(qc, backend=self.backend_with_dt, scheduling_method='alap')
        self.assertEqual(scheduled.duration, 23060)

    def test_transpile_delay_circuit_with_backend(self):
        if False:
            i = 10
            return i + 15
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.delay(100, 1, unit='ns')
        qc.cx(0, 1)
        scheduled = transpile(qc, backend=self.backend_with_dt, scheduling_method='alap', layout_method='trivial')
        self.assertEqual(scheduled.duration, 2082)

    def test_transpile_delay_circuit_without_backend(self):
        if False:
            return 10
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.delay(500, 1)
        qc.cx(0, 1)
        scheduled = transpile(qc, scheduling_method='alap', basis_gates=['h', 'cx'], instruction_durations=[('h', 0, 200), ('cx', [0, 1], 700)])
        self.assertEqual(scheduled.duration, 1200)

    def test_transpile_circuit_with_custom_instruction(self):
        if False:
            print('Hello World!')
        'See: https://github.com/Qiskit/qiskit-terra/issues/5154'
        bell = QuantumCircuit(2, name='bell')
        bell.h(0)
        bell.cx(0, 1)
        qc = QuantumCircuit(2)
        qc.delay(500, 1)
        qc.append(bell.to_instruction(), [0, 1])
        scheduled = transpile(qc, scheduling_method='alap', instruction_durations=[('bell', [0, 1], 1000)])
        self.assertEqual(scheduled.duration, 1500)

    def test_transpile_delay_circuit_with_dt_but_without_scheduling_method(self):
        if False:
            return 10
        qc = QuantumCircuit(1)
        qc.delay(100, 0, unit='ns')
        transpiled = transpile(qc, backend=self.backend_with_dt)
        self.assertEqual(transpiled.duration, None)
        self.assertEqual(transpiled.data[0].operation.duration, 450)

    def test_transpile_delay_circuit_without_scheduling_method_or_durs(self):
        if False:
            i = 10
            return i + 15
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.delay(500, 1)
        qc.cx(0, 1)
        not_scheduled = transpile(qc)
        self.assertEqual(not_scheduled.duration, None)

    def test_raise_error_if_transpile_with_scheduling_method_but_without_durations(self):
        if False:
            i = 10
            return i + 15
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.delay(500, 1)
        qc.cx(0, 1)
        with self.assertRaises(TranspilerError):
            transpile(qc, scheduling_method='alap')

    def test_invalidate_schedule_circuit_if_new_instruction_is_appended(self):
        if False:
            print('Hello World!')
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.delay(500 * self.dt, 1, 's')
        qc.cx(0, 1)
        scheduled = transpile(qc, backend=self.backend_with_dt, scheduling_method='alap')
        scheduled.h(0)
        self.assertEqual(scheduled.duration, None)

    def test_default_units_for_my_own_duration_users(self):
        if False:
            for i in range(10):
                print('nop')
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.delay(500, 1)
        qc.cx(0, 1)
        scheduled = transpile(qc, basis_gates=['h', 'cx', 'delay'], scheduling_method='alap', instruction_durations=[('h', 0, 200), ('cx', None, 900)])
        self.assertEqual(scheduled.duration, 1400)
        scheduled = transpile(qc, basis_gates=['h', 'cx', 'delay'], scheduling_method='alap', instruction_durations=[('h', 0, 200), ('cx', None, 900), ('cx', [0, 1], 800)])
        self.assertEqual(scheduled.duration, 1300)

    def test_unit_seconds_when_using_backend_durations(self):
        if False:
            while True:
                i = 10
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.delay(500 * self.dt, 1, 's')
        qc.cx(0, 1)
        scheduled = transpile(qc, backend=self.backend_with_dt, scheduling_method='alap', layout_method='trivial')
        self.assertEqual(scheduled.duration, 2132)
        durations = InstructionDurations.from_backend(self.backend_with_dt)
        durations.update([('cx', [0, 1], 1000 * self.dt, 's')])
        scheduled = transpile(qc, backend=self.backend_with_dt, scheduling_method='alap', instruction_durations=durations, layout_method='trivial')
        self.assertEqual(scheduled.duration, 1500)

    def test_per_qubit_durations(self):
        if False:
            while True:
                i = 10
        'See: https://github.com/Qiskit/qiskit-terra/issues/5109'
        qc = QuantumCircuit(3)
        qc.h(0)
        qc.delay(500, 1)
        qc.cx(0, 1)
        qc.h(1)
        sc = transpile(qc, scheduling_method='alap', basis_gates=['h', 'cx'], instruction_durations=[('h', None, 200), ('cx', [0, 1], 700)])
        self.assertEqual(sc.qubit_start_time(0), 300)
        self.assertEqual(sc.qubit_stop_time(0), 1200)
        self.assertEqual(sc.qubit_start_time(1), 500)
        self.assertEqual(sc.qubit_stop_time(1), 1400)
        self.assertEqual(sc.qubit_start_time(2), 0)
        self.assertEqual(sc.qubit_stop_time(2), 0)
        self.assertEqual(sc.qubit_start_time(0, 1), 300)
        self.assertEqual(sc.qubit_stop_time(0, 1), 1400)
        qc.measure_all()
        sc = transpile(qc, scheduling_method='alap', basis_gates=['h', 'cx', 'measure'], instruction_durations=[('h', None, 200), ('cx', [0, 1], 700), ('measure', None, 1000)])
        q = sc.qubits
        self.assertEqual(sc.qubit_start_time(q[0]), 300)
        self.assertEqual(sc.qubit_stop_time(q[0]), 2400)
        self.assertEqual(sc.qubit_start_time(q[1]), 500)
        self.assertEqual(sc.qubit_stop_time(q[1]), 2400)
        self.assertEqual(sc.qubit_start_time(q[2]), 1400)
        self.assertEqual(sc.qubit_stop_time(q[2]), 2400)
        self.assertEqual(sc.qubit_start_time(*q), 300)
        self.assertEqual(sc.qubit_stop_time(*q), 2400)

    def test_change_dt_in_transpile(self):
        if False:
            return 10
        qc = QuantumCircuit(1, 1)
        qc.x(0)
        qc.measure(0, 0)
        scheduled = transpile(qc, backend=self.backend_with_dt, scheduling_method='asap')
        org_duration = scheduled.duration
        scheduled = transpile(qc, backend=self.backend_with_dt, scheduling_method='asap', dt=self.dt / 2)
        self.assertEqual(scheduled.duration, org_duration * 2)

    @data('asap', 'alap')
    def test_duration_on_same_instruction_instance(self, scheduling_method):
        if False:
            print('Hello World!')
        'See: https://github.com/Qiskit/qiskit-terra/issues/5771'
        assert self.backend_with_dt.properties().gate_length('cx', (0, 1)) != self.backend_with_dt.properties().gate_length('cx', (1, 2))
        qc = QuantumCircuit(3)
        qc.cz(0, 1)
        qc.cz(1, 2)
        sc = transpile(qc, backend=self.backend_with_dt, scheduling_method=scheduling_method)
        cxs = [inst.operation for inst in sc.data if inst.operation.name == 'cx']
        self.assertNotEqual(cxs[0].duration, cxs[1].duration)

    def test_transpile_and_assemble_delay_circuit_for_simulator(self):
        if False:
            return 10
        'See: https://github.com/Qiskit/qiskit-terra/issues/5962'
        qc = QuantumCircuit(1)
        qc.delay(100, 0, 'ns')
        circ = transpile(qc, self.simulator_backend)
        self.assertEqual(circ.duration, None)
        qobj = assemble(circ, self.simulator_backend)
        self.assertEqual(qobj.experiments[0].instructions[0].name, 'delay')
        self.assertEqual(qobj.experiments[0].instructions[0].params[0], 1e-07)

    def test_transpile_and_assemble_t1_circuit_for_simulator(self):
        if False:
            i = 10
            return i + 15
        'Check if no scheduling is done in transpiling for simulator backends'
        qc = QuantumCircuit(1, 1)
        qc.x(0)
        qc.delay(0.1, 0, 'us')
        qc.measure(0, 0)
        circ = transpile(qc, self.simulator_backend)
        self.assertEqual(circ.duration, None)
        qobj = assemble(circ, self.simulator_backend)
        self.assertEqual(qobj.experiments[0].instructions[1].name, 'delay')
        self.assertAlmostEqual(qobj.experiments[0].instructions[1].params[0], 1e-07)

    def test_can_transpile_circuits_after_assigning_parameters(self):
        if False:
            while True:
                i = 10
        'Check if not scheduled but duration is converted in dt'
        idle_dur = Parameter('t')
        qc = QuantumCircuit(1, 1)
        qc.x(0)
        qc.delay(idle_dur, 0, 'us')
        qc.measure(0, 0)
        qc = qc.assign_parameters({idle_dur: 0.1})
        circ = transpile(qc, self.backend_with_dt)
        self.assertEqual(circ.duration, None)
        self.assertEqual(circ.data[1].operation.duration, 450)

    def test_can_transpile_and_assemble_circuits_with_assigning_parameters_inbetween(self):
        if False:
            while True:
                i = 10
        idle_dur = Parameter('t')
        qc = QuantumCircuit(1, 1)
        qc.x(0)
        qc.delay(idle_dur, 0, 'us')
        qc.measure(0, 0)
        circ = transpile(qc, self.backend_with_dt)
        circ = circ.assign_parameters({idle_dur: 0.1})
        qobj = assemble(circ, self.backend_with_dt)
        self.assertEqual(qobj.experiments[0].instructions[1].name, 'delay')
        self.assertEqual(qobj.experiments[0].instructions[1].params[0], 450)

    def test_can_transpile_circuits_with_unbounded_parameters(self):
        if False:
            return 10
        idle_dur = Parameter('t')
        qc = QuantumCircuit(1, 1)
        qc.x(0)
        qc.delay(idle_dur, 0, 'us')
        qc.measure(0, 0)
        circ = transpile(qc, self.backend_with_dt)
        self.assertEqual(circ.duration, None)
        self.assertEqual(circ.data[1].operation.unit, 'dt')
        self.assertEqual(circ.data[1].operation.duration, idle_dur * 1e-06 / self.dt)

    def test_fail_to_assemble_circuits_with_unbounded_parameters(self):
        if False:
            for i in range(10):
                print('nop')
        idle_dur = Parameter('t')
        qc = QuantumCircuit(1, 1)
        qc.x(0)
        qc.delay(idle_dur, 0, 'us')
        qc.measure(0, 0)
        qc = transpile(qc, self.backend_with_dt)
        with self.assertRaises(QiskitError):
            assemble(qc, self.backend_with_dt)

    @data('asap', 'alap')
    def test_can_schedule_circuits_with_bounded_parameters(self, scheduling_method):
        if False:
            return 10
        idle_dur = Parameter('t')
        qc = QuantumCircuit(1, 1)
        qc.x(0)
        qc.delay(idle_dur, 0, 'us')
        qc.measure(0, 0)
        qc = qc.assign_parameters({idle_dur: 0.1})
        circ = transpile(qc, self.backend_with_dt, scheduling_method=scheduling_method)
        self.assertIsNotNone(circ.duration)

    @data('asap', 'alap')
    def test_fail_to_schedule_circuits_with_unbounded_parameters(self, scheduling_method):
        if False:
            return 10
        idle_dur = Parameter('t')
        qc = QuantumCircuit(1, 1)
        qc.x(0)
        qc.delay(idle_dur, 0, 'us')
        qc.measure(0, 0)
        with self.assertRaises(TranspilerError):
            transpile(qc, self.backend_with_dt, scheduling_method=scheduling_method)