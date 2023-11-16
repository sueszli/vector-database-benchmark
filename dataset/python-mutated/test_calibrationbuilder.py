"""Test the CalibrationBuilder subclasses."""
from math import pi, erf
import numpy as np
from ddt import data, ddt
from qiskit.converters import circuit_to_dag
from qiskit import circuit, schedule, QiskitError, QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.circuit.library.standard_gates import SXGate, RZGate, RXGate
from qiskit.providers.fake_provider import FakeHanoi
from qiskit.providers.fake_provider import FakeArmonk
from qiskit.providers.fake_provider import FakeBelemV2
from qiskit.pulse import ControlChannel, DriveChannel, GaussianSquare, Waveform, Play, InstructionScheduleMap, Schedule, Drag, Square
from qiskit.pulse import builder
from qiskit.pulse.transforms import target_qobj_transform
from qiskit.dagcircuit import DAGOpNode
from qiskit.test import QiskitTestCase
from qiskit.transpiler import PassManager, Target, InstructionProperties
from qiskit.transpiler.passes.calibration.builders import RZXCalibrationBuilder, RZXCalibrationBuilderNoEcho, RXCalibrationBuilder

class TestCalibrationBuilder(QiskitTestCase):
    """Test the Calibration Builder."""
    __risefall = 4
    __angle = np.pi / 2
    __granularity = 16

    @staticmethod
    def get_cr_play(cr_schedule, name):
        if False:
            print('Hello World!')
        'A helper function to filter CR pulses.'

        def _filter_func(time_inst):
            if False:
                return 10
            return isinstance(time_inst[1], Play) and time_inst[1].pulse.name.startswith(name)
        return cr_schedule.filter(_filter_func).instructions[0][1]

    def compute_stretch_duration(self, play_gaussian_square_pulse, theta):
        if False:
            for i in range(10):
                print('nop')
        'Compute duration of stretched Gaussian Square pulse.'
        pulse = play_gaussian_square_pulse.pulse
        sigma = pulse.sigma
        width = self.compute_stretch_width(play_gaussian_square_pulse, theta)
        duration = width + sigma * self.__risefall
        return round(duration / self.__granularity) * self.__granularity

    def compute_stretch_width(self, play_gaussian_square_pulse, theta):
        if False:
            while True:
                i = 10
        'Compute width of stretched Gaussian Square pulse.'
        pulse = play_gaussian_square_pulse.pulse
        sigma = pulse.sigma
        width = pulse.width
        risefall_area = sigma * np.sqrt(2 * np.pi) * erf(self.__risefall)
        full_area = risefall_area + width
        target_area = abs(theta) / self.__angle * full_area
        return max(0, target_area - risefall_area)

    def u0p_play(self, cr_schedule):
        if False:
            i = 10
            return i + 15
        'Returns the positive CR pulse from cr_schedule.'
        return self.get_cr_play(cr_schedule, 'CR90p_u')

    def u0m_play(self, cr_schedule):
        if False:
            return 10
        'Returns the negative CR pulse from cr_schedule.'
        return self.get_cr_play(cr_schedule, 'CR90m_u')

    def d1p_play(self, cr_schedule):
        if False:
            for i in range(10):
                print('nop')
        'Returns the positive rotary echo pulse from cr_schedule.'
        return self.get_cr_play(cr_schedule, 'CR90p_d')

    def d1m_play(self, cr_schedule):
        if False:
            return 10
        'Returns the negative rotary echo pulse from cr_schedule.'
        return self.get_cr_play(cr_schedule, 'CR90m_d')

@ddt
class TestRZXCalibrationBuilder(TestCalibrationBuilder):
    """Test RZXCalibrationBuilder."""

    def build_forward(self, backend, theta, u0p_play, d1p_play, u0m_play, d1m_play):
        if False:
            while True:
                i = 10
        'A helper function to generate reference pulse schedule for forward direction.'
        duration = self.compute_stretch_duration(u0p_play, theta)
        width = self.compute_stretch_width(u0p_play, theta)
        with builder.build(backend, default_alignment='sequential', default_transpiler_settings={'optimization_level': 0}) as ref_sched:
            with builder.align_left():
                u0p_params = u0p_play.pulse.parameters
                u0p_params['duration'] = duration
                u0p_params['width'] = width
                builder.play(GaussianSquare(**u0p_params), ControlChannel(0))
                d1p_params = d1p_play.pulse.parameters
                d1p_params['duration'] = duration
                d1p_params['width'] = width
                builder.play(GaussianSquare(**d1p_params), DriveChannel(1))
            builder.x(0)
            with builder.align_left():
                u0m_params = u0m_play.pulse.parameters
                u0m_params['duration'] = duration
                u0m_params['width'] = width
                builder.play(GaussianSquare(**u0m_params), ControlChannel(0))
                d1m_params = d1m_play.pulse.parameters
                d1m_params['duration'] = duration
                d1m_params['width'] = width
                builder.play(GaussianSquare(**d1m_params), DriveChannel(1))
            builder.x(0)
        return ref_sched

    def build_reverse(self, backend, theta, u0p_play, d1p_play, u0m_play, d1m_play):
        if False:
            return 10
        'A helper function to generate reference pulse schedule for backward direction.'
        duration = self.compute_stretch_duration(u0p_play, theta)
        width = self.compute_stretch_width(u0p_play, theta)
        with builder.build(backend, default_alignment='sequential', default_transpiler_settings={'optimization_level': 0}) as ref_sched:
            builder.call_gate(RZGate(np.pi / 2), qubits=(0,))
            builder.call_gate(SXGate(), qubits=(0,))
            builder.call_gate(RZGate(np.pi / 2), qubits=(0,))
            builder.call_gate(RZGate(np.pi / 2), qubits=(1,))
            builder.call_gate(SXGate(), qubits=(1,))
            builder.call_gate(RZGate(np.pi / 2), qubits=(1,))
            with builder.align_left():
                u0p_params = u0p_play.pulse.parameters
                u0p_params['duration'] = duration
                u0p_params['width'] = width
                builder.play(GaussianSquare(**u0p_params), ControlChannel(0))
                d1p_params = d1p_play.pulse.parameters
                d1p_params['duration'] = duration
                d1p_params['width'] = width
                builder.play(GaussianSquare(**d1p_params), DriveChannel(1))
            builder.x(0)
            with builder.align_left():
                u0m_params = u0m_play.pulse.parameters
                u0m_params['duration'] = duration
                u0m_params['width'] = width
                builder.play(GaussianSquare(**u0m_params), ControlChannel(0))
                d1m_params = d1m_play.pulse.parameters
                d1m_params['duration'] = duration
                d1m_params['width'] = width
                builder.play(GaussianSquare(**d1m_params), DriveChannel(1))
            builder.x(0)
            builder.call_gate(RZGate(np.pi / 2), qubits=(0,))
            builder.call_gate(SXGate(), qubits=(0,))
            builder.call_gate(RZGate(np.pi / 2), qubits=(0,))
            builder.call_gate(RZGate(np.pi / 2), qubits=(1,))
            builder.call_gate(SXGate(), qubits=(1,))
            builder.call_gate(RZGate(np.pi / 2), qubits=(1,))
        return ref_sched

    @data(-np.pi / 4, 0.1, np.pi / 4, np.pi / 2, np.pi)
    def test_rzx_calibration_cr_pulse_stretch(self, theta: float):
        if False:
            while True:
                i = 10
        'Test that cross resonance pulse durations are computed correctly.'
        backend = FakeHanoi()
        inst_map = backend.defaults().instruction_schedule_map
        cr_schedule = inst_map.get('cx', (0, 1))
        with builder.build() as test_sched:
            RZXCalibrationBuilder.rescale_cr_inst(self.u0p_play(cr_schedule), theta)
        self.assertEqual(test_sched.duration, self.compute_stretch_duration(self.u0p_play(cr_schedule), theta))

    @data(-np.pi / 4, 0.1, np.pi / 4, np.pi / 2, np.pi)
    def test_rzx_calibration_rotary_pulse_stretch(self, theta: float):
        if False:
            i = 10
            return i + 15
        'Test that rotary pulse durations are computed correctly.'
        backend = FakeHanoi()
        inst_map = backend.defaults().instruction_schedule_map
        cr_schedule = inst_map.get('cx', (0, 1))
        with builder.build() as test_sched:
            RZXCalibrationBuilder.rescale_cr_inst(self.d1p_play(cr_schedule), theta)
        self.assertEqual(test_sched.duration, self.compute_stretch_duration(self.d1p_play(cr_schedule), theta))

    def test_raise(self):
        if False:
            print('Hello World!')
        'Test that the correct error is raised.'
        theta = np.pi / 4
        qc = circuit.QuantumCircuit(2)
        qc.rzx(theta, 0, 1)
        dag = circuit_to_dag(qc)
        backend = FakeArmonk()
        inst_map = backend.defaults().instruction_schedule_map
        _pass = RZXCalibrationBuilder(inst_map)
        qubit_map = {qubit: i for (i, qubit) in enumerate(dag.qubits)}
        with self.assertRaises(QiskitError):
            for node in dag.gate_nodes():
                qubits = [qubit_map[q] for q in node.qargs]
                _pass.get_calibration(node.op, qubits)

    def test_ecr_cx_forward(self):
        if False:
            for i in range(10):
                print('nop')
        'Test that correct pulse sequence is generated for native CR pair.'
        theta = np.pi / 4
        qc = circuit.QuantumCircuit(2)
        qc.rzx(theta, 0, 1)
        backend = FakeHanoi()
        inst_map = backend.defaults().instruction_schedule_map
        _pass = RZXCalibrationBuilder(inst_map)
        test_qc = PassManager(_pass).run(qc)
        cr_schedule = inst_map.get('cx', (0, 1))
        ref_sched = self.build_forward(backend, theta, self.u0p_play(cr_schedule), self.d1p_play(cr_schedule), self.u0m_play(cr_schedule), self.d1m_play(cr_schedule))
        self.assertEqual(schedule(test_qc, backend), target_qobj_transform(ref_sched))

    def test_ecr_cx_reverse(self):
        if False:
            return 10
        'Test that correct pulse sequence is generated for non-native CR pair.'
        theta = np.pi / 4
        qc = circuit.QuantumCircuit(2)
        qc.rzx(theta, 1, 0)
        backend = FakeHanoi()
        inst_map = backend.defaults().instruction_schedule_map
        _pass = RZXCalibrationBuilder(inst_map)
        test_qc = PassManager(_pass).run(qc)
        cr_schedule = inst_map.get('cx', (0, 1))
        ref_sched = self.build_reverse(backend, theta, self.u0p_play(cr_schedule), self.d1p_play(cr_schedule), self.u0m_play(cr_schedule), self.d1m_play(cr_schedule))
        self.assertEqual(schedule(test_qc, backend), target_qobj_transform(ref_sched))

    def test_pass_alive_with_dcx_ish(self):
        if False:
            return 10
        'Test if the pass is not terminated by error with direct CX input.'
        cx_sched = Schedule()
        cx_sched.insert(0, Play(GaussianSquare(800, 0.2, 64, 544), ControlChannel(1)), inplace=True)
        compensation_tone = Waveform(0.1 * np.ones(800, dtype=complex))
        cx_sched.insert(0, Play(compensation_tone, DriveChannel(0)), inplace=True)
        inst_map = InstructionScheduleMap()
        inst_map.add('cx', (1, 0), schedule=cx_sched)
        theta = pi / 3
        rzx_qc = circuit.QuantumCircuit(2)
        rzx_qc.rzx(theta, 1, 0)
        pass_ = RZXCalibrationBuilder(instruction_schedule_map=inst_map)
        with self.assertWarns(UserWarning):
            cal_qc = PassManager(pass_).run(rzx_qc)
        self.assertEqual(cal_qc, rzx_qc)

class TestRZXCalibrationBuilderNoEcho(TestCalibrationBuilder):
    """Test RZXCalibrationBuilderNoEcho."""

    def build_forward(self, theta, u0p_play, d1p_play):
        if False:
            i = 10
            return i + 15
        'A helper function to generate reference pulse schedule for forward direction.'
        duration = self.compute_stretch_duration(u0p_play, 2.0 * theta)
        width = self.compute_stretch_width(u0p_play, 2.0 * theta)
        with builder.build() as ref_sched:
            u0p_params = u0p_play.pulse.parameters
            u0p_params['duration'] = duration
            u0p_params['width'] = width
            builder.play(GaussianSquare(**u0p_params), ControlChannel(0))
            d1p_params = d1p_play.pulse.parameters
            d1p_params['duration'] = duration
            d1p_params['width'] = width
            builder.play(GaussianSquare(**d1p_params), DriveChannel(1))
            builder.delay(duration, DriveChannel(0))
        return ref_sched

    def test_ecr_cx_forward(self):
        if False:
            i = 10
            return i + 15
        'Test that correct pulse sequence is generated for native CR pair.\n\n        .. notes::\n            No echo builder only supports native direction.\n        '
        theta = np.pi / 4
        qc = circuit.QuantumCircuit(2)
        qc.rzx(theta, 0, 1)
        backend = FakeHanoi()
        inst_map = backend.defaults().instruction_schedule_map
        _pass = RZXCalibrationBuilderNoEcho(inst_map)
        test_qc = PassManager(_pass).run(qc)
        cr_schedule = inst_map.get('cx', (0, 1))
        ref_sched = self.build_forward(theta, self.u0p_play(cr_schedule), self.d1p_play(cr_schedule))
        self.assertEqual(schedule(test_qc, backend), target_qobj_transform(ref_sched))

    def test_pass_alive_with_dcx_ish(self):
        if False:
            print('Hello World!')
        'Test if the pass is not terminated by error with direct CX input.'
        cx_sched = Schedule()
        cx_sched.insert(0, Play(GaussianSquare(800, 0.2, 64, 544), ControlChannel(1)), inplace=True)
        compensation_tone = Waveform(0.1 * np.ones(800, dtype=complex))
        cx_sched.insert(0, Play(compensation_tone, DriveChannel(0)), inplace=True)
        inst_map = InstructionScheduleMap()
        inst_map.add('cx', (1, 0), schedule=cx_sched)
        theta = pi / 3
        rzx_qc = circuit.QuantumCircuit(2)
        rzx_qc.rzx(theta, 1, 0)
        pass_ = RZXCalibrationBuilderNoEcho(instruction_schedule_map=inst_map)
        with self.assertWarns(UserWarning):
            cal_qc = PassManager(pass_).run(rzx_qc)
        self.assertEqual(cal_qc, rzx_qc)

@ddt
class TestRXCalibrationBuilder(QiskitTestCase):
    """Test RXCalibrationBuilder."""

    def compute_correct_rx_amplitude(self, rx_theta: float, sx_amp: float):
        if False:
            i = 10
            return i + 15
        'A helper function to compute the amplitude of the bootstrapped RX pulse.'
        return sx_amp * (np.abs(rx_theta) / (0.5 * np.pi))

    def test_not_supported_if_no_sx_schedule(self):
        if False:
            return 10
        'Test that supported() returns False when the target does not have SX calibration.'
        empty_target = Target()
        tp = RXCalibrationBuilder(empty_target)
        qubits = (0,)
        node_op = DAGOpNode(RXGate(0.5), qubits, [])
        self.assertFalse(tp.supported(node_op, qubits))

    def test_not_supported_if_sx_not_drag(self):
        if False:
            return 10
        'Test that supported() returns False when the default SX calibration is not a DRAG.'
        target = Target()
        with builder.build() as square_sx_cal:
            builder.play(Square(amp=0.1, duration=160, phase=0), DriveChannel(0))
        target.add_instruction(SXGate(), {(0,): InstructionProperties(calibration=square_sx_cal)})
        tp = RXCalibrationBuilder(target)
        qubits = (0,)
        node_op = DAGOpNode(RXGate(0.5), qubits, [])
        self.assertFalse(tp.supported(node_op, qubits))

    def test_raises_error_when_rotation_angle_not_assigned(self):
        if False:
            while True:
                i = 10
        "Test that get_calibration() fails when the RX gate's rotation angle is\n        an unassigned Parameter, not a number.\n        The QiskitError occurs while trying to typecast the Parameter into a float.\n        "
        backend = FakeBelemV2()
        tp = RXCalibrationBuilder(backend.target)
        qubits = (0,)
        rx = RXGate(Parameter('theta'))
        with self.assertRaises(QiskitError):
            tp.get_calibration(rx, qubits)

    @data(0, np.pi / 3, 2 / 3 * np.pi)
    def test_pulse_schedule(self, theta: float):
        if False:
            while True:
                i = 10
        'Test that get_calibration() returns a schedule with correct amplitude.'
        backend = FakeBelemV2()
        dummy_target = Target()
        (sx_amp, sx_beta, sx_sigma, sx_duration, sx_angle) = (0.6, 2, 40, 160, 0.5)
        with builder.build(backend=backend) as dummy_sx_cal:
            builder.play(Drag(amp=sx_amp, beta=sx_beta, sigma=sx_sigma, duration=sx_duration, angle=sx_angle), DriveChannel(0))
        dummy_target.add_instruction(SXGate(), {(0,): InstructionProperties(calibration=dummy_sx_cal)})
        tp = RXCalibrationBuilder(dummy_target)
        test = tp.get_calibration(RXGate(theta), qubits=(0,))
        with builder.build(backend=backend) as correct_rx_schedule:
            builder.play(Drag(amp=self.compute_correct_rx_amplitude(rx_theta=theta, sx_amp=sx_amp), beta=sx_beta, sigma=sx_sigma, duration=sx_duration, angle=0), channel=DriveChannel(0))
        self.assertEqual(test, correct_rx_schedule)

    def test_with_normalizerxangles(self):
        if False:
            i = 10
            return i + 15
        'Checks that this pass works well with the NormalizeRXAngles pass.'
        backend = FakeBelemV2()
        pm = PassManager(RXCalibrationBuilder(backend.target))
        qc = QuantumCircuit(1)
        qc.rx(np.pi / 3, 0)
        qc.rx(np.pi / 2, 0)
        qc.rx(np.pi, 0)
        tc = pm.run(qc)
        self.assertEqual(len(tc.calibrations['rx']), 1)