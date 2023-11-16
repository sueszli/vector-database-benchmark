import numpy as np
from qiskit.circuit import Parameter, QuantumCircuit, Gate
from qiskit.pulse import builder, library, channels

class EchoedCrossResonanceConstructionBench:

    def setup(self):
        if False:
            while True:
                i = 10
        with builder.build() as x_ctrl:
            builder.play(library.Drag(160, 0.2, 40, 1.5), channels.DriveChannel(0))
        self.x_ctrl = x_ctrl
        with builder.build() as cr45p:
            builder.play(library.GaussianSquare(800, 0.4, 64, 544), channels.ControlChannel(0))
            builder.play(library.GaussianSquare(800, 0.1, 64, 544), channels.DriveChannel(1))
        self.cr45p = cr45p

    def time_full_scratch(self):
        if False:
            for i in range(10):
                print('nop')
        with builder.build():
            with builder.align_sequential():
                with builder.align_left():
                    builder.play(library.GaussianSquare(800, 0.4, 64, 544), channels.ControlChannel(0))
                    builder.play(library.GaussianSquare(800, 0.1, 64, 544), channels.DriveChannel(1))
                builder.play(library.Drag(160, 0.2, 40, 1.5), channels.DriveChannel(0))
                with builder.phase_offset(np.pi, channels.ControlChannel(0), channels.DriveChannel(1)):
                    with builder.align_left():
                        builder.play(library.GaussianSquare(800, 0.4, 64, 544), channels.ControlChannel(0))
                        builder.play(library.GaussianSquare(800, 0.1, 64, 544), channels.DriveChannel(1))
                builder.play(library.Drag(160, 0.2, 40, 1.5), channels.DriveChannel(0))

    def time_with_call(self):
        if False:
            print('Hello World!')
        with builder.build():
            with builder.align_sequential():
                builder.call(self.cr45p)
                builder.call(self.x_ctrl)
                with builder.phase_offset(np.pi, channels.ControlChannel(0), channels.DriveChannel(1)):
                    builder.call(self.cr45p)
                builder.call(self.x_ctrl)

    def time_assign_later(self):
        if False:
            return 10
        with builder.build() as temp_sched:
            with builder.align_sequential():
                builder.reference('cr45p', 'q0', 'q1')
                builder.reference('x', 'q0')
                with builder.phase_offset(np.pi, channels.ControlChannel(0), channels.DriveChannel(1)):
                    builder.reference('cr45p', 'q0', 'q1')
                builder.reference('x', 'q0')
        temp_sched.assign_references({('cr45p', 'q0', 'q1'): self.cr45p, ('x', 'q0'): self.x_ctrl}, inplace=True)

class ParameterizedScheduleBench:
    params = [3, 11, 31, 51]

    def setup(self, nscan):
        if False:
            while True:
                i = 10
        self.p0 = Parameter('P0')
        self.p1 = Parameter('P1')
        self.p2 = Parameter('P2')
        with builder.build() as schedule:
            builder.play(library.Constant(self.p0, self.p1), channels.DriveChannel(self.p2))
        self.schedule = schedule
        with builder.build() as outer_schedule:
            builder.reference('subroutine')
        outer_schedule.assign_references({('subroutine',): schedule}, inplace=True)
        self.outer_schedule = outer_schedule
        gate = Gate('my_gate', 1, [self.p0, self.p1, self.p2])
        qc = QuantumCircuit(1)
        qc.append(gate, [0])
        qc.add_calibration(gate, (0,), schedule)
        self.qc = qc
        self.amps = np.linspace(-1, 1, nscan)

    def time_assign_single_schedule(self, _):
        if False:
            while True:
                i = 10
        out = []
        for amp in self.amps:
            assigned = self.schedule.assign_parameters({self.p0: 100, self.p1: amp, self.p2: 0}, inplace=False)
            out.append(assigned)

    def time_assign_parameterized_subroutine(self, _):
        if False:
            print('Hello World!')
        out = []
        for amp in self.amps:
            assigned = self.outer_schedule.assign_parameters({self.p0: 100, self.p1: amp, self.p2: 0}, inplace=False)
            out.append(assigned)

    def time_assign_through_pulse_gate(self, _):
        if False:
            return 10
        out = []
        for amp in self.amps:
            assigned = self.qc.assign_parameters({self.p0: 100, self.p1: amp, self.p2: 0}, inplace=False)
            out.append(assigned)