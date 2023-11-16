"""Test schedule block subroutine reference mechanism."""
import numpy as np
from qiskit import circuit, pulse
from qiskit.pulse import builder
from qiskit.pulse.transforms import inline_subroutines
from qiskit.test import QiskitTestCase

class TestReference(QiskitTestCase):
    """Test for basic behavior of reference mechanism."""

    def test_append_schedule(self):
        if False:
            while True:
                i = 10
        'Test appending schedule without calling.\n\n        Appended schedules are not subroutines.\n        These are directly exposed to the outer block.\n        '
        with pulse.build() as sched_x1:
            pulse.play(pulse.Constant(100, 0.1, name='x1'), pulse.DriveChannel(0))
        with pulse.build() as sched_y1:
            builder.append_schedule(sched_x1)
        with pulse.build() as sched_z1:
            builder.append_schedule(sched_y1)
        self.assertEqual(len(sched_z1.references), 0)

    def test_append_schedule_parameter_scope(self):
        if False:
            for i in range(10):
                print('nop')
        'Test appending schedule without calling.\n\n        Parameter in the appended schedule has the scope of outer schedule.\n        '
        param = circuit.Parameter('name')
        with pulse.build() as sched_x1:
            pulse.play(pulse.Constant(100, param, name='x1'), pulse.DriveChannel(0))
        with pulse.build() as sched_y1:
            builder.append_schedule(sched_x1)
        with pulse.build() as sched_z1:
            builder.append_schedule(sched_y1)
        sched_param = next(iter(sched_z1.scoped_parameters()))
        self.assertEqual(sched_param.name, 'root::name')
        self.assertEqual(sched_z1.search_parameters('root::name')[0], param)

    def test_refer_schedule(self):
        if False:
            return 10
        'Test refer to schedule by name.\n\n        Outer block is only aware of its inner reference.\n        Nested reference is not directly exposed to the most outer block.\n        '
        with pulse.build() as sched_x1:
            pulse.play(pulse.Constant(100, 0.1, name='x1'), pulse.DriveChannel(0))
        with pulse.build() as sched_y1:
            builder.reference('x1', 'd0')
        with pulse.build() as sched_z1:
            builder.reference('y1', 'd0')
        sched_y1.assign_references({('x1', 'd0'): sched_x1})
        sched_z1.assign_references({('y1', 'd0'): sched_y1})
        self.assertEqual(len(sched_z1.references), 1)
        self.assertEqual(sched_z1.references['y1', 'd0'], sched_y1)
        self.assertEqual(len(sched_y1.references), 1)
        self.assertEqual(sched_y1.references['x1', 'd0'], sched_x1)

    def test_refer_schedule_parameter_scope(self):
        if False:
            i = 10
            return i + 15
        'Test refer to schedule by name.\n\n        Parameter in the called schedule has the scope of called schedule.\n        '
        param = circuit.Parameter('name')
        with pulse.build() as sched_x1:
            pulse.play(pulse.Constant(100, param, name='x1'), pulse.DriveChannel(0))
        with pulse.build() as sched_y1:
            builder.reference('x1', 'd0')
        with pulse.build() as sched_z1:
            builder.reference('y1', 'd0')
        sched_y1.assign_references({('x1', 'd0'): sched_x1})
        sched_z1.assign_references({('y1', 'd0'): sched_y1})
        sched_param = next(iter(sched_z1.scoped_parameters()))
        self.assertEqual(sched_param.name, 'root::y1,d0::x1,d0::name')
        self.assertEqual(sched_z1.search_parameters('root::y1,d0::x1,d0::name')[0], param)
        self.assertEqual(sched_z1.search_parameters('\\S::x1,d0::name')[0], param)

    def test_call_schedule(self):
        if False:
            print('Hello World!')
        'Test call schedule.\n\n        Outer block is only aware of its inner reference.\n        Nested reference is not directly exposed to the most outer block.\n        '
        with pulse.build() as sched_x1:
            pulse.play(pulse.Constant(100, 0.1, name='x1'), pulse.DriveChannel(0))
        with pulse.build() as sched_y1:
            builder.call(sched_x1, name='x1')
        with pulse.build() as sched_z1:
            builder.call(sched_y1, name='y1')
        self.assertEqual(len(sched_z1.references), 1)
        self.assertEqual(sched_z1.references['y1',], sched_y1)
        self.assertEqual(len(sched_y1.references), 1)
        self.assertEqual(sched_y1.references['x1',], sched_x1)

    def test_call_schedule_parameter_scope(self):
        if False:
            while True:
                i = 10
        'Test call schedule.\n\n        Parameter in the called schedule has the scope of called schedule.\n        '
        param = circuit.Parameter('name')
        with pulse.build() as sched_x1:
            pulse.play(pulse.Constant(100, param, name='x1'), pulse.DriveChannel(0))
        with pulse.build() as sched_y1:
            builder.call(sched_x1, name='x1')
        with pulse.build() as sched_z1:
            builder.call(sched_y1, name='y1')
        sched_param = next(iter(sched_z1.scoped_parameters()))
        self.assertEqual(sched_param.name, 'root::y1::x1::name')
        self.assertEqual(sched_z1.search_parameters('root::y1::x1::name')[0], param)
        self.assertEqual(sched_z1.search_parameters('\\S::x1::name')[0], param)

    def test_append_and_call_schedule(self):
        if False:
            i = 10
            return i + 15
        'Test call and append schedule.\n\n        Reference is copied to the outer schedule by appending.\n        Original reference remains unchanged.\n        '
        with pulse.build() as sched_x1:
            pulse.play(pulse.Constant(100, 0.1, name='x1'), pulse.DriveChannel(0))
        with pulse.build() as sched_y1:
            builder.call(sched_x1, name='x1')
        with pulse.build() as sched_z1:
            builder.append_schedule(sched_y1)
        self.assertEqual(len(sched_z1.references), 1)
        self.assertEqual(sched_z1.references['x1',], sched_x1)
        self.assertIs(sched_z1.blocks[0].references, sched_z1.references)
        self.assertIsNot(sched_y1.references, sched_z1.references)
        self.assertEqual(len(sched_y1.references), 1)
        self.assertEqual(sched_y1.references['x1',], sched_x1)

    def test_calling_similar_schedule(self):
        if False:
            for i in range(10):
                print('nop')
        'Test calling schedules with the same representation.\n\n        sched_x1 and sched_y1 are the different subroutines, but same representation.\n        Two references shoud be created.\n        '
        param1 = circuit.Parameter('param')
        param2 = circuit.Parameter('param')
        with pulse.build() as sched_x1:
            pulse.play(pulse.Constant(100, param1, name='p'), pulse.DriveChannel(0))
        with pulse.build() as sched_y1:
            pulse.play(pulse.Constant(100, param2, name='p'), pulse.DriveChannel(0))
        with pulse.build() as sched_z1:
            pulse.call(sched_x1)
            pulse.call(sched_y1)
        self.assertEqual(len(sched_z1.references), 2)

    def test_calling_same_schedule(self):
        if False:
            i = 10
            return i + 15
        'Test calling same schedule twice.\n\n        Because it calls the same schedule, no duplication should occur in reference table.\n        '
        param = circuit.Parameter('param')
        with pulse.build() as sched_x1:
            pulse.play(pulse.Constant(100, param, name='x1'), pulse.DriveChannel(0))
        with pulse.build() as sched_z1:
            pulse.call(sched_x1, name='same_sched')
            pulse.call(sched_x1, name='same_sched')
        self.assertEqual(len(sched_z1.references), 1)

    def test_calling_same_schedule_with_different_assignment(self):
        if False:
            return 10
        'Test calling same schedule twice but with different parameters.\n\n        Same schedule is called twice but with different assignment.\n        Two references should be created.\n        '
        param = circuit.Parameter('param')
        with pulse.build() as sched_x1:
            pulse.play(pulse.Constant(100, param, name='x1'), pulse.DriveChannel(0))
        with pulse.build() as sched_z1:
            pulse.call(sched_x1, param=0.1)
            pulse.call(sched_x1, param=0.2)
        self.assertEqual(len(sched_z1.references), 2)

    def test_alignment_context(self):
        if False:
            i = 10
            return i + 15
        'Test nested alignment context.\n\n        Inline alignment is identical to append_schedule operation.\n        Thus scope is not newly generated.\n        '
        with pulse.build(name='x1') as sched_x1:
            with pulse.align_right():
                with pulse.align_left():
                    pulse.play(pulse.Constant(100, 0.1, name='x1'), pulse.DriveChannel(0))
        self.assertEqual(len(sched_x1.references), 0)

    def test_appending_child_block(self):
        if False:
            print('Hello World!')
        "Test for edge case.\n\n        User can append blocks which is an element of another schedule block.\n        But this is not standard use case.\n\n        In this case, references may contain subroutines which don't exist in the context.\n        This is because all references within the program are centrally\n        managed in the most outer block.\n        "
        with pulse.build() as sched_x1:
            pulse.play(pulse.Constant(100, 0.1, name='x1'), pulse.DriveChannel(0))
        with pulse.build() as sched_y1:
            pulse.play(pulse.Constant(100, 0.2, name='y1'), pulse.DriveChannel(0))
        with pulse.build() as sched_x2:
            builder.call(sched_x1, name='x1')
        self.assertEqual(list(sched_x2.references.keys()), [('x1',)])
        with pulse.build() as sched_y2:
            builder.call(sched_y1, name='y1')
        self.assertEqual(list(sched_y2.references.keys()), [('y1',)])
        with pulse.build() as sched_z1:
            builder.append_schedule(sched_x2)
            builder.append_schedule(sched_y2)
        self.assertEqual(list(sched_z1.references.keys()), [('x1',), ('y1',)])
        self.assertIs(sched_z1.blocks[0].references, sched_z1._reference_manager)
        self.assertIs(sched_z1.blocks[1].references, sched_z1._reference_manager)
        with pulse.build() as sched_z2:
            builder.append_schedule(sched_z1.blocks[0])
        self.assertEqual(len(sched_z2.references), 1)
        self.assertEqual(sched_z2.references['x1',], sched_x1)

    def test_replacement(self):
        if False:
            i = 10
            return i + 15
        'Test nested alignment context.\n\n        Replacing schedule block with schedule block.\n        Removed block contains own reference, that should be removed with replacement.\n        New block also contains reference, that should be passed to the current reference.\n        '
        with pulse.build() as sched_x1:
            pulse.play(pulse.Constant(100, 0.1, name='x1'), pulse.DriveChannel(0))
        with pulse.build() as sched_y1:
            pulse.play(pulse.Constant(100, 0.2, name='y1'), pulse.DriveChannel(0))
        with pulse.build() as sched_x2:
            builder.call(sched_x1, name='x1')
        with pulse.build() as sched_y2:
            builder.call(sched_y1, name='y1')
        with pulse.build() as sched_z1:
            builder.append_schedule(sched_x2)
            builder.append_schedule(sched_y2)
        self.assertEqual(len(sched_z1.references), 2)
        self.assertEqual(sched_z1.references['x1',], sched_x1)
        self.assertEqual(sched_z1.references['y1',], sched_y1)
        with pulse.build() as sched_r1:
            pulse.play(pulse.Constant(100, 0.1, name='r1'), pulse.DriveChannel(0))
        with pulse.build() as sched_r2:
            pulse.call(sched_r1, name='r1')
        sched_z2 = sched_z1.replace(sched_x2, sched_r2)
        self.assertEqual(len(sched_z2.references), 2)
        self.assertEqual(sched_z2.references['r1',], sched_r1)
        self.assertEqual(sched_z2.references['y1',], sched_y1)

    def test_special_parameter_name(self):
        if False:
            return 10
        'Testcase to guarantee usage of some special symbols in parameter name.\n\n        These symbols might be often used in user code.\n        No conflict should occur with the default scope delimiter.\n        '
        param = circuit.Parameter('my.parameter_object')
        with pulse.build() as sched_x1:
            pulse.play(pulse.Constant(100, param, name='x1'), pulse.DriveChannel(0))
        with pulse.build() as sched_y1:
            pulse.reference('sub', 'q0')
        sched_y1.assign_references({('sub', 'q0'): sched_x1})
        ret_param = sched_y1.search_parameters('\\Ssub,q0::my.parameter_object')[0]
        self.assertEqual(param, ret_param)

    def test_parameter_in_multiple_scope(self):
        if False:
            print('Hello World!')
        'Testcase for scope-aware parameter getter.\n\n        When a single parameter object is used in multiple scopes,\n        the scoped_parameters method must return parameter objects associated to each scope,\n        while parameters property returns a single parameter object.\n        '
        param = circuit.Parameter('name')
        with pulse.build() as sched_x1:
            pulse.play(pulse.Constant(100, param), pulse.DriveChannel(0))
        with pulse.build() as sched_y1:
            pulse.play(pulse.Constant(100, param), pulse.DriveChannel(1))
        with pulse.build() as sched_z1:
            pulse.call(sched_x1, name='x1')
            pulse.call(sched_y1, name='y1')
        self.assertEqual(len(sched_z1.parameters), 1)
        self.assertEqual(len(sched_z1.scoped_parameters()), 2)
        self.assertEqual(sched_z1.search_parameters('root::x1::name')[0], param)
        self.assertEqual(sched_z1.search_parameters('root::y1::name')[0], param)

    def test_parallel_alignment_equality(self):
        if False:
            while True:
                i = 10
        'Testcase for potential edge case.\n\n        In parallel alignment context, reference instruction is broadcasted to\n        all channels. When new channel is added after reference, this should be\n        connected with reference node.\n        '
        with pulse.build() as subroutine:
            pulse.reference('unassigned')
        with pulse.build() as sched1:
            with pulse.align_left():
                pulse.delay(10, pulse.DriveChannel(0))
                pulse.call(subroutine)
                pulse.delay(10, pulse.DriveChannel(1))
        with pulse.build() as sched2:
            with pulse.align_left():
                pulse.delay(10, pulse.DriveChannel(0))
                pulse.delay(10, pulse.DriveChannel(1))
                pulse.call(subroutine)
        self.assertNotEqual(sched1, sched2)

    def test_subroutine_conflict(self):
        if False:
            print('Hello World!')
        'Test for edge case of appending two schedule blocks having the\n        references with conflicting reference key.\n\n        This operation should fail because one of references will be gone after assignment.\n        '
        with pulse.build() as sched_x1:
            pulse.play(pulse.Constant(100, 0.1), pulse.DriveChannel(0))
        with pulse.build() as sched_x2:
            pulse.call(sched_x1, name='conflict_name')
        self.assertEqual(sched_x2.references['conflict_name',], sched_x1)
        with pulse.build() as sched_y1:
            pulse.play(pulse.Constant(100, 0.2), pulse.DriveChannel(0))
        with pulse.build() as sched_y2:
            pulse.call(sched_y1, name='conflict_name')
        self.assertEqual(sched_y2.references['conflict_name',], sched_y1)
        with self.assertRaises(pulse.exceptions.PulseError):
            with pulse.build():
                builder.append_schedule(sched_x2)
                builder.append_schedule(sched_y2)

    def test_assign_existing_reference(self):
        if False:
            for i in range(10):
                print('nop')
        'Test for explicitly assign existing reference.\n\n        This operation should fail because overriding reference is not allowed.\n        '
        with pulse.build() as sched_x1:
            pulse.play(pulse.Constant(100, 0.1), pulse.DriveChannel(0))
        with pulse.build() as sched_y1:
            pulse.play(pulse.Constant(100, 0.2), pulse.DriveChannel(0))
        with pulse.build() as sched_z1:
            pulse.call(sched_x1, name='conflict_name')
        with self.assertRaises(pulse.exceptions.PulseError):
            sched_z1.assign_references({('conflict_name',): sched_y1})

class TestSubroutineWithCXGate(QiskitTestCase):
    """Test called program scope with practical example of building fully parametrized CX gate."""

    def setUp(self):
        if False:
            while True:
                i = 10
        super().setUp()
        self.xp_dur = circuit.Parameter('dur')
        self.xp_amp = circuit.Parameter('amp')
        self.xp_sigma = circuit.Parameter('sigma')
        self.xp_beta = circuit.Parameter('beta')
        self.sxp_amp = circuit.Parameter('amp')
        self.cr_dur = circuit.Parameter('dur')
        self.cr_amp = circuit.Parameter('amp')
        self.cr_sigma = circuit.Parameter('sigma')
        self.cr_risefall = circuit.Parameter('risefall')
        self.control_ch = circuit.Parameter('ctrl')
        self.target_ch = circuit.Parameter('tgt')
        self.cr_ch = circuit.Parameter('cr')
        with pulse.build(name='xp') as xp_sched_q0:
            pulse.play(pulse.Drag(duration=self.xp_dur, amp=self.xp_amp, sigma=self.xp_sigma, beta=self.xp_beta), channel=pulse.DriveChannel(self.control_ch))
        self.xp_sched = xp_sched_q0
        with pulse.build(name='sx') as sx_sched_q1:
            pulse.play(pulse.Drag(duration=self.xp_dur, amp=self.sxp_amp, sigma=self.xp_sigma, beta=self.xp_beta), channel=pulse.DriveChannel(self.target_ch))
        self.sx_sched = sx_sched_q1
        with pulse.build(name='cr') as cr_sched:
            pulse.play(pulse.GaussianSquare(duration=self.cr_dur, amp=self.cr_amp, sigma=self.cr_sigma, risefall_sigma_ratio=self.cr_risefall), channel=pulse.ControlChannel(self.cr_ch))
        self.cr_sched = cr_sched

    def test_lazy_ecr(self):
        if False:
            for i in range(10):
                print('nop')
        'Test lazy subroutines through ECR schedule construction.'
        with pulse.build(name='lazy_ecr') as sched:
            with pulse.align_sequential():
                pulse.reference('cr', 'q0', 'q1')
                pulse.reference('xp', 'q0')
                with pulse.phase_offset(np.pi, pulse.ControlChannel(self.cr_ch)):
                    pulse.reference('cr', 'q0', 'q1')
                pulse.reference('xp', 'q0')
        self.assertTrue(sched.is_referenced())
        self.assertFalse(sched.is_schedulable())
        self.assertEqual(len(sched.references), 2)
        params = {p.name for p in sched.parameters}
        self.assertSetEqual(params, {'cr'})
        scoped_params = {p.name for p in sched.scoped_parameters()}
        self.assertSetEqual(scoped_params, {'root::cr'})
        sched.assign_references({('cr', 'q0', 'q1'): self.cr_sched})
        sched.assign_references({('xp', 'q0'): self.xp_sched})
        assigned_refs = sched.references
        self.assertEqual(assigned_refs['cr', 'q0', 'q1'], self.cr_sched)
        self.assertEqual(assigned_refs['xp', 'q0'], self.xp_sched)
        scoped_params = {p.name for p in sched.scoped_parameters()}
        ref_params = {'root::cr', 'root::cr,q0,q1::cr', 'root::cr,q0,q1::amp', 'root::cr,q0,q1::dur', 'root::cr,q0,q1::risefall', 'root::cr,q0,q1::sigma', 'root::xp,q0::ctrl', 'root::xp,q0::amp', 'root::xp,q0::beta', 'root::xp,q0::dur', 'root::xp,q0::sigma'}
        self.assertSetEqual(scoped_params, ref_params)
        params = sched.get_parameters(parameter_name='amp')
        self.assertEqual(len(params), 2)
        params = sched.search_parameters(parameter_regex='root::xp,q0::amp')
        self.assertEqual(len(params), 1)

    def test_cnot(self):
        if False:
            for i in range(10):
                print('nop')
        'Integration test with CNOT schedule construction.'
        with pulse.build(name='ecr', default_alignment='sequential') as ecr_sched:
            pulse.call(self.cr_sched, name='cr')
            pulse.call(self.xp_sched, name='xp')
            with pulse.phase_offset(np.pi, pulse.ControlChannel(self.cr_ch)):
                pulse.call(self.cr_sched, name='cr')
            pulse.call(self.xp_sched, name='xp')
        with pulse.build(name='cx', default_alignment='sequential') as cx_sched:
            pulse.shift_phase(np.pi / 2, pulse.DriveChannel(self.control_ch))
            pulse.call(self.sx_sched, name='sx')
            pulse.call(ecr_sched, name='ecr')
        xp_amp = cx_sched.search_parameters('\\S:xp::amp')[0]
        self.assertEqual(self.xp_amp, xp_amp)
        xp_amp_full_scoped = cx_sched.search_parameters('root::ecr::xp::amp')[0]
        self.assertEqual(xp_amp_full_scoped, xp_amp)
        assigned_cx = cx_sched.assign_parameters(value_dict={self.cr_ch: 0, self.control_ch: 0, self.target_ch: 1, self.sxp_amp: 0.1, self.xp_amp: 0.2, self.xp_dur: 160, self.xp_sigma: 40, self.xp_beta: 3.0, self.cr_amp: 0.5, self.cr_dur: 800, self.cr_sigma: 64, self.cr_risefall: 2}, inplace=True)
        flatten_cx = inline_subroutines(assigned_cx)
        with pulse.build(default_alignment='sequential') as ref_cx:
            pulse.shift_phase(np.pi / 2, pulse.DriveChannel(0))
            with pulse.align_left():
                pulse.play(pulse.Drag(duration=160, amp=0.1, sigma=40, beta=3.0), channel=pulse.DriveChannel(1))
            with pulse.align_sequential():
                with pulse.align_left():
                    pulse.play(pulse.GaussianSquare(duration=800, amp=0.5, sigma=64, risefall_sigma_ratio=2), channel=pulse.ControlChannel(0))
                with pulse.align_left():
                    pulse.play(pulse.Drag(duration=160, amp=0.2, sigma=40, beta=3.0), channel=pulse.DriveChannel(0))
                with pulse.phase_offset(np.pi, pulse.ControlChannel(0)):
                    with pulse.align_left():
                        pulse.play(pulse.GaussianSquare(duration=800, amp=0.5, sigma=64, risefall_sigma_ratio=2), channel=pulse.ControlChannel(0))
                with pulse.align_left():
                    pulse.play(pulse.Drag(duration=160, amp=0.2, sigma=40, beta=3.0), channel=pulse.DriveChannel(0))
        self.assertEqual(flatten_cx, ref_cx)