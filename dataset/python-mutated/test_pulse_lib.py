"""Unit tests for pulse waveforms."""
import unittest
from unittest.mock import patch
import numpy as np
from qiskit.circuit import Parameter
from qiskit.pulse.library import SymbolicPulse, Waveform, Constant, Gaussian, GaussianSquare, GaussianSquareDrag, gaussian_square_echo, GaussianDeriv, Drag, Sin, Cos, Sawtooth, Triangle, Square, Sech, SechDeriv, gaussian, gaussian_square, gaussian_deriv, drag as pl_drag, sin, cos, triangle, sawtooth, square, sech, sech_deriv
from qiskit.pulse import functional_pulse, PulseError
from qiskit.test import QiskitTestCase
from qiskit.utils import optionals as _optional
if _optional.HAS_SYMENGINE:
    import symengine as sym
else:
    import sympy as sym

class TestWaveform(QiskitTestCase):
    """Waveform tests."""

    def test_sample_pulse(self):
        if False:
            return 10
        'Test pulse initialization.'
        n_samples = 100
        samples = np.linspace(0, 1.0, n_samples, dtype=np.complex128)
        name = 'test'
        sample_pulse = Waveform(samples, name=name)
        self.assertEqual(sample_pulse.samples.dtype, np.complex128)
        np.testing.assert_almost_equal(sample_pulse.samples, samples)
        self.assertEqual(sample_pulse.duration, n_samples)
        self.assertEqual(sample_pulse.name, name)

    def test_waveform_hashing(self):
        if False:
            return 10
        'Test waveform hashing.'
        n_samples = 100
        samples = np.linspace(0, 1.0, n_samples, dtype=np.complex128)
        name = 'test'
        sample_pulse = Waveform(samples, name=name)
        sample_pulse2 = Waveform(samples, name='test2')
        self.assertEqual({sample_pulse, sample_pulse2}, {sample_pulse})

    def test_type_casting(self):
        if False:
            for i in range(10):
                print('nop')
        'Test casting of input samples to numpy array.'
        n_samples = 100
        samples_f64 = np.linspace(0, 1.0, n_samples, dtype=np.float64)
        sample_pulse_f64 = Waveform(samples_f64)
        self.assertEqual(sample_pulse_f64.samples.dtype, np.complex128)
        samples_c64 = np.linspace(0, 1.0, n_samples, dtype=np.complex64)
        sample_pulse_c64 = Waveform(samples_c64)
        self.assertEqual(sample_pulse_c64.samples.dtype, np.complex128)
        samples_list = np.linspace(0, 1.0, n_samples).tolist()
        sample_pulse_list = Waveform(samples_list)
        self.assertEqual(sample_pulse_list.samples.dtype, np.complex128)

    def test_pulse_limits(self):
        if False:
            for i in range(10):
                print('nop')
        'Test that limits of pulse norm of one are enforced properly.'
        unit_pulse_c128 = np.exp(1j * 2 * np.pi * np.linspace(0, 1, 1000), dtype=np.complex128)
        try:
            Waveform(unit_pulse_c128)
        except PulseError:
            self.fail('Waveform incorrectly failed on approximately unit norm samples.')
        invalid_const = 1.1
        with self.assertRaises(PulseError):
            Waveform(invalid_const * np.exp(1j * 2 * np.pi * np.linspace(0, 1, 1000)))
        with patch('qiskit.pulse.library.pulse.Pulse.limit_amplitude', new=False):
            wave = Waveform(invalid_const * np.exp(1j * 2 * np.pi * np.linspace(0, 1, 1000)))
            self.assertGreater(np.max(np.abs(wave.samples)), 1.0)
        unit_pulse_c64 = np.exp(1j * 2 * np.pi * np.linspace(0, 1, 1000), dtype=np.complex64)
        sample_components = np.stack(np.transpose([np.real(unit_pulse_c64), np.imag(unit_pulse_c64)]))
        pulse_list = sample_components.tolist()
        recombined_pulse = [sample[0] + sample[1] * 1j for sample in pulse_list]
        try:
            Waveform(recombined_pulse)
        except PulseError:
            self.fail('Waveform incorrectly failed to approximately unit norm samples.')

class TestParametricPulses(QiskitTestCase):
    """Tests for all subclasses of ParametricPulse."""

    def test_construction(self):
        if False:
            while True:
                i = 10
        'Test that parametric pulses can be constructed without error.'
        Gaussian(duration=25, sigma=4, amp=0.5, angle=np.pi / 2)
        GaussianSquare(duration=150, amp=0.2, sigma=8, width=140)
        GaussianSquare(duration=150, amp=0.2, sigma=8, risefall_sigma_ratio=2.5)
        Constant(duration=150, amp=0.5, angle=np.pi * 0.23)
        Drag(duration=25, amp=0.6, sigma=7.8, beta=4, angle=np.pi * 0.54)
        GaussianDeriv(duration=150, amp=0.2, sigma=8)
        Sin(duration=25, amp=0.5, freq=0.1, phase=0.5, angle=0.5)
        Cos(duration=30, amp=0.5, freq=0.1, phase=-0.5)
        Sawtooth(duration=40, amp=0.5, freq=0.2, phase=3.14)
        Triangle(duration=50, amp=0.5, freq=0.01, phase=0.5)
        Square(duration=50, amp=0.5, freq=0.01, phase=0.5)
        Sech(duration=50, amp=0.5, sigma=10)
        Sech(duration=50, amp=0.5, sigma=10, zero_ends=False)
        SechDeriv(duration=50, amp=0.5, sigma=10)

    def test_complex_amp_deprecation(self):
        if False:
            print('Hello World!')
        'Test that deprecation warnings and errors are raised for complex amp,\n        and that pulses are equivalent.'
        with self.assertWarns(DeprecationWarning):
            Gaussian(duration=25, sigma=4, amp=0.5j)
        with self.assertWarns(DeprecationWarning):
            GaussianSquare(duration=125, sigma=4, amp=0.5j, width=100)
        with self.assertWarns(DeprecationWarning):
            with self.assertRaises(PulseError):
                Gaussian(duration=25, sigma=4, amp=0.5j, angle=1)
        with self.assertWarns(DeprecationWarning):
            with self.assertRaises(PulseError):
                GaussianSquare(duration=125, sigma=4, amp=0.5j, width=100, angle=0.1)
        with self.assertWarns(DeprecationWarning):
            gauss_pulse_complex_amp = Gaussian(duration=25, sigma=4, amp=0.5j)
        gauss_pulse_amp_angle = Gaussian(duration=25, sigma=4, amp=0.5, angle=np.pi / 2)
        np.testing.assert_almost_equal(gauss_pulse_amp_angle.get_waveform().samples, gauss_pulse_complex_amp.get_waveform().samples)

    def test_gaussian_pulse(self):
        if False:
            while True:
                i = 10
        'Test that Gaussian sample pulse matches the pulse library.'
        gauss = Gaussian(duration=25, sigma=4, amp=0.5, angle=np.pi / 2)
        sample_pulse = gauss.get_waveform()
        self.assertIsInstance(sample_pulse, Waveform)
        with self.assertWarns(DeprecationWarning):
            pulse_lib_gauss = gaussian(duration=25, sigma=4, amp=0.5j, zero_ends=True).samples
        np.testing.assert_almost_equal(sample_pulse.samples, pulse_lib_gauss)

    def test_gaussian_square_pulse(self):
        if False:
            for i in range(10):
                print('nop')
        'Test that GaussianSquare sample pulse matches the pulse library.'
        gauss_sq = GaussianSquare(duration=125, sigma=4, amp=0.5, width=100, angle=np.pi / 2)
        sample_pulse = gauss_sq.get_waveform()
        self.assertIsInstance(sample_pulse, Waveform)
        with self.assertWarns(DeprecationWarning):
            pulse_lib_gauss_sq = gaussian_square(duration=125, sigma=4, amp=0.5j, width=100, zero_ends=True).samples
        np.testing.assert_almost_equal(sample_pulse.samples, pulse_lib_gauss_sq)
        gauss_sq = GaussianSquare(duration=125, sigma=4, amp=0.5, risefall_sigma_ratio=3.125, angle=np.pi / 2)
        sample_pulse = gauss_sq.get_waveform()
        self.assertIsInstance(sample_pulse, Waveform)
        with self.assertWarns(DeprecationWarning):
            pulse_lib_gauss_sq = gaussian_square(duration=125, sigma=4, amp=0.5j, width=100, zero_ends=True).samples
        np.testing.assert_almost_equal(sample_pulse.samples, pulse_lib_gauss_sq)

    def test_gauss_square_extremes(self):
        if False:
            return 10
        'Test that the gaussian square pulse can build a gaussian.'
        duration = 125
        sigma = 4
        amp = 0.5
        angle = np.pi / 2
        gaus_square = GaussianSquare(duration=duration, sigma=sigma, amp=amp, width=0, angle=angle)
        gaus = Gaussian(duration=duration, sigma=sigma, amp=amp, angle=angle)
        np.testing.assert_almost_equal(gaus_square.get_waveform().samples, gaus.get_waveform().samples)
        gaus_square = GaussianSquare(duration=duration, sigma=sigma, amp=amp, width=121, angle=angle)
        const = Constant(duration=duration, amp=amp, angle=angle)
        np.testing.assert_almost_equal(gaus_square.get_waveform().samples[2:-2], const.get_waveform().samples[2:-2])

    def test_gauss_square_passes_validation_after_construction(self):
        if False:
            while True:
                i = 10
        'Test that parameter validation is consistent before and after construction.\n\n        This previously used to raise an exception: see gh-7882.'
        pulse = GaussianSquare(duration=125, sigma=4, amp=0.5, width=100, angle=np.pi / 2)
        pulse.validate_parameters()

    def test_gaussian_square_drag_pulse(self):
        if False:
            for i in range(10):
                print('nop')
        'Test that GaussianSquareDrag sample pulse matches expectations.\n\n        Test that the real part of the envelop matches GaussianSquare and that\n        the rise and fall match Drag.\n        '
        risefall = 32
        sigma = 4
        amp = 0.5
        width = 100
        beta = 1
        duration = width + 2 * risefall
        gsd = GaussianSquareDrag(duration=duration, sigma=sigma, amp=amp, width=width, beta=beta)
        gsd_samples = gsd.get_waveform().samples
        gs_pulse = GaussianSquare(duration=duration, sigma=sigma, amp=amp, width=width)
        np.testing.assert_almost_equal(np.real(gsd_samples), np.real(gs_pulse.get_waveform().samples))
        gsd2 = GaussianSquareDrag(duration=duration, sigma=sigma, amp=amp, beta=beta, risefall_sigma_ratio=risefall / sigma)
        np.testing.assert_almost_equal(gsd_samples, gsd2.get_waveform().samples)
        drag_pulse = Drag(duration=2 * risefall, amp=amp, sigma=sigma, beta=beta)
        np.testing.assert_almost_equal(gsd_samples[:risefall], drag_pulse.get_waveform().samples[:risefall])
        np.testing.assert_almost_equal(gsd_samples[-risefall:], drag_pulse.get_waveform().samples[-risefall:])

    def test_gauss_square_drag_extreme(self):
        if False:
            for i in range(10):
                print('nop')
        'Test that the gaussian square drag pulse can build a drag pulse.'
        duration = 125
        sigma = 4
        amp = 0.5
        angle = 1.5
        beta = 1
        gsd = GaussianSquareDrag(duration=duration, sigma=sigma, amp=amp, width=0, beta=beta, angle=angle)
        drag = Drag(duration=duration, sigma=sigma, amp=amp, beta=beta, angle=angle)
        np.testing.assert_almost_equal(gsd.get_waveform().samples, drag.get_waveform().samples)

    def test_gaussian_square_drag_validation(self):
        if False:
            return 10
        'Test drag beta parameter validation.'
        GaussianSquareDrag(duration=50, width=0, sigma=16, amp=1, beta=2)
        GaussianSquareDrag(duration=50, width=0, sigma=16, amp=1, beta=4)
        GaussianSquareDrag(duration=50, width=0, sigma=16, amp=0.5, beta=20)
        GaussianSquareDrag(duration=50, width=0, sigma=16, amp=-1, beta=2)
        GaussianSquareDrag(duration=50, width=0, sigma=16, amp=1, beta=-2)
        GaussianSquareDrag(duration=50, width=0, sigma=16, amp=1, beta=6)
        GaussianSquareDrag(duration=50, width=0, sigma=16, amp=-0.5, beta=25, angle=1.5)
        with self.assertRaises(PulseError):
            GaussianSquareDrag(duration=50, width=0, sigma=16, amp=1, beta=20)
        with self.assertRaises(PulseError):
            GaussianSquareDrag(duration=50, width=0, sigma=4, amp=0.8, beta=20)
        with self.assertRaises(PulseError):
            GaussianSquareDrag(duration=50, width=0, sigma=4, amp=0.8, beta=-20)

    def test_gaussian_square_echo_pulse(self):
        if False:
            i = 10
            return i + 15
        'Test that gaussian_square_echo sample pulse matches expectations.\n\n        Test that the real part of the envelop matches GaussianSquare with\n        given amplitude and phase active for half duration with another\n        GaussianSquare active for the other half duration with opposite\n        amplitude and a GaussianSquare active on the entire duration with\n        its own amplitude and phase\n        '
        risefall = 32
        sigma = 4
        amp = 0.5
        width = 100
        duration = width + 2 * risefall
        active_amp = 0.1
        width_echo = (duration - 2 * (duration - width)) / 2
        gse = gaussian_square_echo(duration=duration, sigma=sigma, amp=amp, width=width, active_amp=active_amp)
        gse_samples = gse.get_waveform().samples
        gs_echo_pulse_pos = GaussianSquare(duration=duration / 2, sigma=sigma, amp=amp, width=width_echo)
        gs_echo_pulse_neg = GaussianSquare(duration=duration / 2, sigma=sigma, amp=-amp, width=width_echo)
        gs_active_pulse = GaussianSquare(duration=duration, sigma=sigma, amp=active_amp, width=width)
        gs_echo_pulse_pos_samples = np.array(gs_echo_pulse_pos.get_waveform().samples.tolist() + [0] * int(duration / 2))
        gs_echo_pulse_neg_samples = np.array([0] * int(duration / 2) + gs_echo_pulse_neg.get_waveform().samples.tolist())
        gs_active_pulse_samples = gs_active_pulse.get_waveform().samples
        np.testing.assert_almost_equal(gse_samples, gs_echo_pulse_pos_samples + gs_echo_pulse_neg_samples + gs_active_pulse_samples)

    def test_gaussian_square_echo_active_amp_validation(self):
        if False:
            i = 10
            return i + 15
        'Test gaussian square echo active amp parameter validation.'
        gaussian_square_echo(duration=50, width=0, sigma=16, amp=0.1, active_amp=0.2)
        gaussian_square_echo(duration=50, width=0, sigma=16, amp=0.1, active_amp=0.4)
        gaussian_square_echo(duration=50, width=0, sigma=16, amp=0.5, active_amp=0.3)
        gaussian_square_echo(duration=50, width=0, sigma=16, amp=-0.1, active_amp=0.2)
        gaussian_square_echo(duration=50, width=0, sigma=16, amp=0.1, active_amp=-0.2)
        gaussian_square_echo(duration=50, width=0, sigma=16, amp=0.1, active_amp=0.6)
        gaussian_square_echo(duration=50, width=0, sigma=16, amp=-0.5, angle=1.5, active_amp=0.25)
        with self.assertRaises(PulseError):
            gaussian_square_echo(duration=50, width=0, sigma=16, amp=0.1, active_amp=1.1)
        with self.assertRaises(PulseError):
            gaussian_square_echo(duration=50, width=0, sigma=4, amp=-0.8, active_amp=-0.3)

    def test_drag_pulse(self):
        if False:
            while True:
                i = 10
        'Test that the Drag sample pulse matches the pulse library.'
        drag = Drag(duration=25, sigma=4, amp=0.5, beta=1, angle=np.pi / 2)
        sample_pulse = drag.get_waveform()
        self.assertIsInstance(sample_pulse, Waveform)
        with self.assertWarns(DeprecationWarning):
            pulse_lib_drag = pl_drag(duration=25, sigma=4, amp=0.5j, beta=1, zero_ends=True).samples
        np.testing.assert_almost_equal(sample_pulse.samples, pulse_lib_drag)

    def test_drag_validation(self):
        if False:
            i = 10
            return i + 15
        'Test drag parameter validation, specifically the beta validation.'
        duration = 25
        sigma = 4
        amp = 0.5
        angle = np.pi / 2
        beta = 1
        wf = Drag(duration=duration, sigma=sigma, amp=amp, beta=beta, angle=angle)
        samples = wf.get_waveform().samples
        self.assertTrue(max(np.abs(samples)) <= 1)
        with self.assertRaises(PulseError):
            wf = Drag(duration=duration, sigma=sigma, amp=1.2, beta=beta)
        beta = sigma ** 2
        with self.assertRaises(PulseError):
            wf = Drag(duration=duration, sigma=sigma, amp=amp, beta=beta, angle=angle)
        sigma = 100
        wf = Drag(duration=duration, sigma=sigma, amp=amp, beta=beta, angle=angle)

    def test_drag_beta_validation(self):
        if False:
            return 10
        'Test drag beta parameter validation.'

        def check_drag(duration, sigma, amp, beta, angle=0):
            if False:
                print('Hello World!')
            wf = Drag(duration=duration, sigma=sigma, amp=amp, beta=beta, angle=angle)
            samples = wf.get_waveform().samples
            self.assertTrue(max(np.abs(samples)) <= 1)
        check_drag(duration=50, sigma=16, amp=1, beta=2)
        check_drag(duration=50, sigma=16, amp=1, beta=4)
        check_drag(duration=50, sigma=16, amp=0.5, beta=20)
        check_drag(duration=50, sigma=16, amp=-1, beta=2)
        check_drag(duration=50, sigma=16, amp=1, beta=-2)
        check_drag(duration=50, sigma=16, amp=1, beta=6)
        check_drag(duration=50, sigma=16, amp=0.5, beta=25, angle=-np.pi / 2)
        with self.assertRaises(PulseError):
            check_drag(duration=50, sigma=16, amp=1, beta=20)
        with self.assertRaises(PulseError):
            check_drag(duration=50, sigma=4, amp=0.8, beta=20)
        with self.assertRaises(PulseError):
            check_drag(duration=50, sigma=4, amp=0.8, beta=-20)

    def test_sin_pulse(self):
        if False:
            return 10
        'Test that Sin sample pulse matches expectations, and parameter validation'
        duration = 100
        amp = 0.5
        freq = 0.1
        phase = 0
        sin_pulse = Sin(duration=duration, amp=amp, freq=freq, phase=phase)
        with self.assertWarns(DeprecationWarning):
            sin_waveform = sin(duration=duration, amp=amp, freq=freq, phase=phase)
        np.testing.assert_almost_equal(sin_pulse.get_waveform().samples, sin_waveform.samples)
        with self.assertRaises(PulseError):
            Sin(duration=duration, amp=amp, freq=5, phase=phase)

    def test_cos_pulse(self):
        if False:
            while True:
                i = 10
        'Test that Cos sample pulse matches expectations, and parameter validation'
        duration = 100
        amp = 0.5
        freq = 0.1
        phase = 0
        cos_pulse = Cos(duration=duration, amp=amp, freq=freq, phase=phase)
        with self.assertWarns(DeprecationWarning):
            cos_waveform = cos(duration=duration, amp=amp, freq=freq, phase=phase)
        np.testing.assert_almost_equal(cos_pulse.get_waveform().samples, cos_waveform.samples)
        shifted_sin_pulse = Sin(duration=duration, amp=amp, freq=freq, phase=phase + np.pi / 2)
        np.testing.assert_almost_equal(shifted_sin_pulse.get_waveform().samples, cos_pulse.get_waveform().samples)
        with self.assertRaises(PulseError):
            Cos(duration=duration, amp=amp, freq=5, phase=phase)

    def test_square_pulse(self):
        if False:
            return 10
        'Test that Square sample pulse matches expectations, and parameter validation'
        duration = 100
        amp = 0.5
        freq = 0.1
        phase = 0.3
        square_pulse = Square(duration=duration, amp=amp, freq=freq, phase=phase)
        with self.assertWarns(DeprecationWarning):
            square_waveform = square(duration=duration, amp=amp, freq=freq, phase=phase / 2)
        np.testing.assert_almost_equal(square_pulse.get_waveform().samples, square_waveform.samples)
        with self.assertRaises(PulseError):
            Square(duration=duration, amp=amp, freq=5, phase=phase)

    def test_sawtooth_pulse(self):
        if False:
            i = 10
            return i + 15
        'Test that Sawtooth sample pulse matches expectations, and parameter validation'
        duration = 100
        amp = 0.5
        freq = 0.1
        phase = 0.5
        sawtooth_pulse = Sawtooth(duration=duration, amp=amp, freq=freq, phase=phase)
        with self.assertWarns(DeprecationWarning):
            sawtooth_waveform = sawtooth(duration=duration, amp=amp, freq=freq, phase=phase / 2)
        np.testing.assert_almost_equal(sawtooth_pulse.get_waveform().samples, sawtooth_waveform.samples)
        sawtooth_pulse_2 = Sawtooth(duration=duration, amp=amp, freq=freq, phase=phase + 2 * np.pi)
        np.testing.assert_almost_equal(sawtooth_pulse.get_waveform().samples, sawtooth_pulse_2.get_waveform().samples)
        with self.assertRaises(PulseError):
            Sawtooth(duration=duration, amp=amp, freq=5, phase=phase)

    def test_triangle_pulse(self):
        if False:
            print('Hello World!')
        'Test that Triangle sample pulse matches expectations, and parameter validation'
        duration = 100
        amp = 0.5
        freq = 0.1
        phase = 0.5
        triangle_pulse = Triangle(duration=duration, amp=amp, freq=freq, phase=phase)
        with self.assertWarns(DeprecationWarning):
            triangle_waveform = triangle(duration=duration, amp=amp, freq=freq, phase=phase)
        np.testing.assert_almost_equal(triangle_pulse.get_waveform().samples, triangle_waveform.samples)
        triangle_pulse_2 = Triangle(duration=duration, amp=amp, freq=freq, phase=phase + 2 * np.pi)
        np.testing.assert_almost_equal(triangle_pulse.get_waveform().samples, triangle_pulse_2.get_waveform().samples)
        with self.assertRaises(PulseError):
            Triangle(duration=duration, amp=amp, freq=5, phase=phase)

    def test_gaussian_deriv_pulse(self):
        if False:
            i = 10
            return i + 15
        'Test that GaussianDeriv sample pulse matches expectations'
        duration = 300
        amp = 0.5
        sigma = 100
        gaussian_deriv_pulse = GaussianDeriv(duration=duration, amp=amp, sigma=sigma)
        with self.assertWarns(DeprecationWarning):
            gaussian_deriv_waveform = gaussian_deriv(duration=duration, amp=amp, sigma=sigma)
        np.testing.assert_almost_equal(gaussian_deriv_pulse.get_waveform().samples, gaussian_deriv_waveform.samples)
        with self.assertRaises(PulseError):
            Sech(duration=duration, amp=amp, sigma=0)

    def test_sech_pulse(self):
        if False:
            while True:
                i = 10
        'Test that Sech sample pulse matches expectations, and parameter validation'
        duration = 100
        amp = 0.5
        sigma = 10
        sech_pulse = Sech(duration=duration, amp=amp, sigma=sigma)
        with self.assertWarns(DeprecationWarning):
            sech_waveform = sech(duration=duration, amp=amp, sigma=sigma)
        np.testing.assert_almost_equal(sech_pulse.get_waveform().samples, sech_waveform.samples)
        sech_pulse = Sech(duration=duration, amp=amp, sigma=sigma, zero_ends=False)
        with self.assertWarns(DeprecationWarning):
            sech_waveform = sech(duration=duration, amp=amp, sigma=sigma, zero_ends=False)
        np.testing.assert_almost_equal(sech_pulse.get_waveform().samples, sech_waveform.samples)
        with self.assertRaises(PulseError):
            Sech(duration=duration, amp=amp, sigma=-5)

    def test_sech_deriv_pulse(self):
        if False:
            while True:
                i = 10
        'Test that SechDeriv sample pulse matches expectations, and parameter validation'
        duration = 100
        amp = 0.5
        sigma = 10
        sech_deriv_pulse = SechDeriv(duration=duration, amp=amp, sigma=sigma)
        with self.assertWarns(DeprecationWarning):
            sech_deriv_waveform = sech_deriv(duration=duration, amp=amp, sigma=sigma)
        np.testing.assert_almost_equal(sech_deriv_pulse.get_waveform().samples, sech_deriv_waveform.samples)
        with self.assertRaises(PulseError):
            SechDeriv(duration=duration, amp=amp, sigma=-5)

    def test_constant_samples(self):
        if False:
            for i in range(10):
                print('nop')
        'Test the constant pulse and its sampled construction.'
        amp = 0.6
        angle = np.pi * 0.7
        const = Constant(duration=150, amp=amp, angle=angle)
        self.assertEqual(const.get_waveform().samples[0], amp * np.exp(1j * angle))
        self.assertEqual(len(const.get_waveform().samples), 150)

    def test_parameters(self):
        if False:
            for i in range(10):
                print('nop')
        'Test that the parameters can be extracted as a dict through the `parameters`\n        attribute.'
        drag = Drag(duration=25, amp=0.2, sigma=7.8, beta=4, angle=0.2)
        self.assertEqual(set(drag.parameters.keys()), {'duration', 'amp', 'sigma', 'beta', 'angle'})
        const = Constant(duration=150, amp=1)
        self.assertEqual(set(const.parameters.keys()), {'duration', 'amp', 'angle'})

    def test_repr(self):
        if False:
            while True:
                i = 10
        'Test the repr methods for parametric pulses.'
        gaus = Gaussian(duration=25, amp=0.7, sigma=4, angle=0.3)
        self.assertEqual(repr(gaus), 'Gaussian(duration=25, sigma=4, amp=0.7, angle=0.3)')
        gaus_square = GaussianSquare(duration=20, sigma=30, amp=1.0, width=3)
        self.assertEqual(repr(gaus_square), 'GaussianSquare(duration=20, sigma=30, width=3, amp=1.0, angle=0)')
        gaus_square = GaussianSquare(duration=20, sigma=30, amp=1.0, angle=0.2, risefall_sigma_ratio=0.1)
        self.assertEqual(repr(gaus_square), 'GaussianSquare(duration=20, sigma=30, width=14.0, amp=1.0, angle=0.2)')
        gsd = GaussianSquareDrag(duration=20, sigma=30, amp=1.0, width=3, beta=1)
        self.assertEqual(repr(gsd), 'GaussianSquareDrag(duration=20, sigma=30, width=3, beta=1, amp=1.0, angle=0.0)')
        gsd = GaussianSquareDrag(duration=20, sigma=30, amp=1.0, risefall_sigma_ratio=0.1, beta=1)
        self.assertEqual(repr(gsd), 'GaussianSquareDrag(duration=20, sigma=30, width=14.0, beta=1, amp=1.0, angle=0.0)')
        gse = gaussian_square_echo(duration=20, sigma=30, amp=1.0, width=3)
        self.assertEqual(repr(gse), 'gaussian_square_echo(duration=20, amp=1.0, angle=0.0, sigma=30, width=3, active_amp=0.0, active_angle=0.0)')
        gse = gaussian_square_echo(duration=20, sigma=30, amp=1.0, risefall_sigma_ratio=0.1)
        self.assertEqual(repr(gse), 'gaussian_square_echo(duration=20, amp=1.0, angle=0.0, sigma=30, width=14.0, active_amp=0.0, active_angle=0.0)')
        drag = Drag(duration=5, amp=0.5, sigma=7, beta=1)
        self.assertEqual(repr(drag), 'Drag(duration=5, sigma=7, beta=1, amp=0.5, angle=0)')
        const = Constant(duration=150, amp=0.1, angle=0.3)
        self.assertEqual(repr(const), 'Constant(duration=150, amp=0.1, angle=0.3)')
        sin_pulse = Sin(duration=150, amp=0.1, angle=0.3, freq=0.2, phase=0)
        self.assertEqual(repr(sin_pulse), 'Sin(duration=150, freq=0.2, phase=0, amp=0.1, angle=0.3)')
        cos_pulse = Cos(duration=150, amp=0.1, angle=0.3, freq=0.2, phase=0)
        self.assertEqual(repr(cos_pulse), 'Cos(duration=150, freq=0.2, phase=0, amp=0.1, angle=0.3)')
        triangle_pulse = Triangle(duration=150, amp=0.1, angle=0.3, freq=0.2, phase=0)
        self.assertEqual(repr(triangle_pulse), 'Triangle(duration=150, freq=0.2, phase=0, amp=0.1, angle=0.3)')
        sawtooth_pulse = Sawtooth(duration=150, amp=0.1, angle=0.3, freq=0.2, phase=0)
        self.assertEqual(repr(sawtooth_pulse), 'Sawtooth(duration=150, freq=0.2, phase=0, amp=0.1, angle=0.3)')
        sech_pulse = Sech(duration=150, amp=0.1, angle=0.3, sigma=10)
        self.assertEqual(repr(sech_pulse), 'Sech(duration=150, sigma=10, amp=0.1, angle=0.3)')
        sech_deriv_pulse = SechDeriv(duration=150, amp=0.1, angle=0.3, sigma=10)
        self.assertEqual(repr(sech_deriv_pulse), 'SechDeriv(duration=150, sigma=10, amp=0.1, angle=0.3)')
        gaussian_deriv_pulse = GaussianDeriv(duration=150, amp=0.1, angle=0.3, sigma=10)
        self.assertEqual(repr(gaussian_deriv_pulse), 'GaussianDeriv(duration=150, sigma=10, amp=0.1, angle=0.3)')

    def test_param_validation(self):
        if False:
            return 10
        'Test that parametric pulse parameters are validated when initialized.'
        with self.assertRaises(PulseError):
            Gaussian(duration=25, sigma=0, amp=0.5, angle=np.pi / 2)
        with self.assertRaises(PulseError):
            GaussianSquare(duration=150, amp=0.2, sigma=8)
        with self.assertRaises(PulseError):
            GaussianSquare(duration=150, amp=0.2, sigma=8, width=100, risefall_sigma_ratio=5)
        with self.assertRaises(PulseError):
            GaussianSquare(duration=150, amp=0.2, sigma=8, width=160)
        with self.assertRaises(PulseError):
            GaussianSquare(duration=150, amp=0.2, sigma=8, risefall_sigma_ratio=10)
        with self.assertRaises(PulseError):
            GaussianSquareDrag(duration=150, amp=0.2, sigma=8, beta=1)
        with self.assertRaises(PulseError):
            GaussianSquareDrag(duration=150, amp=0.2, sigma=8, width=160, beta=1)
        with self.assertRaises(PulseError):
            GaussianSquareDrag(duration=150, amp=0.2, sigma=8, risefall_sigma_ratio=10, beta=1)
        with self.assertRaises(PulseError):
            gaussian_square_echo(duration=150, amp=0.2, sigma=8)
        with self.assertRaises(PulseError):
            gaussian_square_echo(duration=150, amp=0.2, sigma=8, width=160)
        with self.assertRaises(PulseError):
            gaussian_square_echo(duration=150, amp=0.2, sigma=8, risefall_sigma_ratio=10)
        with self.assertRaises(PulseError):
            Constant(duration=150, amp=1.5, angle=np.pi * 0.8)
        with self.assertRaises(PulseError):
            Drag(duration=25, amp=0.5, sigma=-7.8, beta=4, angle=np.pi / 3)

    def test_class_level_limit_amplitude(self):
        if False:
            for i in range(10):
                print('nop')
        'Test that the check for amplitude less than or equal to 1 can\n        be disabled on the class level.\n\n        Tests for representative examples.\n        '
        with self.assertRaises(PulseError):
            Gaussian(duration=100, sigma=1.0, amp=1.7, angle=np.pi * 1.1)
        with patch('qiskit.pulse.library.pulse.Pulse.limit_amplitude', new=False):
            waveform = Gaussian(duration=100, sigma=1.0, amp=1.7, angle=np.pi * 1.1)
            self.assertGreater(np.abs(waveform.amp), 1.0)
            waveform = GaussianSquare(duration=100, sigma=1.0, amp=1.5, width=10, angle=np.pi / 5)
            self.assertGreater(np.abs(waveform.amp), 1.0)
            waveform = GaussianSquareDrag(duration=100, sigma=1.0, amp=1.1, beta=0.1, width=10)
            self.assertGreater(np.abs(waveform.amp), 1.0)

    def test_class_level_disable_validation(self):
        if False:
            for i in range(10):
                print('nop')
        'Test that pulse validation can be disabled on the class level.\n\n        Tests for representative examples.\n        '
        with self.assertRaises(PulseError):
            Gaussian(duration=100, sigma=-1.0, amp=0.5, angle=np.pi * 1.1)
        with patch('qiskit.pulse.library.symbolic_pulses.SymbolicPulse.disable_validation', new=True):
            waveform = Gaussian(duration=100, sigma=-1.0, amp=0.5, angle=np.pi * 1.1)
            self.assertLess(waveform.sigma, 0)
            waveform = GaussianSquare(duration=100, sigma=1.0, amp=0.5, width=1000, angle=np.pi / 5)
            self.assertGreater(waveform.width, waveform.duration)
            waveform = GaussianSquareDrag(duration=100, sigma=1.0, amp=1.1, beta=0.1, width=-1)
            self.assertLess(waveform.width, 0)

    def test_gaussian_limit_amplitude_per_instance(self):
        if False:
            print('Hello World!')
        'Test limit amplitude option per Gaussian instance.'
        with self.assertRaises(PulseError):
            Gaussian(duration=100, sigma=1.0, amp=1.6, angle=np.pi / 2.5)
        waveform = Gaussian(duration=100, sigma=1.0, amp=1.6, angle=np.pi / 2.5, limit_amplitude=False)
        self.assertGreater(np.abs(waveform.amp), 1.0)

    def test_gaussian_square_limit_amplitude_per_instance(self):
        if False:
            i = 10
            return i + 15
        'Test limit amplitude option per GaussianSquare instance.'
        with self.assertRaises(PulseError):
            GaussianSquare(duration=100, sigma=1.0, amp=1.5, width=10, angle=np.pi / 3)
        waveform = GaussianSquare(duration=100, sigma=1.0, amp=1.5, width=10, angle=np.pi / 3, limit_amplitude=False)
        self.assertGreater(np.abs(waveform.amp), 1.0)

    def test_gaussian_square_drag_limit_amplitude_per_instance(self):
        if False:
            i = 10
            return i + 15
        'Test limit amplitude option per GaussianSquareDrag instance.'
        with self.assertRaises(PulseError):
            GaussianSquareDrag(duration=100, sigma=1.0, amp=1.1, beta=0.1, width=10)
        waveform = GaussianSquareDrag(duration=100, sigma=1.0, amp=1.1, beta=0.1, width=10, limit_amplitude=False)
        self.assertGreater(np.abs(waveform.amp), 1.0)

    def test_gaussian_square_echo_limit_amplitude_per_instance(self):
        if False:
            i = 10
            return i + 15
        'Test limit amplitude option per GaussianSquareEcho instance.'
        with self.assertRaises(PulseError):
            gaussian_square_echo(duration=1000, sigma=4.0, amp=1.01, width=100)
        waveform = gaussian_square_echo(duration=1000, sigma=4.0, amp=1.01, width=100, limit_amplitude=False)
        self.assertGreater(np.abs(waveform.amp), 1.0)

    def test_drag_limit_amplitude_per_instance(self):
        if False:
            while True:
                i = 10
        'Test limit amplitude option per DRAG instance.'
        with self.assertRaises(PulseError):
            Drag(duration=100, sigma=1.0, beta=1.0, amp=1.8, angle=np.pi * 0.3)
        waveform = Drag(duration=100, sigma=1.0, beta=1.0, amp=1.8, angle=np.pi * 0.3, limit_amplitude=False)
        self.assertGreater(np.abs(waveform.amp), 1.0)

    def test_constant_limit_amplitude_per_instance(self):
        if False:
            i = 10
            return i + 15
        'Test limit amplitude option per Constant instance.'
        with self.assertRaises(PulseError):
            Constant(duration=100, amp=1.6, angle=0.5)
        waveform = Constant(duration=100, amp=1.6, angle=0.5, limit_amplitude=False)
        self.assertGreater(np.abs(waveform.amp), 1.0)

    def test_sin_limit_amplitude_per_instance(self):
        if False:
            print('Hello World!')
        'Test limit amplitude option per Sin instance.'
        with self.assertRaises(PulseError):
            Sin(duration=100, amp=1.1, phase=0)
        waveform = Sin(duration=100, amp=1.1, phase=0, limit_amplitude=False)
        self.assertGreater(np.abs(waveform.amp), 1.0)

    def test_sawtooth_limit_amplitude_per_instance(self):
        if False:
            print('Hello World!')
        'Test limit amplitude option per Sawtooth instance.'
        with self.assertRaises(PulseError):
            Sawtooth(duration=100, amp=1.1, phase=0)
        waveform = Sawtooth(duration=100, amp=1.1, phase=0, limit_amplitude=False)
        self.assertGreater(np.abs(waveform.amp), 1.0)

    def test_triangle_limit_amplitude_per_instance(self):
        if False:
            while True:
                i = 10
        'Test limit amplitude option per Triangle instance.'
        with self.assertRaises(PulseError):
            Triangle(duration=100, amp=1.1, phase=0)
        waveform = Triangle(duration=100, amp=1.1, phase=0, limit_amplitude=False)
        self.assertGreater(np.abs(waveform.amp), 1.0)

    def test_square_limit_amplitude_per_instance(self):
        if False:
            while True:
                i = 10
        'Test limit amplitude option per Square instance.'
        with self.assertRaises(PulseError):
            Square(duration=100, amp=1.1, phase=0)
        waveform = Square(duration=100, amp=1.1, phase=0, limit_amplitude=False)
        self.assertGreater(np.abs(waveform.amp), 1.0)

    def test_gaussian_deriv_limit_amplitude_per_instance(self):
        if False:
            while True:
                i = 10
        'Test limit amplitude option per GaussianDeriv instance.'
        with self.assertRaises(PulseError):
            GaussianDeriv(duration=100, amp=5, sigma=1)
        waveform = GaussianDeriv(duration=100, amp=5, sigma=1, limit_amplitude=False)
        self.assertGreater(np.abs(waveform.amp / waveform.sigma), np.exp(0.5))

    def test_sech_limit_amplitude_per_instance(self):
        if False:
            print('Hello World!')
        'Test limit amplitude option per Sech instance.'
        with self.assertRaises(PulseError):
            Sech(duration=100, amp=5, sigma=1)
        waveform = Sech(duration=100, amp=5, sigma=1, limit_amplitude=False)
        self.assertGreater(np.abs(waveform.amp), 1.0)

    def test_sech_deriv_limit_amplitude_per_instance(self):
        if False:
            print('Hello World!')
        'Test limit amplitude option per SechDeriv instance.'
        with self.assertRaises(PulseError):
            SechDeriv(duration=100, amp=5, sigma=1)
        waveform = SechDeriv(duration=100, amp=5, sigma=1, limit_amplitude=False)
        self.assertGreater(np.abs(waveform.amp) / waveform.sigma, 2.0)

    def test_get_parameters(self):
        if False:
            print('Hello World!')
        'Test getting pulse parameters as attribute.'
        drag_pulse = Drag(duration=100, amp=0.1, sigma=40, beta=3)
        self.assertEqual(drag_pulse.duration, 100)
        self.assertEqual(drag_pulse.amp, 0.1)
        self.assertEqual(drag_pulse.sigma, 40)
        self.assertEqual(drag_pulse.beta, 3)
        with self.assertRaises(AttributeError):
            drag_pulse.non_existing_parameter

    def test_envelope_cache(self):
        if False:
            while True:
                i = 10
        'Test speed up of instantiation with lambdify envelope cache.'
        drag_instance1 = Drag(duration=100, amp=0.1, sigma=40, beta=3)
        drag_instance2 = Drag(duration=100, amp=0.1, sigma=40, beta=3)
        self.assertTrue(drag_instance1._envelope_lam is drag_instance2._envelope_lam)

    def test_constraints_cache(self):
        if False:
            i = 10
            return i + 15
        'Test speed up of instantiation with lambdify constraints cache.'
        drag_instance1 = Drag(duration=100, amp=0.1, sigma=40, beta=3)
        drag_instance2 = Drag(duration=100, amp=0.1, sigma=40, beta=3)
        self.assertTrue(drag_instance1._constraints_lam is drag_instance2._constraints_lam)

    def test_deepcopy(self):
        if False:
            return 10
        'Test deep copying instance.'
        import copy
        drag = Drag(duration=100, amp=0.1, sigma=40, beta=3)
        drag_copied = copy.deepcopy(drag)
        self.assertNotEqual(id(drag), id(drag_copied))
        orig_wf = drag.get_waveform()
        copied_wf = drag_copied.get_waveform()
        np.testing.assert_almost_equal(orig_wf.samples, copied_wf.samples)

    def test_fully_parametrized_pulse(self):
        if False:
            print('Hello World!')
        'Test instantiating a pulse with parameters.'
        amp = Parameter('amp')
        duration = Parameter('duration')
        sigma = Parameter('sigma')
        beta = Parameter('beta')
        drag = Drag(duration=duration, amp=amp, sigma=sigma, beta=beta)
        with self.assertRaises(PulseError):
            drag.get_waveform()

    def test_custom_pulse(self):
        if False:
            while True:
                i = 10
        'Test defining a custom pulse which is not in the form of amp * F(t).'
        (t, t1, t2, amp1, amp2) = sym.symbols('t, t1, t2, amp1, amp2')
        envelope = sym.Piecewise((amp1, sym.And(t > t1, t < t2)), (amp2, sym.true))
        custom_pulse = SymbolicPulse(pulse_type='Custom', duration=100, parameters={'t1': 30, 't2': 80, 'amp1': 0.1j, 'amp2': -0.1}, envelope=envelope)
        waveform = custom_pulse.get_waveform()
        reference = np.concatenate([-0.1 * np.ones(30), 0.1j * np.ones(50), -0.1 * np.ones(20)])
        np.testing.assert_array_almost_equal(waveform.samples, reference)

    def test_gaussian_deprecated_type_check(self):
        if False:
            i = 10
            return i + 15
        'Test isinstance check works with deprecation.'
        gaussian_pulse = Gaussian(160, 0.1, 40)
        self.assertTrue(isinstance(gaussian_pulse, SymbolicPulse))
        with self.assertWarns(PendingDeprecationWarning):
            self.assertTrue(isinstance(gaussian_pulse, Gaussian))
            self.assertFalse(isinstance(gaussian_pulse, GaussianSquare))
            self.assertFalse(isinstance(gaussian_pulse, Drag))
            self.assertFalse(isinstance(gaussian_pulse, Constant))

    def test_gaussian_square_deprecated_type_check(self):
        if False:
            return 10
        'Test isinstance check works with deprecation.'
        gaussian_square_pulse = GaussianSquare(800, 0.1, 64, 544)
        self.assertTrue(isinstance(gaussian_square_pulse, SymbolicPulse))
        with self.assertWarns(PendingDeprecationWarning):
            self.assertFalse(isinstance(gaussian_square_pulse, Gaussian))
            self.assertTrue(isinstance(gaussian_square_pulse, GaussianSquare))
            self.assertFalse(isinstance(gaussian_square_pulse, Drag))
            self.assertFalse(isinstance(gaussian_square_pulse, Constant))

    def test_drag_deprecated_type_check(self):
        if False:
            for i in range(10):
                print('nop')
        'Test isinstance check works with deprecation.'
        drag_pulse = Drag(160, 0.1, 40, 1.5)
        self.assertTrue(isinstance(drag_pulse, SymbolicPulse))
        with self.assertWarns(PendingDeprecationWarning):
            self.assertFalse(isinstance(drag_pulse, Gaussian))
            self.assertFalse(isinstance(drag_pulse, GaussianSquare))
            self.assertTrue(isinstance(drag_pulse, Drag))
            self.assertFalse(isinstance(drag_pulse, Constant))

    def test_constant_deprecated_type_check(self):
        if False:
            return 10
        'Test isinstance check works with deprecation.'
        constant_pulse = Constant(160, 0.1, 40, 1.5)
        self.assertTrue(isinstance(constant_pulse, SymbolicPulse))
        with self.assertWarns(PendingDeprecationWarning):
            self.assertFalse(isinstance(constant_pulse, Gaussian))
            self.assertFalse(isinstance(constant_pulse, GaussianSquare))
            self.assertFalse(isinstance(constant_pulse, Drag))
            self.assertTrue(isinstance(constant_pulse, Constant))

class TestFunctionalPulse(QiskitTestCase):
    """Waveform tests."""

    def test_gaussian(self):
        if False:
            while True:
                i = 10
        'Test gaussian pulse.'

        @functional_pulse
        def local_gaussian(duration, amp, t0, sig):
            if False:
                while True:
                    i = 10
            x = np.linspace(0, duration - 1, duration)
            return amp * np.exp(-(x - t0) ** 2 / sig ** 2)
        pulse_wf_inst = local_gaussian(duration=10, amp=1, t0=5, sig=1, name='test_pulse')
        _y = 1 * np.exp(-(np.linspace(0, 9, 10) - 5) ** 2 / 1 ** 2)
        self.assertListEqual(list(pulse_wf_inst.samples), list(_y))
        self.assertEqual(pulse_wf_inst.name, 'test_pulse')
        self.assertEqual(pulse_wf_inst.duration, 10)

    def test_variable_duration(self):
        if False:
            print('Hello World!')
        'Test generation of sample pulse with variable duration.'

        @functional_pulse
        def local_gaussian(duration, amp, t0, sig):
            if False:
                while True:
                    i = 10
            x = np.linspace(0, duration - 1, duration)
            return amp * np.exp(-(x - t0) ** 2 / sig ** 2)
        _durations = np.arange(10, 15, 1)
        for _duration in _durations:
            pulse_wf_inst = local_gaussian(duration=_duration, amp=1, t0=5, sig=1)
            self.assertEqual(len(pulse_wf_inst.samples), _duration)

class TestScalableSymbolicPulse(QiskitTestCase):
    """ScalableSymbolicPulse tests"""

    def test_scalable_comparison(self):
        if False:
            return 10
        'Test equating of pulses'
        gaussian_negamp = Gaussian(duration=25, sigma=4, amp=-0.5, angle=0)
        gaussian_piphase = Gaussian(duration=25, sigma=4, amp=0.5, angle=np.pi)
        self.assertEqual(gaussian_negamp, gaussian_piphase)
        amp = Parameter('amp')
        gaussian1 = Gaussian(duration=25, sigma=4, amp=amp, angle=0)
        gaussian2 = Gaussian(duration=25, sigma=4, amp=amp, angle=0)
        self.assertEqual(gaussian1, gaussian2)
        gaussian1._params['sigma'] = 10
        self.assertNotEqual(gaussian1, gaussian2)
if __name__ == '__main__':
    unittest.main()