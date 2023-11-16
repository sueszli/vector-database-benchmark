import random
import math
from cmath import exp, pi, log, sqrt
import numpy
from gnuradio import gr, gr_unittest, digital, blocks
from gnuradio.digital.utils import mod_codes
from gnuradio.digital import constellation, psk, qam, qamlike
import numpy as np
tested_mod_codes = (mod_codes.NO_CODE, mod_codes.GRAY_CODE)

def twod_constell():
    if False:
        i = 10
        return i + 15
    '\n\n    '
    points = (1 + 0j, 0 + 1j, -1 + 0j, 0 - 1j)
    rot_sym = 2
    dim = 2
    return digital.constellation_calcdist(points, [], rot_sym, dim)

def threed_constell():
    if False:
        for i in range(10):
            print('nop')
    oned_points = (1 + 0j, 0 + 1j, -1 + 0j, 0 - 1j)
    points = []
    r4 = list(range(0, 4))
    for ia in r4:
        for ib in r4:
            for ic in r4:
                points += [oned_points[ia], oned_points[ib], oned_points[ic]]
    rot_sym = 4
    dim = 3
    return digital.constellation_calcdist(points, [], rot_sym, dim)
easy_constellation_info = ((psk.psk_constellation, {'m': (2, 4, 8, 16), 'mod_code': tested_mod_codes}, True, None), (psk.psk_constellation, {'m': (2, 4, 8, 16, 32, 64), 'mod_code': tested_mod_codes, 'differential': (False,)}, False, None), (qam.qam_constellation, {'constellation_points': (4,), 'mod_code': tested_mod_codes, 'large_ampls_to_corners': [False]}, True, None), (qam.qam_constellation, {'constellation_points': (4, 16, 64), 'mod_code': tested_mod_codes, 'differential': (False,)}, False, None), (digital.constellation_bpsk, {}, True, None), (digital.constellation_qpsk, {}, False, None), (digital.constellation_dqpsk, {}, True, None), (digital.constellation_8psk, {}, False, None), (twod_constell, {}, True, None), (threed_constell, {}, True, None))
medium_constellation_info = ((psk.psk_constellation, {'m': (32, 64), 'mod_code': tested_mod_codes}, True, None), (qam.qam_constellation, {'constellation_points': (16,), 'mod_code': tested_mod_codes, 'large_ampls_to_corners': [False, True]}, True, None), (qamlike.qam32_holeinside_constellation, {'large_ampls_to_corners': [True]}, True, None))
difficult_constellation_info = ((qam.qam_constellation, {'constellation_points': (64,), 'mod_code': tested_mod_codes, 'large_ampls_to_corners': [False, True]}, True, None),)

def slicer(x):
    if False:
        for i in range(10):
            print('nop')
    ret = []
    for xi in x:
        if xi < 0:
            ret.append(0.0)
        else:
            ret.append(1.0)
    return ret

def tested_constellations(easy=True, medium=True, difficult=True):
    if False:
        print('Hello World!')
    '\n    Generator to produce (constellation, differential) tuples for testing purposes.\n    '
    constellation_info = []
    if easy:
        constellation_info += easy_constellation_info
    if medium:
        constellation_info += medium_constellation_info
    if difficult:
        constellation_info += difficult_constellation_info
    for (constructor, poss_args, differential, diff_argname) in constellation_info:
        if differential:
            diff_poss = (True, False)
        else:
            diff_poss = (False,)
        poss_args = [[argname, argvalues, 0] for (argname, argvalues) in list(poss_args.items())]
        for current_diff in diff_poss:
            while True:
                current_args = dict([(argname, argvalues[argindex]) for (argname, argvalues, argindex) in poss_args])
                if diff_argname is not None:
                    current_args[diff_argname] = current_diff
                constellation = constructor(**current_args)
                yield (constellation, current_diff)
                for this_poss_arg in poss_args:
                    (argname, argvalues, argindex) = this_poss_arg
                    if argindex < len(argvalues) - 1:
                        this_poss_arg[2] += 1
                        break
                    else:
                        this_poss_arg[2] = 0
                if sum([argindex for (argname, argvalues, argindex) in poss_args]) == 0:
                    break

class test_constellation(gr_unittest.TestCase):
    src_length = 256

    def setUp(self):
        if False:
            print('Hello World!')
        random.seed(0)
        self.src_data = [random.randint(0, 1) for i in range(0, self.src_length)]

    def tearDown(self):
        if False:
            i = 10
            return i + 15
        pass

    def test_normalization(self):
        if False:
            print('Hello World!')
        rot_sym = 1
        side = 2
        width = 2
        for (constel_points, code) in (digital.psk_4_0(), digital.qam_16_0()):
            constel = digital.constellation_rect(constel_points, code, rot_sym, side, side, width, width, constellation.POWER_NORMALIZATION)
            points = np.array(constel.points())
            avg_power = np.sum(abs(points) ** 2) / len(points)
            self.assertAlmostEqual(avg_power, 1.0, 6)
            constel = digital.constellation_rect(constel_points, code, rot_sym, side, side, width, width, constellation.AMPLITUDE_NORMALIZATION)
            points = np.array(constel.points())
            avg_amp = np.sum(abs(points)) / len(points)
            self.assertAlmostEqual(avg_amp, 1.0, 6)

    def test_hard_decision(self):
        if False:
            print('Hello World!')
        for (constellation, differential) in tested_constellations():
            if differential:
                rs = constellation.rotational_symmetry()
                rotations = [exp(i * 2 * pi * (0 + 1j) / rs) for i in range(0, rs)]
            else:
                rotations = [None]
            for rotation in rotations:
                src = blocks.vector_source_b(self.src_data)
                content = mod_demod(constellation, differential, rotation)
                dst = blocks.vector_sink_b()
                self.tb = gr.top_block()
                self.tb.connect(src, content, dst)
                self.tb.run()
                data = dst.data()
                first = constellation.bits_per_symbol()
                equality = all(numpy.equal(self.src_data[first:len(data)], data[first:]))
                if not equality:
                    msg = 'Constellations mismatched. ' + f'{type(constellation)}; ' + f'Differential? {differential}; ' + f'{len(constellation.points())} ' + 'Constellation points: ' + f'{constellation.points()};'
                    self.assertEqual(self.src_data[first:len(data)], data[first:], msg=msg)

    def test_soft_qpsk_gen(self):
        if False:
            print('Hello World!')
        prec = 8
        c = digital.constellation_qpsk().base()
        constel = c.points()
        code = [0, 1, 2, 3]
        Es = 1.0
        c.set_npwr(Es)
        c.normalize(digital.constellation.POWER_NORMALIZATION)
        table = digital.soft_dec_table(constel, code, prec, Es)
        constel = digital.const_normalization(constel, 'POWER')
        maxamp = digital.min_max_axes(constel)
        c.set_soft_dec_lut(table, prec)
        x = sqrt(2.0) / 2.0
        step = (x.real + x.real) / (2 ** prec - 1)
        samples = [-x - x * 1j, -x + x * 1j, x + x * 1j, x - x * 1j, -x + 128 * step + (-x + 128 * step) * 1j, -x + 64 * step + (-x + 64 * step) * 1j, -x + 64 * step + (-x + 192 * step) * 1j, -x + 192 * step + (-x + 192 * step) * 1j, -x + 192 * step + (-x + 64 * step) * 1j]
        y_python_raw_calc = []
        y_python_gen_calc = []
        y_python_table = []
        y_cpp_raw_calc = []
        y_cpp_table = []
        for sample in samples:
            y_python_raw_calc += slicer(digital.calc_soft_dec(sample, constel, code))
            y_python_gen_calc += slicer(digital.sd_psk_4_0(sample, Es))
            y_python_table += slicer(digital.calc_soft_dec_from_table(sample, table, prec, maxamp))
            y_cpp_raw_calc += c.calc_soft_dec(sample)
            y_cpp_table += c.soft_decision_maker(sample)
        self.assertFloatTuplesAlmostEqual(y_python_raw_calc, y_python_gen_calc, 3)
        self.assertFloatTuplesAlmostEqual(y_python_gen_calc, y_python_table, 1)
        self.assertFloatTuplesAlmostEqual(y_cpp_raw_calc, y_cpp_table, 1)

    def test_soft_qpsk_calc(self):
        if False:
            while True:
                i = 10
        prec = 8
        (constel, code) = digital.psk_4_0()
        rot_sym = 1
        side = 2
        width = 2
        c = digital.constellation_rect(constel, code, rot_sym, side, side, width, width)
        constel = c.points()
        Es = max([abs(constel_i) for constel_i in constel])
        table = digital.soft_dec_table(constel, code, prec)
        c.gen_soft_dec_lut(prec)
        x = sqrt(2.0) / 2.0
        step = (x.real + x.real) / (2 ** prec - 1)
        samples = [-x - x * 1j, -x + x * 1j, x + x * 1j, x - x * 1j, -x + 128 * step + (-x + 128 * step) * 1j, -x + 64 * step + (-x + 64 * step) * 1j, -x + 64 * step + (-x + 192 * step) * 1j, -x + 192 * step + (-x + 192 * step) * 1j, -x + 192 * step + (-x + 64 * step) * 1j]
        y_python_raw_calc = []
        y_python_table = []
        y_cpp_raw_calc = []
        y_cpp_table = []
        for sample in samples:
            y_python_raw_calc += slicer(digital.calc_soft_dec(sample, constel, code))
            y_python_table += slicer(digital.calc_soft_dec_from_table(sample, table, prec, Es))
            y_cpp_raw_calc += slicer(c.calc_soft_dec(sample))
            y_cpp_table += slicer(c.soft_decision_maker(sample))
        self.assertEqual(y_python_raw_calc, y_python_table)
        self.assertEqual(y_cpp_raw_calc, y_cpp_table)

    def test_soft_qam16_calc(self):
        if False:
            for i in range(10):
                print('nop')
        prec = 8
        (constel, code) = digital.qam_16_0()
        rot_sym = 1
        side = 2
        width = 2
        c = digital.constellation_rect(constel, code, rot_sym, side, side, width, width)
        constel = c.points()
        Es = 1.0
        padding = 2
        table = digital.soft_dec_table(constel, code, prec)
        c.gen_soft_dec_lut(prec)
        x = sqrt(2.0) / 2.0
        step = (x.real + x.real) / (2 ** prec - 1)
        samples = [-x - x * 1j, -x + x * 1j, x + x * 1j, x - x * 1j, -x + 128 * step + (-x + 128 * step) * 1j, -x + 64 * step + (-x + 64 * step) * 1j, -x + 64 * step + (-x + 192 * step) * 1j, -x + 192 * step + (-x + 192 * step) * 1j, -x + 192 * step + (-x + 64 * step) * 1j]
        y_python_raw_calc = []
        y_python_table = []
        y_cpp_raw_calc = []
        y_cpp_table = []
        for sample in samples:
            y_python_raw_calc += slicer(digital.calc_soft_dec(sample, constel, code))
            y_python_table += slicer(digital.calc_soft_dec_from_table(sample, table, prec, Es))
            y_cpp_raw_calc += slicer(c.calc_soft_dec(sample))
            y_cpp_table += slicer(c.soft_decision_maker(sample))
        self.assertFloatTuplesAlmostEqual(y_python_raw_calc, y_python_table, 3)
        self.assertFloatTuplesAlmostEqual(y_cpp_raw_calc, y_cpp_table, 3)

class mod_demod(gr.hier_block2):

    def __init__(self, constellation, differential, rotation):
        if False:
            while True:
                i = 10
        if constellation.arity() > 256:
            raise ValueError('Constellation cannot contain more than 256 points.')
        gr.hier_block2.__init__(self, 'mod_demod', gr.io_signature(1, 1, gr.sizeof_char), gr.io_signature(1, 1, gr.sizeof_char))
        arity = constellation.arity()
        self.constellation = constellation
        self.differential = differential
        import weakref
        self.blocks = [weakref.proxy(self)]
        self.blocks.append(blocks.unpacked_to_packed_bb(1, gr.GR_MSB_FIRST))
        self.blocks.append(blocks.packed_to_unpacked_bb(self.constellation.bits_per_symbol(), gr.GR_MSB_FIRST))
        if self.constellation.apply_pre_diff_code():
            self.blocks.append(digital.map_bb(self.constellation.pre_diff_code()))
        if self.differential:
            self.blocks.append(digital.diff_encoder_bb(arity))
        self.blocks.append(digital.chunks_to_symbols_bc(self.constellation.points(), self.constellation.dimensionality()))
        if rotation is not None:
            self.blocks.append(blocks.multiply_const_cc(rotation))
        self.blocks.append(digital.constellation_decoder_cb(self.constellation.base()))
        if self.differential:
            self.blocks.append(digital.diff_decoder_bb(arity))
        if self.constellation.apply_pre_diff_code():
            self.blocks.append(digital.map_bb(mod_codes.invert_code(self.constellation.pre_diff_code())))
        self.blocks.append(blocks.unpack_k_bits_bb(self.constellation.bits_per_symbol()))
        check_index = len(self.blocks)
        self.blocks = self.blocks[:check_index]
        self.blocks.append(weakref.proxy(self))
        self.connect(*self.blocks)
if __name__ == '__main__':
    gr_unittest.run(test_constellation)