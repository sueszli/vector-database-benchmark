from gnuradio import blocks
from gnuradio import gr
from gnuradio.filter import firdes
import math

class iqbal_gen(gr.hier_block2):

    def __init__(self, magnitude=0, phase=0, mode=0):
        if False:
            return 10
        '\n        This block implements the single branch IQ imbalance\n        transmitter and receiver models.\n\n        Developed from source (2014):\n        "In-Phase and Quadrature Imbalance:\n          Modeling, Estimation, and Compensation"\n\n        TX Impairment:\n\n                                  {R}--|Multiply: 10**(mag/20)|--+--|Multiply: cos(pi*degree/180)|--X1\n        Input ---|Complex2Float|---|                             +--|Multiply: sin(pi*degree/180)|--X2\n                                  {I}--|  Adder  |\n                                   X2--|   (+)   |--X3\n\n                          X1--{R}--| Float 2 |--- Output\n                          X3--{I}--| Complex |\n\n        RX Impairment:\n\n                                  {R}--|Multiply: cos(pi*degree/180)|-------|       |\n        Input ---|Complex2Float|---|                                        | Adder |--X1\n                                  {I}--+--|Multiply: sin(pi*degree/180)|----|  (+)  |\n                                       |\n                                       +--X2\n\n                        X1--|Multiply: 10**(mag/20)|--{R}--| Float 2 |--- Output\n                        X2---------------------------{I}--| Complex |\n\n        (ASCII ART monospace viewing)\n        '
        gr.hier_block2.__init__(self, 'IQ Imbalance Generator', gr.io_signature(1, 1, gr.sizeof_gr_complex * 1), gr.io_signature(1, 1, gr.sizeof_gr_complex * 1))
        self.magnitude = magnitude
        self.phase = phase
        self.mode = mode
        self.mag = blocks.multiply_const_vff((math.pow(10, magnitude / 20.0),))
        self.sin_phase = blocks.multiply_const_vff((math.sin(phase * math.pi / 180.0),))
        self.cos_phase = blocks.multiply_const_vff((math.cos(phase * math.pi / 180.0),))
        self.f2c = blocks.float_to_complex(1)
        self.c2f = blocks.complex_to_float(1)
        self.adder = blocks.add_vff(1)
        if self.mode:
            self.connect((self, 0), (self.c2f, 0))
            self.connect((self.c2f, 0), (self.cos_phase, 0))
            self.connect((self.cos_phase, 0), (self.adder, 0))
            self.connect((self.c2f, 1), (self.sin_phase, 0))
            self.connect((self.sin_phase, 0), (self.adder, 1))
            self.connect((self.adder, 0), (self.mag, 0))
            self.connect((self.mag, 0), (self.f2c, 0))
            self.connect((self.c2f, 1), (self.f2c, 1))
            self.connect((self.f2c, 0), (self, 0))
        else:
            self.connect((self, 0), (self.c2f, 0))
            self.connect((self.c2f, 0), (self.mag, 0))
            self.connect((self.mag, 0), (self.cos_phase, 0))
            self.connect((self.cos_phase, 0), (self.f2c, 0))
            self.connect((self.mag, 0), (self.sin_phase, 0))
            self.connect((self.sin_phase, 0), (self.adder, 0))
            self.connect((self.c2f, 1), (self.adder, 1))
            self.connect((self.adder, 0), (self.f2c, 1))
            self.connect((self.f2c, 0), (self, 0))

    def get_magnitude(self):
        if False:
            for i in range(10):
                print('nop')
        return self.magnitude

    def set_magnitude(self, magnitude):
        if False:
            while True:
                i = 10
        self.magnitude = magnitude
        self.mag.set_k((math.pow(10, self.magnitude / 20.0),))

    def get_phase(self):
        if False:
            for i in range(10):
                print('nop')
        return self.phase

    def set_phase(self, phase):
        if False:
            while True:
                i = 10
        self.phase = phase
        self.sin_phase.set_k((math.sin(self.phase * math.pi / 180.0),))
        self.cos_phase.set_k((math.cos(self.phase * math.pi / 180.0),))