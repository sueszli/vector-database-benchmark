"""
Copyright 2022 Ettus Research, A National Instruments Brand.

SPDX-License-Identifier: GPL-3.0-or-later
"""
import sys
MAX_NUM_CHANNELS = 16
MAIN_TMPL = 'id: uhd_rfnoc_fir_filter\nlabel: RFNoC FIR Filter Block\ncategory: \'[Core]/UHD/RFNoC/Blocks\'\n\nparameters:\n- id: num_chans\n  label: Channel count\n  dtype: int\n  options: [ ${", ".join([str(n) for n in range(1, max_num_chans+1)])} ]\n  default: 1\n  hide: ${\'$\'}{ \'part\' }\n- id: block_args\n  label: Block Args\n  dtype: string\n  default: ""\n  hide: ${\'$\'}{ \'part\' if not block_args else \'none\'}\n- id: device_select\n  label: Device Select\n  dtype: int\n  default: -1\n  hide: ${\'$\'}{ \'part\' if device_select == -1 else \'none\'}\n- id: instance_index\n  label: Instance Select\n  dtype: int\n  default: -1\n  hide: ${\'$\'}{ \'part\' if instance_index == -1 else \'none\'}\n${coeffs_params}\ninputs:\n- domain: rfnoc\n  dtype: \'sc16\'\n  multiplicity: ${\'$\'}{num_chans}\n\noutputs:\n- domain: rfnoc\n  dtype: \'sc16\'\n  multiplicity: ${\'$\'}{num_chans}\n\ntemplates:\n  imports: |-\n    from gnuradio import uhd\n  make: |-\n    uhd.rfnoc_fir_filter(\n        self.rfnoc_graph,\n        uhd.device_addr(${\'$\'}{block_args}),\n        ${\'$\'}{device_select},\n        ${\'$\'}{instance_index})\n${init_params}  callbacks:\n${callback_params}\n\ndocumentation: |-\n  RFNoC FIR Filter Block:\n  Applies a FIR filter with user defined coefficients to the input sample\n  stream.\n\n  Channel count:\n  Number of channels / streams to use with the RFNoC FIR filter block.\n  Note, this is defined by the RFNoC FIR Filter Block\'s FPGA build\n  parameters and GNU Radio Companion is not aware of this value. An error\n  will occur at runtime when connecting blocks if the number of channels is\n  too large.\n\n  Coefficient Type:\n  Choice between floating point or integer coefficients.\n  Floating point coefficients must be within range [-1.0, 1.0].\n  Integer coefficients must be within range [-32768, 32767].\n\n  Coefficients:\n  Array of coefficients. Number of elements in the array implicitly sets\n  the filter length, and must be less than or equal to the maximum number\n  of coefficients supported by the block.\n\nfile_format: 1\n'
COEFFS_PARAM = "- id: coeffs_type${n}\n  label: 'Ch${n}: Coeffs Type'\n  dtype: enum\n  options: ['float', 'integer']\n  default: 'float'\n  hide: ${'$'}{ 'part' if num_chans > ${n} else 'all' }\n- id: coeffs_float${n}\n  label: 'Ch${n}: Coefficients'\n  dtype: float_vector\n  default: [1.0]\n  hide: ${'$'}{ 'none' if coeffs_type${n} == 'float' and num_chans > ${n} else 'all' }\n- id: coeffs_int${n}\n  label: 'Ch${n}: Coefficients'\n  dtype: int_vector\n  default: [32767]\n  hide: ${'$'}{ 'none' if coeffs_type${n} == 'integer' and num_chans > ${n} else 'all' }\n"
INIT_PARAM = '    ${\'%\'} if coeffs_type${n} == "float" and context.get(\'num_chans\')() > ${n}:\n    self.${\'$\'}{id}.set_coefficients(${\'$\'}{coeffs_float${n}}, ${n})\n    ${\'%\'} endif\n    ${\'%\'} if coeffs_type${n} == "integer" and context.get(\'num_chans\')() > ${n}:\n    self.${\'$\'}{id}.set_coefficients(${\'$\'}{coeffs_int${n}}, ${n})\n    ${\'%\'} endif\n'
CALLBACKS_PARAM = '  - |\n    ${\'%\'} if coeffs_type${n} == "float" and context.get(\'num_chans\')() > ${n}:\n    self.${\'$\'}{id}.set_coefficients(${\'$\'}{coeffs_float${n}}, ${n})\n    ${\'%\'} endif\n  - |\n    ${\'%\'} if coeffs_type${n} == "integer" and context.get(\'num_chans\')() > ${n}:\n    self.${\'$\'}{id}.set_coefficients(${\'$\'}{coeffs_int${n}}, ${n})\n    ${\'%\'} endif\n'

def parse_tmpl(_tmpl, **kwargs):
    if False:
        while True:
            i = 10
    ' Render _tmpl using the kwargs '
    from mako.template import Template
    block_template = Template(_tmpl)
    return str(block_template.render(**kwargs))
if __name__ == '__main__':
    file = sys.argv[1]
    coeffs_params = ''.join([parse_tmpl(COEFFS_PARAM, n=n) for n in range(MAX_NUM_CHANNELS)])
    init_params = ''.join([parse_tmpl(INIT_PARAM, n=n) for n in range(MAX_NUM_CHANNELS)])
    callback_params = ''.join([parse_tmpl(CALLBACKS_PARAM, n=n) for n in range(MAX_NUM_CHANNELS)])
    open(file, 'w').write(parse_tmpl(MAIN_TMPL, max_num_chans=MAX_NUM_CHANNELS, coeffs_params=coeffs_params, init_params=init_params, callback_params=callback_params))