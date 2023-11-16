"""
Copyright 2023 Ettus Research, A National Instruments Brand.

SPDX-License-Identifier: GPL-3.0-or-later
"""
import sys
MAX_NUM_CHANNELS = 4
MAIN_TMPL = 'id: uhd_rfnoc_duc\nlabel: RFNoC Digital Upconverter Block\ncategory: \'[Core]/UHD/RFNoC/Blocks\'\n\nparameters:\n- id: num_chans\n  label: Channel count\n  dtype: int\n  default: 1\n  hide: ${\'$\'}{ \'part\' }\n- id: block_args\n  label: Block Args\n  dtype: string\n  default: ""\n  hide: ${\'$\'}{ \'part\' if not block_args else \'none\'}\n- id: device_select\n  label: Device Select\n  dtype: int\n  default: -1\n  hide: ${\'$\'}{ \'part\' if device_select == -1 else \'none\'}\n- id: instance_index\n  label: Instance Select\n  dtype: int\n  default: -1\n  hide: ${\'$\'}{ \'part\' if instance_index == -1 else \'none\'}\n${coeffs_params}\n\ninputs:\n- domain: rfnoc\n  dtype: \'sc16\'\n  vlen: 1\n  multiplicity: ${\'$\'}{num_chans}\n\noutputs:\n- domain: rfnoc\n  dtype: \'sc16\'\n  vlen: 1\n  multiplicity: ${\'$\'}{num_chans}\n\ntemplates:\n  imports: |-\n    from gnuradio import uhd\n  make: |-\n    uhd.rfnoc_duc(\n        self.rfnoc_graph,\n        uhd.device_addr(${\'$\'}{block_args}),\n        ${\'$\'}{device_select},\n        ${\'$\'}{instance_index})\n${init_params}  callbacks:\n${callback_params}\n\nfile_format: 1\n'
COEFFS_PARAM = "- id: freq${n}\n  label: 'Ch${n}: Frequency Shift (Hz)'\n  dtype: real\n  default: 0\n  hide: ${'$'}{ 'none' if num_chans > ${n} else 'all' }\n- id: input_rate${n}\n  label: 'Ch${n}: Input Rate (Hz)'\n  dtype: real\n  default: 0\n  hide: ${'$'}{ 'none' if num_chans > ${n} else 'all' }\n"
INIT_PARAM = "    ${'%'} if context.get('num_chans')() > ${n}:\n    self.${'$'}{id}.set_freq(${'$'}{freq${n}}, ${n})\n    self.${'$'}{id}.set_input_rate(${'$'}{input_rate${n}}, ${n})\n    ${'%'} endif\n"
CALLBACKS_PARAM = "  - |\n    ${'%'} if context.get('num_chans')() > ${n}:\n    set_freq(${'$'}{freq${n}}, ${n})\n    ${'%'} endif\n  - |\n    ${'%'} if context.get('num_chans')() > ${n}:\n    set_input_rate(${'$'}{input_rate${n}}, ${n})\n    ${'%'} endif\n"

def parse_tmpl(_tmpl, **kwargs):
    if False:
        for i in range(10):
            print('nop')
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