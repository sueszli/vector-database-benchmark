"""Script to run run_eval.py locally.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
from subprocess import call
import sys
CONFIGS_PATH = 'configs'
CONTEXT_CONFIGS_PATH = 'context/configs'

def main():
    if False:
        for i in range(10):
            print('nop')
    bb = './'
    base_num_args = 6
    if len(sys.argv) < base_num_args:
        print('usage: python %s <exp_name> <context_setting_gin> <context_gin> <agent_gin> <suite> [params...]' % sys.argv[0])
        sys.exit(0)
    exp = sys.argv[1]
    context_setting = sys.argv[2]
    context = sys.argv[3]
    agent = sys.argv[4]
    assert sys.argv[5] in ['suite'], "args[5] must be `suite'"
    suite = ''
    binary = 'python {bb}/run_eval{suite}.py '.format(bb=bb, suite=suite)
    h = os.environ['HOME']
    ucp = CONFIGS_PATH
    ccp = CONTEXT_CONFIGS_PATH
    extra = ''
    command_str = '{binary} --logtostderr --checkpoint_dir={h}/tmp/{context_setting}/{context}/{agent}/{exp}/train --eval_dir={h}/tmp/{context_setting}/{context}/{agent}/{exp}/eval --config_file={ucp}/{agent}.gin --config_file={ucp}/eval_{extra}uvf.gin --config_file={ccp}/{context_setting}.gin --config_file={ccp}/{context}.gin '.format(h=h, ucp=ucp, ccp=ccp, context_setting=context_setting, context=context, agent=agent, extra=extra, suite=suite, exp=exp, binary=binary)
    for extra_arg in sys.argv[base_num_args:]:
        command_str += "--params='%s' " % extra_arg
    print(command_str)
    call(command_str, shell=True)
if __name__ == '__main__':
    main()