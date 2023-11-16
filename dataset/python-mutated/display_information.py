"""
Library for getting information of model, hardware, enviroment
during neon tests and benchmarking
"""
from neon import logger as neon_logger

def add_param_to_output(output_string, param_name, param):
    if False:
        for i in range(10):
            print('nop')
    try:
        output_string += '{:>25}: {}\n'.format(param_name, param)
    except Exception:
        output_string += '{:>25}: UNKNOWN\n'.format(param_name)
    return output_string

def display_model_params(neon_args, neon_root_yaml):
    if False:
        print('Hello World!')
    '\n    Display model parameters\n    :param      neon_args: contains command line arguments,\n    :param neon_root_yaml: contains YAML elements\n    '
    output_string = '\n-- INFORMATION: HYPER PARAMETERS ------\n'
    try:
        output_string = add_param_to_output(output_string, 'backend', neon_args.backend)
        output_string = add_param_to_output(output_string, 'batch size', neon_args.batch_size)
        output_string = add_param_to_output(output_string, 'epochs', neon_args.epochs)
        output_string = add_param_to_output(output_string, 'optimizer type', neon_root_yaml['optimizer']['type'])
        output_string = add_param_to_output(output_string, 'learning rate', neon_root_yaml['optimizer']['config']['learning_rate'])
        output_string = add_param_to_output(output_string, 'momentum coef', neon_root_yaml['optimizer']['config']['momentum_coef'])
    except Exception:
        output_string += 'Some parameters cannot be displayed\n'
    output_string += '----------------------------------------'
    neon_logger.display(output_string)

def display_cpu_information():
    if False:
        i = 10
        return i + 15
    '\n    Display CPU information.\n    Assumes all CPUs are the same.\n    '
    import cpuinfo
    output_string = '\n-- INFORMATION: CPU -------------------\n'
    cpu_info = cpuinfo.get_cpu_info()
    try:
        output_string = add_param_to_output(output_string, 'brand', cpu_info['brand'])
        output_string = add_param_to_output(output_string, 'vendor id', cpu_info['vendor_id'])
        output_string = add_param_to_output(output_string, 'model', cpu_info['model'])
        output_string = add_param_to_output(output_string, 'family', cpu_info['family'])
        output_string = add_param_to_output(output_string, 'bits', cpu_info['bits'])
        output_string = add_param_to_output(output_string, 'architecture', cpu_info['arch'])
        output_string = add_param_to_output(output_string, 'cores', cpu_info['count'])
        output_string = add_param_to_output(output_string, 'advertised Hz', cpu_info['hz_advertised'])
        output_string = add_param_to_output(output_string, 'actual Hz', cpu_info['hz_actual'])
        output_string = add_param_to_output(output_string, 'l2 cache size', cpu_info['l2_cache_size'])
    except Exception:
        output_string += 'Some CPU information cannot be displayed\n'
    output_string += '----------------------------------------'
    neon_logger.display(output_string)

def display_platform_information():
    if False:
        while True:
            i = 10
    '\n    Display platform information.\n    '
    import platform
    output_string = '\n-- INFORMATION: PLATFORM & OS ---------\n'
    try:
        output_string = add_param_to_output(output_string, 'OS', platform.platform())
        output_string = add_param_to_output(output_string, 'OS release version', platform.version())
        output_string = add_param_to_output(output_string, 'machine', platform.machine())
        output_string = add_param_to_output(output_string, 'node', platform.node())
        output_string = add_param_to_output(output_string, 'python version', platform.python_version())
        output_string = add_param_to_output(output_string, 'python build', platform.python_build())
        output_string = add_param_to_output(output_string, 'python compiler', platform.python_compiler())
    except Exception:
        output_string += 'Some platform information cannot be displayed\n'
    output_string += '----------------------------------------'
    neon_logger.display(output_string)