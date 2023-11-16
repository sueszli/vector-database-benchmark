import os
from spack.util.executable import Executable, which

def compile_c_and_execute(source_file, include_flags, link_flags):
    if False:
        i = 10
        return i + 15
    'Compile C @p source_file with @p include_flags and @p link_flags,\n    run and return the output.\n    '
    cc = which('cc')
    flags = include_flags
    flags.extend([source_file])
    cc('-c', *flags)
    name = os.path.splitext(os.path.basename(source_file))[0]
    cc('-o', 'check', '%s.o' % name, *link_flags)
    check = Executable('./check')
    return check(output=str)

def compare_output(current_output, blessed_output):
    if False:
        i = 10
        return i + 15
    'Compare blessed and current output of executables.'
    if not current_output == blessed_output:
        print('Produced output does not match expected output.')
        print('Expected output:')
        print('-' * 80)
        print(blessed_output)
        print('-' * 80)
        print('Produced output:')
        print('-' * 80)
        print(current_output)
        print('-' * 80)
        raise RuntimeError('Ouput check failed.', 'See spack_output.log for details')

def compare_output_file(current_output, blessed_output_file):
    if False:
        return 10
    'Same as above, but when the blessed output is given as a file.'
    with open(blessed_output_file, 'r') as f:
        blessed_output = f.read()
    compare_output(current_output, blessed_output)