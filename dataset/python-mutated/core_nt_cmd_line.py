import BoostBuild
import os
import re
import sys
executable = sys.executable.replace('\\', '/')
if ' ' in executable:
    executable = '"%s"' % executable

def string_of_length(n):
    if False:
        for i in range(10):
            print('nop')
    if n <= 0:
        return ''
    n -= 1
    y = ['', '$(1x10-1)', '$(10x10-1)', '$(100x10-1)', '$(1000x10-1)']
    result = []
    for i in reversed(xrange(5)):
        (x, n) = divmod(n, 10 ** i)
        result += [y[i]] * x
    result.append('x')
    return ' '.join(result)

def test_raw_empty():
    if False:
        for i in range(10):
            print('nop')
    whitespace_in = '  \n\n\r\r\x0b\x0b\t\t   \t   \r\r   \n\n'
    whitespace_out = whitespace_in.replace('\r\n', '\n').replace('\n', '\r\n')
    t = BoostBuild.Tester(['-d2', '-d+4'], pass_toolset=0, use_test_config=False)
    t.write('file.jam', 'actions do_empty {%s}\nJAMSHELL = %% ;\ndo_empty all ;\n' % whitespace_in)
    t.run_build_system(['-ffile.jam'], universal_newlines=False)
    t.expect_output_lines('do_empty all')
    t.expect_output_lines('Executing raw command directly', False)
    if '\r\n%s\r\n' % whitespace_out not in t.stdout():
        BoostBuild.annotation('failure', 'Whitespace action content not found on stdout.')
        t.fail_test(1, dump_difference=False)
    t.cleanup()

def test_raw_nt(n=None, error=False):
    if False:
        for i in range(10):
            print('nop')
    t = BoostBuild.Tester(['-d1', '-d+4'], pass_toolset=0, use_test_config=False)
    cmd_prefix = '%s -c "print(\'XXX: ' % executable
    cmd_suffix = '\')"'
    cmd_extra_length = len(cmd_prefix) + len(cmd_suffix)
    if n == None:
        n = cmd_extra_length
    data_length = n - cmd_extra_length
    if data_length < 0:
        BoostBuild.annotation('failure', 'Can not construct Windows command of desired length. Requested command length\ntoo short for the current test configuration.\n    Requested command length: %d\n    Minimal supported command length: %d\n' % (n, cmd_extra_length))
        t.fail_test(1, dump_difference=False)
    t.write('file.jam', 'ten = 0 1 2 3 4 5 6 7 8 9 ;\n\n1x10-1 = 123456789 ;\n10x10-1 = $(ten)12345678 ;\n100x10-1 = $(ten)$(ten)1234567 ;\n1000x10-1 = $(ten)$(ten)$(ten)123456 ;\n\nactions do_echo\n{\n    %s%s%s\n}\nJAMSHELL = %% ;\ndo_echo all ;\n' % (cmd_prefix, string_of_length(data_length), cmd_suffix))
    if error:
        expected_status = 1
    else:
        expected_status = 0
    t.run_build_system(['-ffile.jam'], status=expected_status)
    if error:
        t.expect_output_lines('Executing raw command directly', False)
        t.expect_output_lines('do_echo action is too long (%d, max 32766):' % n)
        t.expect_output_lines('XXX: *', False)
    else:
        t.expect_output_lines('Executing raw command directly')
        t.expect_output_lines('do_echo action is too long*', False)
        m = re.search('^XXX: (.*)$', t.stdout(), re.MULTILINE)
        if not m:
            BoostBuild.annotation('failure', "Expected output line starting with 'XXX: ' not found.")
            t.fail_test(1, dump_difference=False)
        if len(m.group(1)) != data_length:
            BoostBuild.annotation('failure', 'Unexpected output data length.\n    Expected: %d\n    Received: %d' % (n, len(m.group(1))))
            t.fail_test(1, dump_difference=False)
    t.cleanup()

def test_raw_to_shell_fallback_nt():
    if False:
        print('Hello World!')
    t = BoostBuild.Tester(['-d1', '-d+4'], pass_toolset=0, use_test_config=False)
    cmd_prefix = '%s -c print(' % executable
    cmd_suffix = ')'
    t.write('file_multiline.jam', 'actions do_multiline\n{\n    echo one\n\n\n    echo two\n}\nJAMSHELL = % ;\ndo_multiline all ;\n')
    t.run_build_system(['-ffile_multiline.jam'])
    t.expect_output_lines('do_multiline all')
    t.expect_output_lines('one')
    t.expect_output_lines('two')
    t.expect_output_lines('Executing raw command directly', False)
    t.expect_output_lines('Executing using a command file and the shell: cmd.exe /Q/C')
    t.write('file_redirect.jam', 'actions do_redirect { echo one > two.txt }\nJAMSHELL = % ;\ndo_redirect all ;\n')
    t.run_build_system(['-ffile_redirect.jam'])
    t.expect_output_lines('do_redirect all')
    t.expect_output_lines('one', False)
    t.expect_output_lines('Executing raw command directly', False)
    t.expect_output_lines('Executing using a command file and the shell: cmd.exe /Q/C')
    t.expect_addition('two.txt')
    t.write('file_pipe.jam', 'actions do_pipe\n{\n    echo one | echo two\n}\nJAMSHELL = % ;\ndo_pipe all ;\n')
    t.run_build_system(['-ffile_pipe.jam'])
    t.expect_output_lines('do_pipe all')
    t.expect_output_lines('one*', False)
    t.expect_output_lines('two')
    t.expect_output_lines('Executing raw command directly', False)
    t.expect_output_lines('Executing using a command file and the shell: cmd.exe /Q/C')
    t.write('file_single_quoted.jam', "actions do_single_quoted { %s'5>10'%s }\nJAMSHELL = %% ;\ndo_single_quoted all ;\n" % (cmd_prefix, cmd_suffix))
    t.run_build_system(['-ffile_single_quoted.jam'])
    t.expect_output_lines('do_single_quoted all')
    t.expect_output_lines('5>10')
    t.expect_output_lines('Executing raw command directly')
    t.expect_output_lines('Executing using a command file and the shell: cmd.exe /Q/C', False)
    t.expect_nothing_more()
    t.write('file_double_quoted.jam', 'actions do_double_quoted { %s"5>10"%s }\nJAMSHELL = %% ;\ndo_double_quoted all ;\n' % (cmd_prefix, cmd_suffix))
    t.run_build_system(['-ffile_double_quoted.jam'])
    t.expect_output_lines('do_double_quoted all')
    t.expect_output_lines('False')
    t.expect_output_lines('Executing raw command directly')
    t.expect_output_lines('Executing using a command file and the shell: cmd.exe /Q/C', False)
    t.expect_nothing_more()
    t.write('file_escaped_quote.jam', 'actions do_escaped_quote { %s\\"5>10\\"%s }\nJAMSHELL = %% ;\ndo_escaped_quote all ;\n' % (cmd_prefix, cmd_suffix))
    t.run_build_system(['-ffile_escaped_quote.jam'])
    t.expect_output_lines('do_escaped_quote all')
    t.expect_output_lines('5>10')
    t.expect_output_lines('Executing raw command directly', False)
    t.expect_output_lines('Executing using a command file and the shell: cmd.exe /Q/C')
    t.expect_nothing_more()
    t.cleanup()
if os.name == 'nt':
    test_raw_empty()
    test_raw_nt()
    test_raw_nt(255)
    test_raw_nt(1000)
    test_raw_nt(8000)
    test_raw_nt(8191)
    test_raw_nt(8192)
    test_raw_nt(10000)
    test_raw_nt(30000)
    test_raw_nt(32766)
    test_raw_nt(32767, error=True)
    test_raw_nt(40000, error=True)
    test_raw_nt(100001, error=True)
    test_raw_to_shell_fallback_nt()