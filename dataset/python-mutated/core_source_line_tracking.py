import BoostBuild

def test_eof_in_string():
    if False:
        i = 10
        return i + 15
    t = BoostBuild.Tester(pass_toolset=False)
    t.write('file.jam', '\n\n\naaa = "\n\n\n\n\n\n')
    t.run_build_system(['-ffile.jam'], status=1)
    t.expect_output_lines('file.jam:4: unmatched " in string at keyword =')
    t.expect_output_lines('file.jam:4: syntax error at EOF')
    t.cleanup()

def test_error_missing_argument(eof):
    if False:
        return 10
    "\n      This use case used to cause a missing argument error to be reported in\n    module '(builtin)' in line -1 when the input file did not contain a\n    trailing newline.\n\n    "
    t = BoostBuild.Tester(pass_toolset=False)
    t.write('file.jam', 'rule f ( param ) { }\nf ;%s' % __trailing_newline(eof))
    t.run_build_system(['-ffile.jam'], status=1)
    t.expect_output_lines('file.jam:2: in module scope')
    t.expect_output_lines("file.jam:1:see definition of rule 'f' being called")
    t.cleanup()

def test_error_syntax(eof):
    if False:
        print('Hello World!')
    t = BoostBuild.Tester(pass_toolset=False)
    t.write('file.jam', 'ECHO%s' % __trailing_newline(eof))
    t.run_build_system(['-ffile.jam'], status=1)
    t.expect_output_lines('file.jam:1: syntax error at EOF')
    t.cleanup()

def test_traceback():
    if False:
        i = 10
        return i + 15
    t = BoostBuild.Tester(pass_toolset=False)
    t.write('file.jam', 'NOTFILE all ;\nECHO [ BACKTRACE ] ;')
    t.run_build_system(['-ffile.jam'])
    t.expect_output_lines('file.jam 2  module scope')
    t.cleanup()

def __trailing_newline(eof):
    if False:
        print('Hello World!')
    '\n      Helper function returning an empty string or a newling character to\n    append to the current output line depending on whether we want that line to\n    be the last line in the file (eof == True) or not (eof == False).\n\n    '
    if eof:
        return ''
    return '\n'
test_error_missing_argument(eof=False)
test_error_missing_argument(eof=True)
test_error_syntax(eof=False)
test_error_syntax(eof=True)
test_traceback()
test_eof_in_string()