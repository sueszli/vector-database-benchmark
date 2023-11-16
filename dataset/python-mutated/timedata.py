import BoostBuild
import re

def basic_jam_action_test():
    if False:
        print('Hello World!')
    'Tests basic Jam action timing support.'
    t = BoostBuild.Tester(pass_toolset=0)
    t.write('file.jam', 'rule time\n{\n    DEPENDS $(<) : $(>) ;\n    __TIMING_RULE__ on $(>) = record_time $(<) ;\n    DEPENDS all : $(<) ;\n}\n\nactions time\n{\n    echo $(>) user: $(__USER_TIME__) system: $(__SYSTEM_TIME__) clock: $(__CLOCK_TIME__)\n    echo timed from $(>) >> $(<)\n}\n\nrule record_time ( target : source : start end user system clock )\n{\n    __USER_TIME__ on $(target) = $(user) ;\n    __SYSTEM_TIME__ on $(target) = $(system) ;\n    __CLOCK_TIME__ on $(target) = $(clock) ;\n}\n\nrule make\n{\n    DEPENDS $(<) : $(>) ;\n}\n\nactions make\n{\n    echo made from $(>) >> $(<)\n}\n\ntime foo : bar ;\nmake bar : baz ;\n')
    t.write('baz', 'nothing')
    expected_output = '\\.\\.\\.found 4 targets\\.\\.\\.\n\\.\\.\\.updating 2 targets\\.\\.\\.\nmake bar\ntime foo\nbar +user: [0-9\\.]+ +system: +[0-9\\.]+ +clock: +[0-9\\.]+ *\n\\.\\.\\.updated 2 targets\\.\\.\\.$\n'
    t.run_build_system(['-ffile.jam', '-d+1'], stdout=expected_output, match=lambda actual, expected: re.search(expected, actual, re.DOTALL))
    t.expect_addition('foo')
    t.expect_addition('bar')
    t.expect_nothing_more()
    t.cleanup()

def boost_build_testing_support_timing_rule():
    if False:
        return 10
    '\n      Tests the target build timing rule provided by the Boost Build testing\n    support system.\n\n    '
    t = BoostBuild.Tester(use_test_config=False)
    t.write('aaa.cpp', 'int main() {}\n')
    t.write('jamroot.jam', 'import testing ;\nexe my-exe : aaa.cpp ;\ntime my-time : my-exe ;\n')
    t.run_build_system()
    t.expect_addition('bin/$toolset/debug*/aaa.obj')
    t.expect_addition('bin/$toolset/debug*/my-exe.exe')
    t.expect_addition('bin/$toolset/debug*/my-time.time')
    t.expect_content_lines('bin/$toolset/debug*/my-time.time', 'user: *[0-9] seconds')
    t.expect_content_lines('bin/$toolset/debug*/my-time.time', 'system: *[0-9] seconds')
    t.expect_content_lines('bin/$toolset/debug*/my-time.time', 'clock: *[0-9] seconds')
    t.cleanup()

def boost_build_testing_support_timing_rule_with_spaces_in_names():
    if False:
        i = 10
        return i + 15
    '\n      Tests the target build timing rule provided by the Boost Build testing\n    support system when used with targets contining spaces in their names.\n\n    '
    t = BoostBuild.Tester(use_test_config=False)
    t.write('aaa bbb.cpp', 'int main() {}\n')
    t.write('jamroot.jam', 'import testing ;\nexe "my exe" : "aaa bbb.cpp" ;\ntime "my time" : "my exe" ;\n')
    t.run_build_system()
    t.expect_addition('bin/$toolset/debug*/aaa bbb.obj')
    t.expect_addition('bin/$toolset/debug*/my exe.exe')
    t.expect_addition('bin/$toolset/debug*/my time.time')
    t.expect_content_lines('bin/$toolset/debug*/my time.time', 'user: *')
    t.expect_content_lines('bin/$toolset/debug*/my time.time', 'system: *')
    t.cleanup()
basic_jam_action_test()
boost_build_testing_support_timing_rule()
boost_build_testing_support_timing_rule_with_spaces_in_names()