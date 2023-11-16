import BoostBuild

def test_basic():
    if False:
        i = 10
        return i + 15
    t = BoostBuild.Tester(pass_toolset=0)
    t.write('file.jam', '    actions fail\n    {\n        invalid-dd0eeb5899734622\n    }\n\n    FAIL_EXPECTED t1 ;\n    fail t1 ;\n\n    UPDATE t1 ;\n    ')
    t.run_build_system(['-ffile.jam'])
    t.expect_output_lines('...failed*', False)
    t.expect_nothing_more()
    t.cleanup()

def test_error():
    if False:
        print('Hello World!')
    t = BoostBuild.Tester(pass_toolset=0)
    t.write('file.jam', '    actions pass\n    {\n        echo okay >$(<)\n    }\n\n    FAIL_EXPECTED t1 ;\n    pass t1 ;\n\n    UPDATE t1 ;\n    ')
    t.run_build_system(['-ffile.jam'], status=1)
    t.expect_output_lines('...failed pass t1...')
    t.expect_nothing_more()
    t.cleanup()

def test_multiple_actions():
    if False:
        for i in range(10):
            print('nop')
    'FAIL_EXPECTED targets are considered to pass if the first\n    updating action fails.  Further actions will be skipped.'
    t = BoostBuild.Tester(pass_toolset=0)
    t.write('file.jam', '    actions fail\n    {\n        invalid-dd0eeb5899734622\n    }\n\n    actions pass\n    {\n         echo okay >$(<)\n    }\n\n    FAIL_EXPECTED t1 ;\n    fail t1 ;\n    pass t1 ;\n\n    UPDATE t1 ;\n    ')
    t.run_build_system(['-ffile.jam', '-d1'])
    t.expect_output_lines('...failed*', False)
    t.expect_output_lines('pass t1', False)
    t.expect_nothing_more()
    t.cleanup()

def test_quitquick():
    if False:
        i = 10
        return i + 15
    'Tests that FAIL_EXPECTED targets do not cause early exit\n    on failure.'
    t = BoostBuild.Tester(pass_toolset=0)
    t.write('file.jam', '    actions fail\n    {\n        invalid-dd0eeb5899734622\n    }\n\n    actions pass\n    {\n        echo okay >$(<)\n    }\n\n    FAIL_EXPECTED t1 ;\n    fail t1 ;\n\n    pass t2 ;\n\n    UPDATE t1 t2 ;\n    ')
    t.run_build_system(['-ffile.jam', '-q', '-d1'])
    t.expect_output_lines('pass t2')
    t.expect_addition('t2')
    t.expect_nothing_more()
    t.cleanup()

def test_quitquick_error():
    if False:
        i = 10
        return i + 15
    'FAIL_EXPECTED targets should cause early exit if they unexpectedly pass.'
    t = BoostBuild.Tester(pass_toolset=0)
    t.write('file.jam', '    actions pass\n    {\n        echo okay >$(<)\n    }\n\n    FAIL_EXPECTED t1 ;\n    pass t1 ;\n    pass t2 ;\n\n    UPDATE t1 t2 ;\n    ')
    t.run_build_system(['-ffile.jam', '-q', '-d1'], status=1)
    t.expect_output_lines('pass t2', False)
    t.expect_nothing_more()
    t.cleanup()
test_basic()
test_error()
test_multiple_actions()
test_quitquick()
test_quitquick_error()