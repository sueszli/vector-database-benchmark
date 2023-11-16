import BoostBuild

def testSORTCorrectness():
    if False:
        i = 10
        return i + 15
    "Testing that Boost Jam's SORT builtin rule actually sorts correctly."
    t = BoostBuild.Tester(['-ftest.jam', '-d1'], pass_toolset=False, use_test_config=False)
    t.write('test.jam', 'NOCARE all ;\nsource-data = 1 8 9 2 7 3 4 7 1 27 27 9 98 98 1 1 4 5 6 2 3 4 8 1 -2 -2 0 0 0 ;\ntarget-data = -2 -2 0 0 0 1 1 1 1 1 2 2 27 27 3 3 4 4 4 5 6 7 7 8 8 9 9 98 98 ;\nECHO "starting up" ;\nsorted-data = [ SORT $(source-data) ] ;\nECHO "done" ;\nif $(sorted-data) != $(target-data)\n{\n    ECHO "Source       :" $(source-data) ;\n    ECHO "Expected     :" $(target-data) ;\n    ECHO "SORT returned:" $(sorted-data) ;\n    EXIT "SORT error" : -2 ;\n}\n')
    t.run_build_system()
    t.expect_output_lines('starting up')
    t.expect_output_lines('done')
    t.expect_output_lines('SORT error', False)
    t.cleanup()

def testSORTDuration():
    if False:
        while True:
            i = 10
    "\n      Regression test making sure Boost Jam's SORT builtin rule does not get\n    quadratic behaviour again in this use case.\n\n    "
    t = BoostBuild.Tester(['-ftest.jam', '-d1'], pass_toolset=False, use_test_config=False)
    f = open(t.workpath('test.jam'), 'w')
    (print >> f, 'data = ')
    for i in range(0, 20000):
        if i % 2:
            (print >> f, '"aaa"')
        else:
            (print >> f, '"bbb"')
    (print >> f, ';\n\nECHO "starting up" ;\nsorted = [ SORT $(data) ] ;\nECHO "done" ;\nNOCARE all ;\n')
    f.close()
    t.run_build_system(expected_duration=1)
    t.expect_output_lines('starting up')
    t.expect_output_lines('done')
    t.cleanup()
testSORTCorrectness()
testSORTDuration()