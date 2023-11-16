import BoostBuild
t = BoostBuild.Tester(use_test_config=False)
t.write('jamroot.jam', '')
t.write('lib/c.cpp', 'int bar() { return 0; }\n')
t.write('lib/jamfile.jam', 'static-lib auxilliary1 : c.cpp ;\nlib auxilliary2 : c.cpp ;\n')

def reset():
    if False:
        for i in range(10):
            print('nop')
    t.rm('lib/bin')
t.run_build_system(subdir='lib')
t.expect_addition('lib/bin/$toolset/debug*/' * BoostBuild.List('c.obj auxilliary1.lib auxilliary2.dll'))
t.expect_nothing_more()
reset()
t.run_build_system(['link=shared'], subdir='lib')
t.expect_addition('lib/bin/$toolset/debug*/' * BoostBuild.List('c.obj auxilliary1.lib auxilliary2.dll'))
t.expect_nothing_more()
reset()
t.run_build_system(['link=static'], subdir='lib')
t.expect_addition('lib/bin/$toolset/debug*/' * BoostBuild.List('c.obj auxilliary1.lib auxilliary2.lib'))
t.expect_nothing_more()
t.cleanup()