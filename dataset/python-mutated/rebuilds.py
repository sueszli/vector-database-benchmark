import BoostBuild

def wait_for_bar(t):
    if False:
        print('Hello World!')
    "\n      Wait to make the test system correctly recognize the 'bar' file as\n    touched after the next build run. Without the wait, the next build run may\n    rebuild the 'bar' file with the new and the old file modification timestamp\n    too close to each other - which could, depending on the currently supported\n    file modification timestamp resolution, be detected as 'no change' by the\n    testing system.\n\n    "
    t.wait_for_time_change('bar', touch=False)
t = BoostBuild.Tester(['-ffile.jam', '-d+3', '-d+12', '-d+13'], pass_toolset=0)
t.write('file.jam', 'rule make\n{\n    DEPENDS $(<) : $(>) ;\n    DEPENDS all : $(<) ;\n}\nactions make\n{\n    echo "******" making $(<) from $(>) "******"\n    echo made from $(>) > $(<)\n}\n\nmake aux1 : bar ;\nmake foo : bar ;\nREBUILDS foo : bar ;\nmake bar : baz ;\nmake aux2 : bar ;\n')
t.write('baz', 'nothing')
t.run_build_system(['bar'])
t.expect_addition('bar')
t.expect_nothing_more()
wait_for_bar(t)
t.run_build_system(['foo'])
t.expect_touch('bar')
t.expect_addition('foo')
t.expect_nothing_more()
t.run_build_system()
t.expect_addition(['aux1', 'aux2'])
t.expect_nothing_more()
t.touch('bar')
wait_for_bar(t)
t.run_build_system()
t.expect_touch(['foo', 'bar', 'aux1', 'aux2'])
t.expect_nothing_more()
t.cleanup()