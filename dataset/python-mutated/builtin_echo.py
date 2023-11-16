import BoostBuild

def test_echo(name):
    if False:
        print('Hello World!')
    t = BoostBuild.Tester(['-ffile.jam'], pass_toolset=0)
    t.write('file.jam', '%s ;\nUPDATE ;\n' % name)
    t.run_build_system(stdout='\n')
    t.write('file.jam', '%s a message ;\nUPDATE ;\n' % name)
    t.run_build_system(stdout='a message\n')
    t.cleanup()
test_echo('ECHO')
test_echo('Echo')
test_echo('echo')