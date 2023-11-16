import BoostBuild

def test_basic():
    if False:
        return 10
    'Tests that feature=value works'
    t = BoostBuild.Tester()
    t.write('Jamroot.jam', '\n    import feature : feature ;\n    import toolset : flags ;\n    feature f1 : 1 2 ;\n    make output.txt : : @run ;\n    flags run OPTIONS <f1> ;\n    actions run { echo $(OPTIONS) > $(<) }\n    ')
    t.run_build_system(['f1=2'])
    t.expect_content('bin/*/output.txt', '2')
    t.cleanup()

def test_implicit():
    if False:
        return 10
    'Tests that implicit features can be named without a feature'
    t = BoostBuild.Tester()
    t.write('Jamroot.jam', '\n    import feature : feature ;\n    import toolset : flags ;\n    feature f1 : v1 v2 : implicit ;\n    make output.txt : : @run ;\n    flags run OPTIONS <f1> ;\n    actions run { echo $(OPTIONS) > $(<) }\n    ')
    t.run_build_system(['v2'])
    t.expect_content('bin/*/output.txt', 'v2')
    t.cleanup()

def test_optional():
    if False:
        return 10
    'Tests that feature= works for optional features'
    t = BoostBuild.Tester()
    t.write('Jamroot.jam', '\n    import feature : feature ;\n    import toolset : flags ;\n    feature f1 : 1 2 : optional ;\n    make output.txt : : @run ;\n    flags run OPTIONS <f1> ;\n    actions run { echo b $(OPTIONS) > $(<) }\n    ')
    t.run_build_system(['f1='])
    t.expect_content('bin/*/output.txt', 'b')
    t.cleanup()

def test_free():
    if False:
        i = 10
        return i + 15
    'Free features named on the command line apply to all targets\n    everywhere.  Free features can contain any characters, even those\n    that have a special meaning.'
    t = BoostBuild.Tester()
    t.write('Jamroot.jam', '\n    import feature : feature ;\n    import toolset : flags ;\n    feature f1 : : free ;\n    make output1.txt : : @run : <dependency>output2.txt ;\n    make output2.txt : : @run ;\n    explicit output2.txt ;\n    flags run OPTIONS <f1> ;\n    actions run { echo $(OPTIONS) > $(<) }\n    ')
    t.run_build_system(['f1=x,/:-'])
    t.expect_content('bin*/output1.txt', 'x,/:-')
    t.expect_content('bin*/output2.txt', 'x,/:-')
    t.cleanup()

def test_subfeature():
    if False:
        return 10
    'Subfeatures should be expressed as feature=value-subvalue'
    t = BoostBuild.Tester()
    t.write('Jamroot.jam', '\n    import feature : feature subfeature ;\n    import toolset : flags ;\n    feature f1 : 1 2 ;\n    subfeature f1 2 : sub : x y ;\n    make output.txt : : @run ;\n    flags run OPTIONS <f1-2:sub> ;\n    actions run { echo $(OPTIONS) > $(<) }\n    ')
    t.run_build_system(['f1=2-y'])
    t.expect_content('bin/*/output.txt', 'y')
    t.cleanup()

def test_multiple_values():
    if False:
        return 10
    'Multiple values of a feature can be given in a comma-separated list'
    t = BoostBuild.Tester()
    t.write('Jamroot.jam', '\n    import feature : feature ;\n    import toolset : flags ;\n    feature f1 : 1 2 3 ;\n    make output.txt : : @run ;\n    flags run OPTIONS <f1> ;\n    actions run { echo $(OPTIONS) > $(<) }\n    ')
    t.run_build_system(['f1=2,3'])
    t.expect_content('bin*/f1-2*/output.txt', '2')
    t.expect_content('bin*/f1-3*/output.txt', '3')
    t.cleanup()

def test_multiple_properties():
    if False:
        i = 10
        return i + 15
    'Multiple properties can be grouped with /'
    t = BoostBuild.Tester()
    t.write('Jamroot.jam', '\n    import feature : feature ;\n    import toolset : flags ;\n    feature f1 : 1 2 ;\n    feature f2 : 3 4 ;\n    make output.txt : : @run ;\n    flags run OPTIONS <f1> ;\n    flags run OPTIONS <f2> ;\n    actions run { echo $(OPTIONS) > $(<) }\n    ')
    t.run_build_system(['f1=2/f2=4'])
    t.expect_content('bin/*/output.txt', '2 4')
    t.cleanup()

def test_cross_product():
    if False:
        print('Hello World!')
    'If multiple properties are specified on the command line\n    we expand to every possible maximum set of non-conflicting features.\n    This test should be run after testing individual components in\n    isolation.'
    t = BoostBuild.Tester()
    t.write('Jamroot.jam', '\n    import feature : feature ;\n    import toolset : flags ;\n    # Make features symmetric to make the paths easier to distingush\n    feature f1 : 11 12 13 14 15 : symmetric ;\n    feature f2 : 21 22 23 : symmetric ;\n    feature f3 : v1 v2 v3 v4 : implicit symmetric ;\n    feature f4 : : free ;\n    make output.txt : : @run ;\n    flags run OPTIONS <f1> ;\n    flags run OPTIONS <f2> ;\n    flags run OPTIONS <f3> ;\n    flags run OPTIONS <f4> ;\n    actions run { echo $(OPTIONS) > $(<) }\n    ')
    t.run_build_system(['f1=12,13/f2=22', 'v2', 'v3', 'f1=14', 'f2=23', 'f4=xxx', 'f4=yyy', 'v4/f1=15/f4=zzz'])
    t.expect_content('bin*/v2*/f1-12/f2-22*/output.txt', '12 22 v2 xxx yyy')
    t.expect_addition('bin*/v2*/f1-12/f2-22*/output.txt')
    t.expect_content('bin*/v2*/f1-13/f2-22*/output.txt', '13 22 v2 xxx yyy')
    t.expect_addition('bin*/v2*/f1-13/f2-22*/output.txt')
    t.expect_content('bin*/v2*/f1-14/f2-23*/output.txt', '14 23 v2 xxx yyy')
    t.expect_addition('bin*/v2*/f1-14/f2-23*/output.txt')
    t.expect_content('bin*/v3*/f1-12/f2-22*/output.txt', '12 22 v3 xxx yyy')
    t.expect_addition('bin*/v3*/f1-12/f2-22*/output.txt')
    t.expect_content('bin*/v3*/f1-13/f2-22*/output.txt', '13 22 v3 xxx yyy')
    t.expect_addition('bin*/v3*/f1-13/f2-22*/output.txt')
    t.expect_content('bin*/v3*/f1-14/f2-23*/output.txt', '14 23 v3 xxx yyy')
    t.expect_addition('bin*/v3*/f1-14/f2-23*/output.txt')
    t.expect_content('bin*/v4*/f1-15/f2-23*/output.txt', '15 23 v4 xxx yyy zzz')
    t.expect_addition('bin*/v4*/f1-15/f2-23*/output.txt')
    t.expect_nothing_more()
    t.cleanup()
test_basic()
test_implicit()
test_optional()
test_free()
test_subfeature()
test_multiple_values()
test_multiple_properties()
test_cross_product()