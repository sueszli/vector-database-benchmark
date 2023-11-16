import BoostBuild

def test_invalid(params, expected_error_line):
    if False:
        i = 10
        return i + 15
    t = BoostBuild.Tester(pass_toolset=0)
    t.write('file.jam', 'SPLIT_BY_CHARACTERS %s ;' % params)
    t.run_build_system(['-ffile.jam'], status=1)
    t.expect_output_lines('[*] %s' % expected_error_line)
    t.cleanup()

def test_valid():
    if False:
        for i in range(10):
            print('nop')
    t = BoostBuild.Tester(pass_toolset=0)
    t.write('jamroot.jam', 'import assert ;\n\nassert.result FooBarBaz : SPLIT_BY_CHARACTERS FooBarBaz : "" ;\nassert.result FooBarBaz : SPLIT_BY_CHARACTERS FooBarBaz : x ;\nassert.result FooBa Baz : SPLIT_BY_CHARACTERS FooBarBaz : r ;\nassert.result FooBa Baz : SPLIT_BY_CHARACTERS FooBarBaz : rr ;\nassert.result FooBa Baz : SPLIT_BY_CHARACTERS FooBarBaz : rrr ;\nassert.result FooB rB z : SPLIT_BY_CHARACTERS FooBarBaz : a ;\nassert.result FooB B z : SPLIT_BY_CHARACTERS FooBarBaz : ar ;\nassert.result ooBarBaz : SPLIT_BY_CHARACTERS FooBarBaz : F ;\nassert.result FooBarBa : SPLIT_BY_CHARACTERS FooBarBaz : z ;\nassert.result ooBarBa : SPLIT_BY_CHARACTERS FooBarBaz : Fz ;\nassert.result F B rB z : SPLIT_BY_CHARACTERS FooBarBaz : oa ;\nassert.result Alib b : SPLIT_BY_CHARACTERS Alibaba : oa ;\nassert.result libaba : SPLIT_BY_CHARACTERS Alibaba : oA ;\nassert.result : SPLIT_BY_CHARACTERS FooBarBaz : FooBarBaz ;\nassert.result : SPLIT_BY_CHARACTERS FooBarBaz : FoBarz ;\n\n# Questionable results - should they return an empty string or an empty list?\nassert.result : SPLIT_BY_CHARACTERS "" : "" ;\nassert.result : SPLIT_BY_CHARACTERS "" : x ;\nassert.result : SPLIT_BY_CHARACTERS "" : r ;\nassert.result : SPLIT_BY_CHARACTERS "" : rr ;\nassert.result : SPLIT_BY_CHARACTERS "" : rrr ;\nassert.result : SPLIT_BY_CHARACTERS "" : oa ;\n')
    t.run_build_system()
    t.cleanup()
test_invalid('', 'missing argument string')
test_invalid('Foo', 'missing argument delimiters')
test_invalid(': Bar', 'missing argument string')
test_invalid('a : b : c', 'extra argument c')
test_invalid('a b : c', 'extra argument b')
test_invalid('a : b c', 'extra argument c')
test_valid()