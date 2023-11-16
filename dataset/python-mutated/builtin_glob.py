import os
import BoostBuild

def test_glob(files, glob, expected, setup=''):
    if False:
        while True:
            i = 10
    t = BoostBuild.Tester(['-ffile.jam'], pass_toolset=0)
    t.write('file.jam', setup + '\n    for local p in [ SORT %s ]\n    {\n        ECHO $(p) ;\n    }\n    UPDATE ;\n    ' % glob)
    for f in files:
        t.write(f, '')
    expected = [os.path.join(*p.split('/')) for p in expected]
    expected.sort()
    t.run_build_system(stdout='\n'.join(expected + ['']))
    t.cleanup()
test_glob([], '[ GLOB : ]', [])
test_glob([], '[ GLOB . : ]', [])
test_glob([], '[ GLOB : * ]', [])
test_glob([], '[ GLOB . : * ]', ['./file.jam'])
test_glob([], '[ GLOB . : file*.jam ]', ['./file.jam'])
test_glob([], '[ GLOB . : f*am ]', ['./file.jam'])
test_glob([], '[ GLOB . : fi?e.?am ]', ['./file.jam'])
test_glob([], '[ GLOB . : fi?.jam ]', [])
test_glob([], '[ GLOB . : "[f][i][l][e].jam" ]', ['./file.jam'])
test_glob([], '[ GLOB . : "[fghau][^usdrwe][k-o][^f-s].jam" ]', ['./file.jam'])
test_glob([], '[ GLOB . : \\f\\i\\l\\e.jam ]', ['./file.jam'])
test_glob(['test.txt'], '[ GLOB . : * ]', ['./file.jam', './test.txt'])
test_glob(['dir1/dir2/test.txt'], '[ GLOB dir1 : * ]', ['dir1/dir2'])
test_glob([], '[ GLOB dir1 : * ] ', [])
test_glob(['dir1/file1.txt', 'dir2/file1.txt', 'dir2/file2.txt'], '[ GLOB dir1 dir2 : file1* file2* ]', ['dir1/file1.txt', 'dir2/file1.txt', 'dir2/file2.txt'])
test_glob(['dir/test.txt'], '[ GLOB dir/. : test.txt ]', ['dir/./test.txt'])
test_glob(['dir/test.txt'], '[ GLOB dir/.. : file.jam ]', ['dir/../file.jam'])
test_glob(['TEST.TXT'], '[ GLOB . : TEST.TXT ]', ['./TEST.TXT'])
case_insensitive = os.path.normcase('FILE') == 'file'
if case_insensitive:
    test_glob(['TEST.TXT'], '[ GLOB . : test.txt ]', ['./TEST.TXT'])
    test_glob(['D1/D2/TEST.TXT'], '[ GLOB D1/./D2 : test.txt ]', ['D1/./D2/TEST.TXT'])
    test_glob(['D1/TEST.TXT', 'TEST.TXT'], '[ GLOB D1/../D1 : test.txt ]', ['D1/../D1/TEST.TXT'])
    test_glob(['D1/D2/TEST.TXT'], '[ GLOB d1/d2 : test.txt ]', ['D1/D2/TEST.TXT'], 'GLOB . : * ;')