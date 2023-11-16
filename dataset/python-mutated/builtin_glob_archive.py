import os
import sys
import StringIO
import BoostBuild
vms = os.name == 'posix' and sys.platform == 'OpenVMS'
t = BoostBuild.Tester()
sources = {'a.cpp': ['a'], 'b.cpp': ['b'], 'b_match.cpp': ['b_match'], 'c/nopath_check.cpp': ['nopath_check'], 'CaseCheck.cpp': ['CaseCheck'], 'seq_check1.cpp': ['seq_check1'], 'seq_check2.cpp': ['seq_check2'], 'seq_check3.cpp': ['seq_check3'], 'symbols_check.c': ['symbol', 'symbol_match'], 'members_and_symbols_check.c': ['member_and_symbol_match'], 'symbol_case_check.c': ['SymbolCaseCheck'], 'main_check.cpp': ['main']}

def create_sources(path, sources):
    if False:
        print('Hello World!')
    for s in sources:
        f = os.path.join(path, s)
        t.write(f, '')
        output = StringIO.StringIO()
        for sym in sources[s]:
            output.write('int %s() { return 0; }\n' % sym)
        t.write(f, output.getvalue())

def setup_archive(name, sources):
    if False:
        for i in range(10):
            print('nop')
    global archive
    global obj_suffix
    archive = t.adjust_names(name)[0]
    obj_suffix = t.adjust_names('.obj')[0]
    output = StringIO.StringIO()
    t.write('jamroot.jam', '')
    output.write('static-lib %s :\n' % name.split('.')[0])
    for s in sorted(sources):
        output.write('    %s\n' % s)
    output.write('    ;\n')
    t.write('lib/jamfile.jam', output.getvalue())
    create_sources('lib', sources)
    t.run_build_system(subdir='lib')
    built_archive = 'lib/bin/$toolset/debug*/%s' % name
    t.expect_addition(built_archive)
    t.copy(built_archive, name)
    t.rm('lib')

def test_glob_archive(archives, glob, expected, sort_results=False):
    if False:
        i = 10
        return i + 15
    output = StringIO.StringIO()
    glob = glob.replace('$archive1', archives[0]).replace('$obj', obj_suffix)
    expected = [m.replace('$archive1', archives[0]).replace('$obj', obj_suffix) for m in expected]
    if len(archives) > 1:
        glob = glob.replace('$archive2', archives[1]).replace('$obj', obj_suffix)
        expected = [m.replace('$archive2', archives[1]).replace('$obj', obj_suffix) for m in expected]
    if sort_results:
        glob = '[ SORT %s ]' % glob
    output.write('    for local p in %s\n    {\n        ECHO $(p) ;\n    }\n    UPDATE ;\n    ' % glob)
    t.write('file.jam', output.getvalue())
    if sort_results:
        expected.sort()
    t.run_build_system(['-ffile.jam'], stdout='\n'.join(expected + ['']))
    t.rm('file.jam')
setup_archive('auxilliary1.lib', sources)
archive1 = archive
setup_archive('auxilliary2.lib', sources)
archive2 = archive
test_glob_archive([archive1], '[ GLOB_ARCHIVE ]', [])
test_glob_archive([archive1], '[ GLOB_ARCHIVE $archive1 : ]', [])
test_glob_archive([archive1], '[ GLOB_ARCHIVE $archive1 : a ]', [])
test_glob_archive([archive1], '[ GLOB_ARCHIVE $archive1 : a$obj ]', ['$archive1(a$obj)'])
test_glob_archive([archive1], '[ GLOB_ARCHIVE $archive1 : b.* ]', ['$archive1(b$obj)'])
test_glob_archive([archive1], '[ GLOB_ARCHIVE $archive1 : "\\b?match[\\.]*" ]', ['$archive1(b_match$obj)'])
test_glob_archive([archive1], '[ GLOB_ARCHIVE $archive1 : b* ]', ['$archive1(b$obj)', '$archive1(b_match$obj)'])
test_glob_archive([archive1], '[ GLOB_ARCHIVE $archive1 : b.* b_* ]', ['$archive1(b$obj)', '$archive1(b_match$obj)'])
test_glob_archive([archive1, archive2], '[ GLOB_ARCHIVE $archive1 $archive2 : b.* b_* ]', ['$archive1(b$obj)', '$archive1(b_match$obj)', '$archive2(b$obj)', '$archive2(b_match$obj)'])
test_glob_archive([archive1, archive1], '[ GLOB_ARCHIVE $archive1 $archive2 $archive1 : b.* ]', ['$archive1(b$obj)', '$archive2(b$obj)', '$archive1(b$obj)'])
test_glob_archive([archive1], '[ GLOB_ARCHIVE $archive1 : nopath_check$obj ]', ['$archive1(nopath_check$obj)'])
case_sensitive_members = not vms
if case_sensitive_members:
    test_glob_archive([archive1], '[ GLOB_ARCHIVE $archive1 : casecheck$obj : true ]', ['$archive1(CaseCheck$obj)'])
elif vms:
    test_glob_archive([archive1], '[ GLOB_ARCHIVE $archive1 : CaseCheck$obj : false ]', ['$archive1(casecheck$obj)'])
test_glob_archive([archive1], '[ GLOB_ARCHIVE $archive1 : seq_check*$obj ]', ['$archive1(seq_check1$obj)', '$archive1(seq_check2$obj)', '$archive1(seq_check3$obj)'])
symbol_glob_supported = vms
if symbol_glob_supported:
    test_glob_archive([archive1], '[ GLOB_ARCHIVE $archive1 : : : symbol ]', ['$archive1(symbols_check$obj)'])
    test_glob_archive([archive1], '[ GLOB_ARCHIVE $archive1 : : : symbol_* ]', ['$archive1(symbols_check$obj)'])
    test_glob_archive([archive1], '[ GLOB_ARCHIVE $archive1 : *symbol* : : *member* ]', ['$archive1(members_and_symbols_check$obj)'])
    test_glob_archive([archive1], '[ GLOB_ARCHIVE $archive1 : : true : symbolcasecheck ]', ['$archive1(symbol_case_check$obj)'])
    test_glob_archive([archive1], '[ GLOB_ARCHIVE $archive1 : : : main _main ]', ['$archive1(main_check$obj)'])
else:
    test_glob_archive([archive1], '[ GLOB_ARCHIVE $archive1 : : : symbol ]', [])
t.cleanup()