import BoostBuild

def test_basic():
    if False:
        i = 10
        return i + 15
    t = BoostBuild.Tester(use_test_config=False)
    t.write('jamroot.jam', 'lib a : a.cpp : <include>. ;')
    t.write('a.cpp', '#include <a.h>\nvoid\n# ifdef _WIN32\n__declspec(dllexport)\n# endif\nfoo() {}\n')
    t.write('a.h', '//empty file\n')
    t.write('d/jamfile.jam', 'exe b : b.cpp ..//a ;')
    t.write('d/b.cpp', 'void foo();\nint main() { foo(); }\n')
    t.run_build_system(subdir='d')
    t.write('jamroot.jam', 'lib a : a.cpp : <variant>debug:<include>. ;')
    t.rm('bin')
    t.run_build_system(subdir='d')
    t.write('jamroot.jam', 'lib a : a.cpp : <include>. : : <variant>debug:<include>. ;\n')
    t.write('d/b.cpp', '#include <a.h>\nvoid foo();\nint main() { foo(); }\n')
    t.rm('d/bin')
    t.run_build_system(subdir='d')
    t.cleanup()

def test_absolute_paths():
    if False:
        return 10
    '\n      Test that absolute paths inside requirements are ok. The problems\n    appeared only when building targets in subprojects.\n\n    '
    t = BoostBuild.Tester(use_test_config=False)
    t.write('jamroot.jam', 'build-project x ;')
    t.write('x/jamfile.jam', 'local pwd = [ PWD ] ;\nproject : requirements <include>$(pwd)/x/include ;\nexe m : m.cpp : <include>$(pwd)/x/include2 ;\n')
    t.write('x/m.cpp', '#include <h1.hpp>\n#include <h2.hpp>\nint main() {}\n')
    t.write('x/include/h1.hpp', '\n')
    t.write('x/include2/h2.hpp', '\n')
    t.run_build_system()
    t.expect_addition('x/bin/$toolset/debug*/m.exe')
    t.cleanup()

def test_ordered_paths():
    if False:
        return 10
    'Test that "&&" in path features is handled correctly.'
    t = BoostBuild.Tester(use_test_config=False)
    t.write('jamroot.jam', 'build-project sub ;')
    t.write('sub/jamfile.jam', 'exe a : a.cpp : <include>../h1&&../h2 ;')
    t.write('sub/a.cpp', '#include <header.h>\nint main() { return OK; }\n')
    t.write('h2/header.h', 'int const OK = 0;\n')
    t.run_build_system()
    t.expect_addition('sub/bin/$toolset/debug*/a.exe')
    t.cleanup()

def test_paths_set_by_indirect_conditionals():
    if False:
        for i in range(10):
            print('nop')
    t = BoostBuild.Tester(use_test_config=False)
    header = 'child_dir/folder_to_include/some_header.h'
    t.write('jamroot.jam', '\nbuild-project child_dir ;\nrule attach-include-parent ( properties * )\n{\n    return <include>another_folder ;\n}\n# requirements inherited from a parent project will bind paths\n# relative to the project that actually names the rule.\nproject : requirements <conditional>@attach-include-parent ;\n')
    t.write('child_dir/jamfile.jam', 'import remote/remote ;\n\n# If we set the <include>folder_to_include property directly, it will work\nobj x1 : x.cpp : <conditional>@attach-include-local ;\nobj x2 : x.cpp : <conditional>@remote.attach-include-remote ;\n\nrule attach-include-local ( properties * )\n{\n    return <include>folder_to_include ;\n}\n')
    t.write('child_dir/remote/remote.jam', 'rule attach-include-remote ( properties * )\n{\n    return <include>folder_to_include ;\n}\n')
    t.write('child_dir/x.cpp', '#include <some_header.h>\n#include <header2.h>\nint main() {}\n')
    t.write(header, 'int some_func();\n')
    t.write('another_folder/header2.h', 'int f2();\n')
    t.write('child_dir/folder_to_include/jamfile.jam', '')
    expected_x1 = 'child_dir/bin/$toolset/debug*/x1.obj'
    expected_x2 = 'child_dir/bin/$toolset/debug*/x2.obj'
    t.run_build_system()
    t.expect_addition(expected_x1)
    t.expect_addition(expected_x2)
    t.touch(header)
    t.run_build_system(subdir='child_dir')
    t.expect_touch(expected_x1)
    t.expect_touch(expected_x2)
    t.touch(header)
    t.run_build_system(['..'], subdir='child_dir/folder_to_include')
    t.expect_touch(expected_x1)
    t.expect_touch(expected_x2)
    t.cleanup()
test_basic()
test_absolute_paths()
test_ordered_paths()
test_paths_set_by_indirect_conditionals()