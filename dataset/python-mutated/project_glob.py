import BoostBuild

def test_basic():
    if False:
        while True:
            i = 10
    t = BoostBuild.Tester(use_test_config=False)
    t.write('jamroot.jam', '')
    t.write('d1/a.cpp', 'int main() {}\n')
    t.write('d1/jamfile.jam', 'exe a : [ glob *.cpp ] ../d2/d//l ;')
    t.write('d2/d/l.cpp', '#if defined(_WIN32)\n__declspec(dllexport)\nvoid force_import_lib_creation() {}\n#endif\n')
    t.write('d2/d/jamfile.jam', 'lib l : [ glob *.cpp ] ;')
    t.write('d3/d/jamfile.jam', 'exe a : [ glob ../*.cpp ] ;')
    t.write('d3/a.cpp', 'int main() {}\n')
    t.run_build_system(subdir='d1')
    t.expect_addition('d1/bin/$toolset/debug*/a.exe')
    t.run_build_system(subdir='d3/d')
    t.expect_addition('d3/d/bin/$toolset/debug*/a.exe')
    t.rm('d2/d/bin')
    t.run_build_system(subdir='d2/d')
    t.expect_addition('d2/d/bin/$toolset/debug*/l.dll')
    t.cleanup()

def test_source_location():
    if False:
        while True:
            i = 10
    "\n      Test that when 'source-location' is explicitly-specified glob works\n    relative to the source location.\n\n    "
    t = BoostBuild.Tester(use_test_config=False)
    t.write('jamroot.jam', '')
    t.write('d1/a.cpp', 'very bad non-compilable file\n')
    t.write('d1/src/a.cpp', 'int main() {}\n')
    t.write('d1/jamfile.jam', 'project : source-location src ;\nexe a : [ glob *.cpp ] ../d2/d//l ;\n')
    t.write('d2/d/l.cpp', '#if defined(_WIN32)\n__declspec(dllexport)\nvoid force_import_lib_creation() {}\n#endif\n')
    t.write('d2/d/jamfile.jam', 'lib l : [ glob *.cpp ] ;')
    t.run_build_system(subdir='d1')
    t.expect_addition('d1/bin/$toolset/debug*/a.exe')
    t.cleanup()

def test_wildcards_and_exclusion_patterns():
    if False:
        i = 10
        return i + 15
    '\n        Test that wildcards can include directories. Also test exclusion\n     patterns.\n\n    '
    t = BoostBuild.Tester(use_test_config=False)
    t.write('jamroot.jam', '')
    t.write('d1/src/foo/a.cpp', 'void bar(); int main() { bar(); }\n')
    t.write('d1/src/bar/b.cpp', 'void bar() {}\n')
    t.write('d1/src/bar/bad.cpp', 'very bad non-compilable file\n')
    t.write('d1/jamfile.jam', 'project : source-location src ;\nexe a : [ glob foo/*.cpp bar/*.cpp : bar/bad* ] ../d2/d//l ;\n')
    t.write('d2/d/l.cpp', '#if defined(_WIN32)\n__declspec(dllexport)\nvoid force_import_lib_creation() {}\n#endif\n')
    t.write('d2/d/jamfile.jam', 'lib l : [ glob *.cpp ] ;')
    t.run_build_system(subdir='d1')
    t.expect_addition('d1/bin/$toolset/debug*/a.exe')
    t.cleanup()

def test_glob_tree():
    if False:
        print('Hello World!')
    "Test that 'glob-tree' works."
    t = BoostBuild.Tester(use_test_config=False)
    t.write('jamroot.jam', '')
    t.write('d1/src/foo/a.cpp', 'void bar(); int main() { bar(); }\n')
    t.write('d1/src/bar/b.cpp', 'void bar() {}\n')
    t.write('d1/src/bar/bad.cpp', 'very bad non-compilable file\n')
    t.write('d1/jamfile.jam', 'project : source-location src ;\nexe a : [ glob-tree *.cpp : bad* ] ../d2/d//l ;\n')
    t.write('d2/d/l.cpp', '#if defined(_WIN32)\n__declspec(dllexport)\nvoid force_import_lib_creation() {}\n#endif\n')
    t.write('d2/d/jamfile.jam', 'lib l : [ glob *.cpp ] ;')
    t.run_build_system(subdir='d1')
    t.expect_addition('d1/bin/$toolset/debug*/a.exe')
    t.cleanup()

def test_directory_names_in_glob_tree():
    if False:
        for i in range(10):
            print('nop')
    "Test that directory names in patterns for 'glob-tree' are rejected."
    t = BoostBuild.Tester(use_test_config=False)
    t.write('jamroot.jam', '')
    t.write('d1/src/a.cpp', 'very bad non-compilable file\n')
    t.write('d1/src/foo/a.cpp', 'void bar(); int main() { bar(); }\n')
    t.write('d1/src/bar/b.cpp', 'void bar() {}\n')
    t.write('d1/src/bar/bad.cpp', 'very bad non-compilable file\n')
    t.write('d1/jamfile.jam', 'project : source-location src ;\nexe a : [ glob-tree foo/*.cpp bar/*.cpp : bad* ] ../d2/d//l ;\n')
    t.write('d2/d/l.cpp', '#if defined(_WIN32)\n__declspec(dllexport)\nvoid force_import_lib_creation() {}\n#endif\n')
    t.write('d2/d/jamfile.jam', 'lib l : [ glob *.cpp ] ;')
    t.run_build_system(subdir='d1', status=1)
    t.expect_output_lines('error: The patterns * may not include directory')
    t.cleanup()

def test_glob_with_absolute_names():
    if False:
        for i in range(10):
            print('nop')
    "Test that 'glob' works with absolute names."
    t = BoostBuild.Tester(use_test_config=False)
    t.write('jamroot.jam', '')
    t.write('d1/src/a.cpp', 'very bad non-compilable file\n')
    t.write('d1/src/foo/a.cpp', 'void bar(); int main() { bar(); }\n')
    t.write('d1/src/bar/b.cpp', 'void bar() {}\n')
    t.write('d1/jamfile.jam', 'project : source-location src ;\nlocal pwd = [ PWD ] ;  # Always absolute.\nexe a : [ glob $(pwd)/src/foo/*.cpp $(pwd)/src/bar/*.cpp ] ../d2/d//l ;\n')
    t.write('d2/d/l.cpp', '#if defined(_WIN32)\n__declspec(dllexport)\nvoid force_import_lib_creation() {}\n#endif\n')
    t.write('d2/d/jamfile.jam', 'lib l : [ glob *.cpp ] ;')
    t.run_build_system(subdir='d1')
    t.expect_addition('d1/bin/$toolset/debug*/a.exe')
    t.cleanup()

def test_glob_excludes_in_subdirectory():
    if False:
        print('Hello World!')
    '\n      Regression test: glob excludes used to be broken when building from a\n    subdirectory.\n\n    '
    t = BoostBuild.Tester(use_test_config=False)
    t.write('jamroot.jam', 'build-project p ;')
    t.write('p/p.c', 'int main() {}\n')
    t.write('p/p_x.c', 'very bad non-compilable file\n')
    t.write('p/jamfile.jam', 'exe p : [ glob *.c : p_x.c ] ;')
    t.run_build_system(subdir='p')
    t.expect_addition('p/bin/$toolset/debug*/p.exe')
    t.cleanup()
test_basic()
test_source_location()
test_wildcards_and_exclusion_patterns()
test_glob_tree()
test_directory_names_in_glob_tree()
test_glob_with_absolute_names()
test_glob_excludes_in_subdirectory()