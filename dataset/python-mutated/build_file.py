import BoostBuild

def test_building_file_from_specific_project():
    if False:
        while True:
            i = 10
    t = BoostBuild.Tester(use_test_config=False)
    t.write('jamroot.jam', 'exe hello : hello.cpp ;\nexe hello2 : hello.cpp ;\nbuild-project sub ;\n')
    t.write('hello.cpp', 'int main() {}\n')
    t.write('sub/jamfile.jam', '\nexe hello : hello.cpp ;\nexe hello2 : hello.cpp ;\nexe sub : hello.cpp ;\n')
    t.write('sub/hello.cpp', 'int main() {}\n')
    t.run_build_system(['sub', t.adjust_suffix('hello.obj')])
    t.expect_output_lines('*depends on itself*', False)
    t.expect_addition('sub/bin/$toolset/debug*/hello.obj')
    t.expect_nothing_more()
    t.cleanup()

def test_building_file_from_specific_target():
    if False:
        i = 10
        return i + 15
    t = BoostBuild.Tester(use_test_config=False)
    t.write('jamroot.jam', 'exe hello1 : hello1.cpp ;\nexe hello2 : hello2.cpp ;\nexe hello3 : hello3.cpp ;\n')
    t.write('hello1.cpp', 'int main() {}\n')
    t.write('hello2.cpp', 'int main() {}\n')
    t.write('hello3.cpp', 'int main() {}\n')
    t.run_build_system(['hello1', t.adjust_suffix('hello1.obj')])
    t.expect_addition('bin/$toolset/debug*/hello1.obj')
    t.expect_nothing_more()
    t.cleanup()

def test_building_missing_file_from_specific_target():
    if False:
        i = 10
        return i + 15
    t = BoostBuild.Tester(use_test_config=False)
    t.write('jamroot.jam', 'exe hello1 : hello1.cpp ;\nexe hello2 : hello2.cpp ;\nexe hello3 : hello3.cpp ;\n')
    t.write('hello1.cpp', 'int main() {}\n')
    t.write('hello2.cpp', 'int main() {}\n')
    t.write('hello3.cpp', 'int main() {}\n')
    obj = t.adjust_suffix('hello2.obj')
    t.run_build_system(['hello1', obj], status=1)
    t.expect_output_lines("don't know how to make*" + obj)
    t.expect_nothing_more()
    t.cleanup()

def test_building_multiple_files_with_different_names():
    if False:
        i = 10
        return i + 15
    t = BoostBuild.Tester(use_test_config=False)
    t.write('jamroot.jam', 'exe hello1 : hello1.cpp ;\nexe hello2 : hello2.cpp ;\nexe hello3 : hello3.cpp ;\n')
    t.write('hello1.cpp', 'int main() {}\n')
    t.write('hello2.cpp', 'int main() {}\n')
    t.write('hello3.cpp', 'int main() {}\n')
    t.run_build_system([t.adjust_suffix('hello1.obj'), t.adjust_suffix('hello2.obj')])
    t.expect_addition('bin/$toolset/debug*/hello1.obj')
    t.expect_addition('bin/$toolset/debug*/hello2.obj')
    t.expect_nothing_more()
    t.cleanup()

def test_building_multiple_files_with_the_same_name():
    if False:
        print('Hello World!')
    t = BoostBuild.Tester(use_test_config=False)
    t.write('jamroot.jam', 'exe hello : hello.cpp ;\nexe hello2 : hello.cpp ;\nbuild-project sub ;\n')
    t.write('hello.cpp', 'int main() {}\n')
    t.write('sub/jamfile.jam', '\nexe hello : hello.cpp ;\nexe hello2 : hello.cpp ;\nexe sub : hello.cpp ;\n')
    t.write('sub/hello.cpp', 'int main() {}\n')
    t.run_build_system([t.adjust_suffix('hello.obj')])
    t.expect_output_lines('*depends on itself*', False)
    t.expect_addition('bin/$toolset/debug*/hello.obj')
    t.expect_addition('sub/bin/$toolset/debug*/hello.obj')
    t.expect_nothing_more()
    t.cleanup()
test_building_file_from_specific_project()
test_building_file_from_specific_target()
test_building_missing_file_from_specific_target()
test_building_multiple_files_with_different_names()
test_building_multiple_files_with_the_same_name()