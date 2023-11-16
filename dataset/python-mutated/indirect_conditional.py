import BoostBuild

def test_basic():
    if False:
        for i in range(10):
            print('nop')
    t = BoostBuild.Tester(use_test_config=False)
    t.write('jamroot.jam', 'exe a1 : a1.cpp : <conditional>@a1-rule ;\nrule a1-rule ( properties * )\n{\n    if <variant>debug in $(properties)\n    {\n        return <define>OK ;\n    }\n}\n\nexe a2 : a2.cpp : <conditional>@$(__name__).a2-rule\n    <variant>debug:<optimization>speed ;\nrule a2-rule ( properties * )\n{\n    if <optimization>speed in $(properties)\n    {\n        return <define>OK ;\n    }\n}\n\nexe a3 : a3.cpp :\n    <conditional>@$(__name__).a3-rule-1\n    <conditional>@$(__name__).a3-rule-2 ;\nrule a3-rule-1 ( properties * )\n{\n    if <optimization>speed in $(properties)\n    {\n        return <define>OK ;\n    }\n}\nrule a3-rule-2 ( properties * )\n{\n    if <variant>debug in $(properties)\n    {\n        return <optimization>speed ;\n    }\n}\n')
    t.write('a1.cpp', '#ifdef OK\nint main() {}\n#endif\n')
    t.write('a2.cpp', '#ifdef OK\nint main() {}\n#endif\n')
    t.write('a3.cpp', '#ifdef OK\nint main() {}\n#endif\n')
    t.run_build_system()
    t.expect_addition('bin/$toolset/debug*/a1.exe')
    t.expect_addition('bin/$toolset/debug/optimization-speed*/a2.exe')
    t.expect_addition('bin/$toolset/debug/optimization-speed*/a3.exe')
    t.cleanup()

def test_glob_in_indirect_conditional():
    if False:
        print('Hello World!')
    "\n      Regression test: project-rules.glob rule run from inside an indirect\n    conditional should report an error as it depends on the 'currently loaded\n    project' concept and indirect conditional rules get called only after all\n    the project modules have already finished loading.\n\n    "
    t = BoostBuild.Tester(use_test_config=False)
    t.write('jamroot.jam', 'use-project /library-example/foo : util/foo ;\nbuild-project app ;\n')
    t.write('app/app.cpp', 'int main() {}\n')
    t.write('app/jamfile.jam', 'exe app : app.cpp /library-example/foo//bar ;')
    t.write('util/foo/bar.cpp', '#ifdef _WIN32\n__declspec(dllexport)\n#endif\nvoid foo() {}\n')
    t.write('util/foo/jamfile.jam', 'rule print-my-sources ( properties * )\n{\n    ECHO My sources: ;\n    ECHO [ glob *.cpp ] ;\n}\nlib bar : bar.cpp : <conditional>@print-my-sources ;\n')
    t.run_build_system(status=1)
    t.expect_output_lines(['My sources:', 'bar.cpp'], False)
    t.expect_output_lines('error: Reference to the project currently being loaded requested when there was no project module being loaded.')
    t.cleanup()
test_basic()
test_glob_in_indirect_conditional()