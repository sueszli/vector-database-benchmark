import BoostBuild
import sys

def test_assigning_project_ids():
    if False:
        print('Hello World!')
    t = BoostBuild.Tester(pass_toolset=False)
    t.write('jamroot.jam', 'import assert ;\nimport modules ;\nimport notfile ;\nimport project ;\n\nrule assert-project-id ( id ? : module-name ? )\n{\n    module-name ?= [ CALLER_MODULE ] ;\n    assert.result $(id) : project.attribute $(module-name) id ;\n}\n\n# Project rule modifies the main project id.\nassert-project-id ;  # Initial project id is empty\nproject foo  ; assert-project-id /foo ;\nproject      ; assert-project-id /foo ;\nproject foo  ; assert-project-id /foo ;\nproject bar  ; assert-project-id /bar ;\nproject /foo ; assert-project-id /foo ;\nproject ""   ; assert-project-id /foo ;\n\n# Calling the use-project rule does not modify the project\'s main id.\nuse-project id1 : a ;\n# We need to load the \'a\' Jamfile module manually as the use-project rule will\n# only schedule the load to be done after the current module load finishes.\na-module = [ project.load a ] ;\nassert-project-id : $(a-module) ;\nuse-project id2 : a ;\nassert-project-id : $(a-module) ;\nmodules.call-in $(a-module) : project baz ;\nassert-project-id /baz : $(a-module) ;\nuse-project id3 : a ;\nassert-project-id /baz : $(a-module) ;\n\n# Make sure the project id still holds after all the scheduled use-project loads\n# complete. We do this by scheduling the assert for the Jam action scheduling\n# phase.\nnotfile x : @assert-a-rule ;\nrule assert-a-rule ( target : : properties * )\n{\n    assert-project-id /baz : $(a-module) ;\n}\n')
    t.write('a/jamfile.jam', '# Initial project id for this module is empty.\nassert-project-id ;\n')
    t.run_build_system()
    t.cleanup()

def test_using_project_ids_in_target_references():
    if False:
        for i in range(10):
            print('nop')
    t = BoostBuild.Tester()
    __write_appender(t, 'appender.jam')
    t.write('jamroot.jam', 'import type ;\ntype.register AAA : _a ;\ntype.register BBB : _b ;\n\nimport appender ;\nappender.register aaa-to-bbb : AAA : BBB ;\n\nuse-project id1 : a ;\nuse-project /id2 : a ;\n\nbbb b1 : /id1//target ;\nbbb b2 : /id2//target ;\nbbb b3 : /id3//target ;\nbbb b4 : a//target ;\nbbb b5 : /project-a1//target ;\nbbb b6 : /project-a2//target ;\nbbb b7 : /project-a3//target ;\n\nuse-project id3 : a ;\n')
    t.write('a/source._a', '')
    t.write('a/jamfile.jam', 'project project-a1 ;\nproject /project-a2 ;\nimport alias ;\nalias target : source._a ;\nproject /project-a3 ;\n')
    t.run_build_system()
    t.expect_addition(('bin/b%d._b' % x for x in range(1, 8)))
    t.expect_nothing_more()
    t.cleanup()

def test_repeated_ids_for_different_projects():
    if False:
        for i in range(10):
            print('nop')
    t = BoostBuild.Tester()
    t.write('a/jamfile.jam', '')
    t.write('jamroot.jam', 'project foo ; use-project foo : a ;')
    t.run_build_system(status=1)
    t.expect_output_lines("error: Attempt to redeclare already registered project id '/foo'.\nerror: Original project:\nerror:     Name: Jamfile<*>\nerror:     Module: Jamfile<*>\nerror:     Main id: /foo\nerror:     File: jamroot.jam\nerror:     Location: .\nerror: New project:\nerror:     Module: Jamfile<*>\nerror:     File: a*jamfile.jam\nerror:     Location: a")
    t.write('jamroot.jam', 'use-project foo : a ; project foo ;')
    t.run_build_system(status=1)
    t.expect_output_lines("error: Attempt to redeclare already registered project id '/foo'.\nerror: Original project:\nerror:     Name: Jamfile<*>\nerror:     Module: Jamfile<*>\nerror:     Main id: /foo\nerror:     File: jamroot.jam\nerror:     Location: .\nerror: New project:\nerror:     Module: Jamfile<*>\nerror:     File: a*jamfile.jam\nerror:     Location: a")
    t.write('jamroot.jam', 'import modules ;\nimport project ;\nmodules.call-in [ project.load a ] : project foo ;\nproject foo ;\n')
    t.run_build_system(status=1)
    t.expect_output_lines("error: at jamroot.jam:4\nerror: Attempt to redeclare already registered project id '/foo'.\nerror: Original project:\nerror:     Name: Jamfile<*>\nerror:     Module: Jamfile<*>\nerror:     Main id: /foo\nerror:     File: a*jamfile.jam\nerror:     Location: a\nerror: New project:\nerror:     Module: Jamfile<*>\nerror:     File: jamroot.jam\nerror:     Location: .")
    t.cleanup()

def test_repeated_ids_for_same_project():
    if False:
        while True:
            i = 10
    t = BoostBuild.Tester()
    t.write('jamroot.jam', 'project foo ; project foo ;')
    t.run_build_system()
    t.write('jamroot.jam', 'project foo ; use-project foo : . ;')
    t.run_build_system()
    t.write('jamroot.jam', 'project foo ; use-project foo : ./. ;')
    t.run_build_system()
    t.write('jamroot.jam', 'project foo ;\nuse-project foo : . ;\nuse-project foo : ./aaa/.. ;\nuse-project foo : ./. ;\n')
    t.run_build_system()
    if sys.platform in ['win32']:
        t.write('a/fOo bAr/b/jamfile.jam', '')
        t.write('jamroot.jam', '\nuse-project bar : "a/foo bar/b" ;\nuse-project bar : "a/foO Bar/b" ;\nuse-project bar : "a/foo BAR/b/" ;\nuse-project bar : "a\\\\.\\\\FOO bar\\\\b\\\\" ;\n')
        t.run_build_system()
        t.rm('a')
    t.write('bar/jamfile.jam', '')
    t.write('jamroot.jam', 'use-project bar : bar ;\nuse-project bar : bar/ ;\nuse-project bar : bar// ;\nuse-project bar : bar/// ;\nuse-project bar : bar//// ;\nuse-project bar : bar/. ;\nuse-project bar : bar/./ ;\nuse-project bar : bar/////./ ;\nuse-project bar : bar/../bar/xxx/.. ;\nuse-project bar : bar/..///bar/xxx///////.. ;\nuse-project bar : bar/./../bar/xxx/.. ;\nuse-project bar : bar/.////../bar/xxx/.. ;\nuse-project bar : bar/././../bar/xxx/.. ;\nuse-project bar : bar/././//////////../bar/xxx/.. ;\nuse-project bar : bar/.///.////../bar/xxx/.. ;\nuse-project bar : bar/./././xxx/.. ;\nuse-project bar : bar/xxx////.. ;\nuse-project bar : bar/xxx/.. ;\nuse-project bar : bar///////xxx/.. ;\n')
    t.run_build_system()
    t.rm('bar')
    if sys.platform in ['win32']:
        t.write('baR/jamfile.jam', '')
        t.write('jamroot.jam', '\nuse-project bar : bar ;\nuse-project bar : BAR ;\nuse-project bar : bAr ;\nuse-project bar : bAr/ ;\nuse-project bar : bAr\\\\ ;\nuse-project bar : bAr\\\\\\\\ ;\nuse-project bar : bAr\\\\\\\\///// ;\nuse-project bar : bAr/. ;\nuse-project bar : bAr/./././ ;\nuse-project bar : bAr\\\\.\\\\.\\\\.\\\\ ;\nuse-project bar : bAr\\\\./\\\\/.\\\\.\\\\ ;\nuse-project bar : bAr/.\\\\././ ;\nuse-project bar : Bar ;\nuse-project bar : BaR ;\nuse-project bar : BaR/./../bAr/xxx/.. ;\nuse-project bar : BaR/./..\\\\bAr\\\\xxx/.. ;\nuse-project bar : BaR/xxx/.. ;\nuse-project bar : BaR///\\\\\\\\\\\\//xxx/.. ;\nuse-project bar : Bar\\\\xxx/.. ;\nuse-project bar : BAR/xXx/.. ;\nuse-project bar : BAR/xXx\\\\\\\\/\\\\/\\\\//\\\\.. ;\n')
        t.run_build_system()
        t.rm('baR')
    t.cleanup()

def test_unresolved_project_references():
    if False:
        return 10
    t = BoostBuild.Tester()
    __write_appender(t, 'appender.jam')
    t.write('a/source._a', '')
    t.write('a/jamfile.jam', 'import alias ; alias target : source._a ;')
    t.write('jamroot.jam', 'import type ;\ntype.register AAA : _a ;\ntype.register BBB : _b ;\n\nimport appender ;\nappender.register aaa-to-bbb : AAA : BBB ;\n\nuse-project foo : a ;\n\nbbb b1 : a//target ;\nbbb b2 : /foo//target ;\nbbb b-invalid : invalid//target ;\nbbb b-root-invalid : /invalid//target ;\nbbb b-missing-root : foo//target ;\nbbb b-invalid-target : /foo//invalid ;\n')
    t.run_build_system(['b1', 'b2'])
    t.expect_addition(('bin/b%d._b' % x for x in range(1, 3)))
    t.expect_nothing_more()
    t.run_build_system(['b-invalid'], status=1)
    t.expect_output_lines("error: Unable to find file or target named\nerror:     'invalid//target'\nerror: referred to from project at\nerror:     '.'\nerror: could not resolve project reference 'invalid'")
    t.run_build_system(['b-root-invalid'], status=1)
    t.expect_output_lines("error: Unable to find file or target named\nerror:     '/invalid//target'\nerror: referred to from project at\nerror:     '.'\nerror: could not resolve project reference '/invalid'")
    t.run_build_system(['b-missing-root'], status=1)
    t.expect_output_lines("error: Unable to find file or target named\nerror:     'foo//target'\nerror: referred to from project at\nerror:     '.'\nerror: could not resolve project reference 'foo' - possibly missing a leading slash ('/') character.")
    t.run_build_system(['b-invalid-target'], status=1)
    t.expect_output_lines("error: Unable to find file or target named\nerror:     '/foo//invalid'\nerror: referred to from project at\nerror:     '.'")
    t.expect_output_lines('*could not resolve project reference*', False)
    t.cleanup()

def __write_appender(t, name):
    if False:
        for i in range(10):
            print('nop')
    t.write(name, '# Copyright 2012 Jurko Gospodnetic\n# Distributed under the Boost Software License, Version 1.0.\n# (See accompanying file LICENSE_1_0.txt or copy at\n# http://www.boost.org/LICENSE_1_0.txt)\n\n#   Support for registering test generators that construct their targets by\n# simply appending their given input data, e.g. list of sources & targets.\n\nimport "class" : new ;\nimport generators ;\nimport modules ;\nimport sequence ;\n\nrule register ( id composing ? : source-types + : target-types + )\n{\n    local caller-module = [ CALLER_MODULE ] ;\n    id = $(caller-module).$(id) ;\n    local g = [ new generator $(id) $(composing) : $(source-types) :\n        $(target-types) ] ;\n    $(g).set-rule-name $(__name__).appender ;\n    generators.register $(g) ;\n    return $(id) ;\n}\n\nif [ modules.peek : NT ]\n{\n    X = ")" ;\n    ECHO_CMD = (echo. ;\n}\nelse\n{\n    X = \\" ;\n    ECHO_CMD = "echo $(X)" ;\n}\n\nlocal appender-runs ;\n\n# We set up separate actions for building each target in order to avoid having\n# to iterate over them in action (i.e. shell) code. We have to be extra careful\n# though to achieve the exact same effect as if doing all the work in just one\n# action. Otherwise Boost Jam might, under some circumstances, run only some of\n# our actions. To achieve this we register a series of actions for all the\n# targets (since they all have the same target list - either all or none of them\n# get run independent of which target actually needs to get built), each\n# building only a single target. Since all our actions use the same targets, we\n# can not use \'on-target\' parameters to pass data to a specific action so we\n# pass them using the second \'sources\' parameter which our actions then know how\n# to interpret correctly. This works well since Boost Jam does not automatically\n# add dependency relations between specified action targets & sources and so the\n# second argument, even though most often used to pass in a list of sources, can\n# actually be used for passing in any type of information.\nrule appender ( targets + : sources + : properties * )\n{\n    appender-runs = [ CALC $(appender-runs:E=0) + 1 ] ;\n    local target-index = 0 ;\n    local target-count = [ sequence.length $(targets) ] ;\n    local original-targets ;\n    for t in $(targets)\n    {\n        target-index = [ CALC $(target-index) + 1 ] ;\n        local appender-run = $(appender-runs) ;\n        if $(targets[2])-defined\n        {\n            appender-run += [$(target-index)/$(target-count)] ;\n        }\n        append $(targets) : $(appender-run:J=" ") $(t) $(sources) ;\n    }\n}\n\nactions append\n{\n    $(ECHO_CMD)-------------------------------------------------$(X)\n    $(ECHO_CMD)Appender run: $(>[1])$(X)\n    $(ECHO_CMD)Appender run: $(>[1])$(X)>> "$(>[2])"\n    $(ECHO_CMD)Target group: $(<:J=\' \')$(X)\n    $(ECHO_CMD)Target group: $(<:J=\' \')$(X)>> "$(>[2])"\n    $(ECHO_CMD)      Target: \'$(>[2])\'$(X)\n    $(ECHO_CMD)      Target: \'$(>[2])\'$(X)>> "$(>[2])"\n    $(ECHO_CMD)     Sources: \'$(>[3-]:J=\' \')\'$(X)\n    $(ECHO_CMD)     Sources: \'$(>[3-]:J=\' \')\'$(X)>> "$(>[2])"\n    $(ECHO_CMD)=================================================$(X)\n    $(ECHO_CMD)-------------------------------------------------$(X)>> "$(>[2])"\n}\n')
test_assigning_project_ids()
test_using_project_ids_in_target_references()
test_repeated_ids_for_same_project()
test_repeated_ids_for_different_projects()
test_unresolved_project_references()