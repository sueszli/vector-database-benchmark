import BoostBuild
import TestCmd
import re

def split_stdin_stdout(text):
    if False:
        i = 10
        return i + 15
    'stdin is all text after the prompt up to and including\n    the next newline.  Everything else is stdout.  stdout\n    may contain regular expressions enclosed in {{}}.'
    prompt = re.escape('(gdb) \n')
    pattern = re.compile('(?<=%s)((?:\\d*-.*)\n)' % prompt)
    stdin = ''.join(re.findall(pattern, text))
    stdout = re.sub(pattern, '', text)
    outside_pattern = re.compile('(?:\\A|(?<=\\}\\}))(?:[^\\{]|(?:\\{(?!\\{)))*(?:(?=\\{\\{)|\\Z)')

    def escape_line(line):
        if False:
            while True:
                i = 10
        line = re.sub(outside_pattern, lambda m: re.escape(m.group(0)), line)
        return re.sub('\\{\\{|\\}\\}', '', line)
    stdout = '\n'.join([escape_line(line) for line in stdout.split('\n')])
    return (stdin, stdout)

def run(tester, io):
    if False:
        i = 10
        return i + 15
    (input, output) = split_stdin_stdout(io)
    tester.run_build_system(stdin=input, stdout=output, match=TestCmd.match_re)

def make_tester():
    if False:
        return 10
    return BoostBuild.Tester(['-dmi'], pass_toolset=False, pass_d0=False, use_test_config=False, ignore_toolset_requirements=False, match=TestCmd.match_re)

def test_exec_run():
    if False:
        while True:
            i = 10
    t = make_tester()
    t.write('test.jam', '        UPDATE ;\n    ')
    run(t, '=thread-group-added,id="i1"\n(gdb)\n72-exec-run -ftest.jam\n=thread-created,id="1",group-id="i1"\n72^running\n(gdb)\n*stopped,reason="exited-normally"\n(gdb)\n73-gdb-exit\n73^exit\n')
    t.cleanup()

def test_exit_status():
    if False:
        for i in range(10):
            print('nop')
    t = make_tester()
    t.write('test.jam', '        EXIT : 1 ;\n    ')
    run(t, '=thread-group-added,id="i1"\n(gdb)\n72-exec-run -ftest.jam\n=thread-created,id="1",group-id="i1"\n72^running\n(gdb)\n\n*stopped,reason="exited",exit-code="1"\n(gdb)\n73-gdb-exit\n73^exit\n')
    t.cleanup()

def test_exec_step():
    if False:
        while True:
            i = 10
    t = make_tester()
    t.write('test.jam', '        rule g ( )\n        {\n            a = 1 ;\n            b = 2 ;\n        }\n        rule f ( )\n        {\n            g ;\n            c = 3 ;\n        }\n        f ;\n    ')
    run(t, '=thread-group-added,id="i1"\n(gdb)\n-break-insert f\n^done,bkpt={number="1",type="breakpoint",disp="keep",enabled="y",func="f"}\n(gdb)\n72-exec-run -ftest.jam\n=thread-created,id="1",group-id="i1"\n72^running\n(gdb)\n*stopped,reason="breakpoint-hit",bkptno="1",disp="keep",frame={func="f",args=[],file="test.jam",fullname="{{.*}}test.jam",line="8"},thread-id="1",stopped-threads="all"\n(gdb)\n1-exec-step\n1^running\n(gdb)\n*stopped,reason="end-stepping-range",frame={func="g",args=[],file="test.jam",fullname="{{.*}}test.jam",line="3"},thread-id="1"\n(gdb)\n2-exec-step\n2^running\n(gdb)\n*stopped,reason="end-stepping-range",frame={func="g",args=[],file="test.jam",fullname="{{.*}}test.jam",line="4"},thread-id="1"\n(gdb)\n3-exec-step\n3^running\n(gdb)\n*stopped,reason="end-stepping-range",frame={func="f",args=[],file="test.jam",fullname="{{.*}}test.jam",line="9"},thread-id="1"\n(gdb)\n73-gdb-exit\n73^exit\n')
    t.cleanup()

def test_exec_next():
    if False:
        for i in range(10):
            print('nop')
    t = make_tester()
    t.write('test.jam', '        rule g ( )\n        {\n            a = 1 ;\n        }\n        rule f ( )\n        {\n            g ;\n            b = 2 ;\n            c = 3 ;\n        }\n        rule h ( )\n        {\n            f ;\n            g ;\n        }\n        h ;\n        d = 4 ;\n    ')
    run(t, '=thread-group-added,id="i1"\n(gdb)\n-break-insert f\n^done,bkpt={number="1",type="breakpoint",disp="keep",enabled="y",func="f"}\n(gdb)\n72-exec-run -ftest.jam\n=thread-created,id="1",group-id="i1"\n72^running\n(gdb)\n*stopped,reason="breakpoint-hit",bkptno="1",disp="keep",frame={func="f",args=[],file="test.jam",fullname="{{.*}}test.jam",line="7"},thread-id="1",stopped-threads="all"\n(gdb)\n1-exec-next\n1^running\n(gdb)\n*stopped,reason="end-stepping-range",frame={func="f",args=[],file="test.jam",fullname="{{.*}}test.jam",line="8"},thread-id="1"\n(gdb)\n2-exec-next\n2^running\n(gdb)\n*stopped,reason="end-stepping-range",frame={func="f",args=[],file="test.jam",fullname="{{.*}}test.jam",line="9"},thread-id="1"\n(gdb)\n3-exec-next\n3^running\n(gdb)\n*stopped,reason="end-stepping-range",frame={func="h",args=[],file="test.jam",fullname="{{.*}}test.jam",line="14"},thread-id="1"\n(gdb)\n4-exec-next\n4^running\n(gdb)\n*stopped,reason="end-stepping-range",frame={func="module scope",args=[],file="test.jam",fullname="{{.*}}test.jam",line="17"},thread-id="1"\n(gdb)\n73-gdb-exit\n73^exit\n')
    t.cleanup()

def test_exec_finish():
    if False:
        return 10
    t = make_tester()
    t.write('test.jam', '        rule f ( )\n        {\n            a = 1 ;\n        }\n        rule g ( )\n        {\n            f ;\n            b = 2 ;\n            i ;\n        }\n        rule h ( )\n        {\n            g ;\n            i ;\n        }\n        rule i ( )\n        {\n            c = 3 ;\n        }\n        h ;\n        d = 4 ;\n    ')
    run(t, '=thread-group-added,id="i1"\n(gdb)\n-break-insert f\n^done,bkpt={number="1",type="breakpoint",disp="keep",enabled="y",func="f"}\n(gdb)\n72-exec-run -ftest.jam\n=thread-created,id="1",group-id="i1"\n72^running\n(gdb)\n*stopped,reason="breakpoint-hit",bkptno="1",disp="keep",frame={func="f",args=[],file="test.jam",fullname="{{.*}}test.jam",line="3"},thread-id="1",stopped-threads="all"\n(gdb)\n1-exec-finish\n1^running\n(gdb)\n*stopped,reason="end-stepping-range",frame={func="g",args=[],file="test.jam",fullname="{{.*}}test.jam",line="8"},thread-id="1"\n(gdb)\n2-exec-finish\n2^running\n(gdb)\n*stopped,reason="end-stepping-range",frame={func="h",args=[],file="test.jam",fullname="{{.*}}test.jam",line="14"},thread-id="1"\n(gdb)\n3-exec-finish\n3^running\n(gdb)\n*stopped,reason="end-stepping-range",frame={func="module scope",args=[],file="test.jam",fullname="{{.*}}test.jam",line="21"},thread-id="1"\n(gdb)\n73-gdb-exit\n73^exit\n')
    t.cleanup()

def test_breakpoints():
    if False:
        i = 10
        return i + 15
    'Tests the interaction between the following commands:\n    break, clear, delete, disable, enable'
    t = make_tester()
    t.write('test.jam', '        rule f ( )\n        {\n            a = 1 ;\n        }\n        rule g ( )\n        {\n            b = 2 ;\n        }\n        rule h ( )\n        {\n            c = 3 ;\n            d = 4 ;\n        }\n        f ;\n        g ;\n        h ;\n        UPDATE ;\n    ')
    run(t, '=thread-group-added,id="i1"\n(gdb)\n-break-insert f\n^done,bkpt={number="1",type="breakpoint",disp="keep",enabled="y",func="f"}\n(gdb)\n72-exec-run -ftest.jam\n=thread-created,id="1",group-id="i1"\n72^running\n(gdb)\n*stopped,reason="breakpoint-hit",bkptno="1",disp="keep",frame={func="f",args=[],file="test.jam",fullname="{{.*}}test.jam",line="3"},thread-id="1",stopped-threads="all"\n(gdb)\n-interpreter-exec console kill\n^done\n(gdb)\n-break-insert g\n^done,bkpt={number="2",type="breakpoint",disp="keep",enabled="y",func="g"}\n(gdb)\n-break-disable 1\n^done\n(gdb)\n73-exec-run -ftest.jam\n=thread-created,id="1",group-id="i1"\n73^running\n(gdb)\n*stopped,reason="breakpoint-hit",bkptno="2",disp="keep",frame={func="g",args=[],file="test.jam",fullname="{{.*}}test.jam",line="7"},thread-id="1",stopped-threads="all"\n(gdb)\n-interpreter-exec console kill\n^done\n(gdb)\n-break-enable 1\n^done\n(gdb)\n74-exec-run -ftest.jam\n=thread-created,id="1",group-id="i1"\n74^running\n(gdb)\n*stopped,reason="breakpoint-hit",bkptno="1",disp="keep",frame={func="f",args=[],file="test.jam",fullname="{{.*}}test.jam",line="3"},thread-id="1",stopped-threads="all"\n(gdb)\n-interpreter-exec console kill\n^done\n(gdb)\n-break-delete 1\n^done\n(gdb)\n75-exec-run -ftest.jam\n=thread-created,id="1",group-id="i1"\n75^running\n(gdb)\n*stopped,reason="breakpoint-hit",bkptno="2",disp="keep",frame={func="g",args=[],file="test.jam",fullname="{{.*}}test.jam",line="7"},thread-id="1",stopped-threads="all"\n(gdb)\n76-gdb-exit\n76^exit\n')
    t.cleanup()
test_exec_run()
test_exit_status()
test_exec_step()
test_exec_next()
test_exec_finish()
test_breakpoints()