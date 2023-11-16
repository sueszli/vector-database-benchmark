import BoostBuild
import string
t = BoostBuild.Tester(pass_toolset=0)
t.write('core-dependency-helpers', '\nrule hdrrule\n{\n   INCLUDES $(1) : $(2) ;\n}\nactions copy\n{\n   cp $(>) $(<)\n}\n')
code = 'include core-dependency-helpers ;\nDEPENDS all : a ;\nDEPENDS a : b ;\n\nactions create-b\n{\n   echo \'#include <foo.h>\' > $(<)\n}\ncopy a : b ;\ncreate-b b ;\nHDRRULE on b foo.h bar.h = hdrrule ;\nHDRSCAN on b foo.h bar.h = "#include <(.*)>" ;\n'
t.run_build_system('-f-', stdin=code)
t.fail_test(string.find(t.stdout(), '...skipped a for lack of foo.h...') == -1)
t.rm('b')
t.run_build_system('-f-', stdin=code + ' copy c : b ; DEPENDS c : b ; DEPENDS all : c ; ')
t.fail_test(string.find(t.stdout(), '...skipped c for lack of foo.h...') == -1)
t.rm('b')
code += '\nactions create-foo\n{\n    echo // > $(<)\n}\ncreate-foo foo.h ;\n'
t.run_build_system('-f-', stdin=code)

def mk_correct_order_func(s1, s2):
    if False:
        i = 10
        return i + 15

    def correct_order(s):
        if False:
            i = 10
            return i + 15
        n1 = string.find(s, s1)
        n2 = string.find(s, s2)
        return n1 != -1 and n2 != -1 and (n1 < n2)
    return correct_order
correct_order = mk_correct_order_func('create-foo', 'copy a')
t.rm(['a', 'b', 'foo.h'])
t.run_build_system('-d+2 -f-', stdin=code + ' DEPENDS all : foo.h ;')
t.fail_test(not correct_order(t.stdout()))
t.rm(['a', 'b', 'foo.h'])
t.run_build_system('-d+2 -f-', stdin=' DEPENDS all : foo.h ; ' + code)
t.fail_test(not correct_order(t.stdout()))
t.rm(['a', 'b'])
t.write('foo.h', '#include <bar.h>')
t.write('bar.h', '#include <biz.h>')
t.run_build_system('-d+2 -f-', stdin=code)
t.fail_test(string.find(t.stdout(), '...skipped a for lack of biz.h...') == -1)
code += '\nactions create-biz\n{\n   echo // > $(<)\n}\ncreate-biz biz.h ;\n'
t.rm(['b'])
correct_order = mk_correct_order_func('create-biz', 'copy a')
t.run_build_system('-d+2 -f-', stdin=code + ' DEPENDS all : biz.h ;')
t.fail_test(not correct_order(t.stdout()))
t.rm(['a', 'biz.h'])
t.run_build_system('-d+2 -f-', stdin=' DEPENDS all : biz.h ; ' + code)
t.fail_test(not correct_order(t.stdout()))
t.write('a', '')
code = '\nDEPENDS all : main d ;\n\nactions copy\n{\n    cp $(>) $(<) ;\n}\n\nDEPENDS main : a ;\ncopy main : a ;\n\nINCLUDES a : <1>c ;\n\nNOCARE <1>c ;\nSEARCH on <1>c = . ;\n\nactions create-c\n{\n    echo d > $(<)\n}\n\nactions create-d\n{\n    echo // > $(<)\n}\n\ncreate-c <2>c ;\nLOCATE on <2>c = . ;\ncreate-d d ;\n\nHDRSCAN on <1>c = (.*) ;\nHDRRULE on <1>c = hdrrule ;\n\nrule hdrrule\n{\n    INCLUDES $(1) : d ;\n}\n'
correct_order = mk_correct_order_func('create-d', 'copy main')
t.run_build_system('-d2 -f-', stdin=code)
t.fail_test(not correct_order(t.stdout()))
t.cleanup()