import BoostBuild
import os

def basic():
    if False:
        return 10
    t = BoostBuild.Tester(pass_toolset=0)
    t.write('file.jam', 'actions do-print\n{\n    echo updating $(<)\n}\n\nNOTFILE target1 ;\nALWAYS target1 ;\ndo-print target1 ;\n\nUPDATE_NOW target1 ;\n\nDEPENDS all : target1 ;\n')
    t.run_build_system(['-ffile.jam'], stdout='...found 1 target...\n...updating 1 target...\ndo-print target1\nupdating target1\n...updated 1 target...\n...found 1 target...\n')
    t.cleanup()

def ignore_minus_n():
    if False:
        while True:
            i = 10
    t = BoostBuild.Tester(pass_toolset=0)
    t.write('file.jam', 'actions do-print\n{\n    echo updating $(<)\n}\n\nNOTFILE target1 ;\nALWAYS target1 ;\ndo-print target1 ;\n\nUPDATE_NOW target1 : : ignore-minus-n ;\n\nDEPENDS all : target1 ;\n')
    t.run_build_system(['-ffile.jam', '-n'], stdout='...found 1 target...\n...updating 1 target...\ndo-print target1\n\n    echo updating target1\n\nupdating target1\n...updated 1 target...\n...found 1 target...\n')
    t.cleanup()

def failed_target():
    if False:
        print('Hello World!')
    t = BoostBuild.Tester(pass_toolset=0)
    t.write('file.jam', 'actions fail\n{\n    exit 1\n}\n\nNOTFILE target1 ;\nALWAYS target1 ;\nfail target1 ;\n\nactions do-print\n{\n    echo updating $(<)\n}\n\nNOTFILE target2 ;\ndo-print target2 ;\nDEPENDS target2 : target1 ;\n\nUPDATE_NOW target1 : : ignore-minus-n ;\n\nDEPENDS all : target1 target2 ;\n')
    t.run_build_system(['-ffile.jam', '-n'], stdout='...found 1 target...\n...updating 1 target...\nfail target1\n\n    exit 1\n\n...failed fail target1...\n...failed updating 1 target...\n...found 2 targets...\n...updating 1 target...\ndo-print target2\n\n    echo updating target2\n\n...updated 1 target...\n')
    t.cleanup()

def missing_target():
    if False:
        while True:
            i = 10
    t = BoostBuild.Tester(pass_toolset=0)
    t.write('file.jam', 'actions do-print\n{\n    echo updating $(<)\n}\n\nNOTFILE target2 ;\ndo-print target2 ;\nDEPENDS target2 : target1 ;\n\nUPDATE_NOW target1 : : ignore-minus-n ;\n\nDEPENDS all : target1 target2 ;\n')
    t.run_build_system(['-ffile.jam', '-n'], status=1, stdout="don't know how to make target1\n...found 1 target...\n...can't find 1 target...\n...found 2 targets...\n...can't make 1 target...\n")
    t.cleanup()

def build_once():
    if False:
        while True:
            i = 10
    '\n      Make sure that if we call UPDATE_NOW with ignore-minus-n, the target gets\n    updated exactly once regardless of previous calls to UPDATE_NOW with -n in\n    effect.\n\n    '
    t = BoostBuild.Tester(pass_toolset=0)
    t.write('file.jam', 'actions do-print\n{\n    echo updating $(<)\n}\n\nNOTFILE target1 ;\nALWAYS target1 ;\ndo-print target1 ;\n\nUPDATE_NOW target1 ;\nUPDATE_NOW target1 : : ignore-minus-n ;\nUPDATE_NOW target1 : : ignore-minus-n ;\n\nDEPENDS all : target1 ;\n')
    t.run_build_system(['-ffile.jam', '-n'], stdout='...found 1 target...\n...updating 1 target...\ndo-print target1\n\n    echo updating target1\n\n...updated 1 target...\ndo-print target1\n\n    echo updating target1\n\nupdating target1\n...updated 1 target...\n...found 1 target...\n')
    t.cleanup()

def return_status():
    if False:
        print('Hello World!')
    '\n    Make sure that UPDATE_NOW returns a failure status if\n    the target failed in a previous call to UPDATE_NOW\n    '
    t = BoostBuild.Tester(pass_toolset=0)
    t.write('file.jam', 'actions fail\n{\n    exit 1\n}\n\nNOTFILE target1 ;\nALWAYS target1 ;\nfail target1 ;\n\nECHO "update1:" [ UPDATE_NOW target1 ] ;\nECHO "update2:" [ UPDATE_NOW target1 ] ;\n\nDEPENDS all : target1 ;\n')
    t.run_build_system(['-ffile.jam'], status=1, stdout='...found 1 target...\n...updating 1 target...\nfail target1\n\n    exit 1\n\n...failed fail target1...\n...failed updating 1 target...\nupdate1:\nupdate2:\n...found 1 target...\n')
    t.cleanup()

def save_restore():
    if False:
        while True:
            i = 10
    'Tests that ignore-minus-n and ignore-minus-q are\n    local to the call to UPDATE_NOW'
    t = BoostBuild.Tester(pass_toolset=0)
    t.write('actions.jam', 'rule fail\n{\n    NOTFILE $(<) ;\n    ALWAYS $(<) ;\n}\nactions fail\n{\n    exit 1\n}\n\nrule pass\n{\n    NOTFILE $(<) ;\n    ALWAYS $(<) ;\n}\nactions pass\n{\n    echo updating $(<)\n}\n')
    t.write('file.jam', '\ninclude actions.jam ;\nfail target1 ;\nfail target2 ;\nUPDATE_NOW target1 target2 : : $(IGNORE_MINUS_N) : $(IGNORE_MINUS_Q) ;\nfail target3 ;\nfail target4 ;\nUPDATE_NOW target3 target4 ;\nUPDATE ;\n')
    t.run_build_system(['-n', '-sIGNORE_MINUS_N=1', '-ffile.jam'], stdout='...found 2 targets...\n...updating 2 targets...\nfail target1\n\n    exit 1\n\n...failed fail target1...\nfail target2\n\n    exit 1\n\n...failed fail target2...\n...failed updating 2 targets...\n...found 2 targets...\n...updating 2 targets...\nfail target3\n\n    exit 1\n\nfail target4\n\n    exit 1\n\n...updated 2 targets...\n')
    t.run_build_system(['-q', '-sIGNORE_MINUS_N=1', '-ffile.jam'], status=1, stdout='...found 2 targets...\n...updating 2 targets...\nfail target1\n\n    exit 1\n\n...failed fail target1...\n...failed updating 1 target...\n...found 2 targets...\n...updating 2 targets...\nfail target3\n\n    exit 1\n\n...failed fail target3...\n...failed updating 1 target...\n')
    t.run_build_system(['-n', '-sIGNORE_MINUS_Q=1', '-ffile.jam'], stdout='...found 2 targets...\n...updating 2 targets...\nfail target1\n\n    exit 1\n\nfail target2\n\n    exit 1\n\n...updated 2 targets...\n...found 2 targets...\n...updating 2 targets...\nfail target3\n\n    exit 1\n\nfail target4\n\n    exit 1\n\n...updated 2 targets...\n')
    t.run_build_system(['-q', '-sIGNORE_MINUS_Q=1', '-ffile.jam'], status=1, stdout='...found 2 targets...\n...updating 2 targets...\nfail target1\n\n    exit 1\n\n...failed fail target1...\nfail target2\n\n    exit 1\n\n...failed fail target2...\n...failed updating 2 targets...\n...found 2 targets...\n...updating 2 targets...\nfail target3\n\n    exit 1\n\n...failed fail target3...\n...failed updating 1 target...\n')
    t.cleanup()
basic()
ignore_minus_n()
failed_target()
missing_target()
build_once()
return_status()
save_restore()