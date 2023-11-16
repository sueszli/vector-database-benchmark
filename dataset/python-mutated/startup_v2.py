import BoostBuild
import os.path
import re

def check_for_existing_boost_build_jam(t):
    if False:
        i = 10
        return i + 15
    "\n      This test depends on no boost-build.jam file existing in any of the\n    folders along the current folder's path. If it does exist, not only would\n    this test fail but it could point to a completely wrong Boost Build\n    installation, thus causing headaches when attempting to diagnose the\n    problem. That is why we explicitly check for this scenario.\n\n    "
    problem = find_up_to_root(t.workdir, 'boost-build.jam')
    if problem:
        BoostBuild.annotation('misconfiguration', "This test expects to be run from a folder with no 'boost-build.jam' file in any\nof the folders along its path.\n\nWorking folder:\n  '%s'\n\nProblematic boost-build.jam found at:\n  '%s'\n\nPlease remove this file or change the test's working folder and rerun the test.\n" % (t.workdir, problem))
        t.fail_test(1, dump_stdio=False, dump_stack=False)

def find_up_to_root(folder, name):
    if False:
        i = 10
        return i + 15
    last = ''
    while last != folder:
        candidate = os.path.join(folder, name)
        if os.path.exists(candidate):
            return candidate
        last = folder
        folder = os.path.dirname(folder)

def match_re(actual, expected):
    if False:
        i = 10
        return i + 15
    return re.match(expected, actual, re.DOTALL) != None
t = BoostBuild.Tester(match=match_re, boost_build_path='', pass_toolset=0)
t.set_tree('startup')
check_for_existing_boost_build_jam(t)
t.run_build_system(status=1, stdout='Unable to load Boost\\.Build: could not find "boost-build\\.jam"\n.*Attempted search from .* up to the root')
t.run_build_system(status=1, subdir='no-bootstrap1', stdout='Unable to load Boost\\.Build: could not find build system\\..*attempted to load the build system by invoking.*\'boost-build ;\'.*but we were unable to find "bootstrap\\.jam"')
t.run_build_system(status=1, subdir=os.path.join('no-bootstrap1', 'subdir'), stdout='Unable to load Boost\\.Build: could not find build system\\..*attempted to load the build system by invoking.*\'boost-build ;\'.*but we were unable to find "bootstrap\\.jam"')
t.run_build_system(status=1, subdir='no-bootstrap2', stdout='Unable to load Boost\\.Build: could not find build system\\..*attempted to load the build system by invoking.*\'boost-build \\. ;\'.*but we were unable to find "bootstrap\\.jam"')
t.run_build_system(status=1, subdir='no-bootstrap3', stdout='Unable to load Boost.Build\n.*boost-build\\.jam" was found.*\nHowever, it failed to call the "boost-build" rule')
t.run_build_system(['-sBOOST_BUILD_PATH=../boost-root/build'], subdir='bootstrap-env', stdout='build system bootstrapped')
t.run_build_system(subdir='bootstrap-explicit', stdout='build system bootstrapped')
t.cleanup()