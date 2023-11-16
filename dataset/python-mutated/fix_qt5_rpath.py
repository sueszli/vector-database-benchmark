"""Methods to fix Mac OS X @rpath and /usr/local dependencies, and analyze the
   app bundle for OpenShot to find all external dependencies"""
import os
import subprocess
from subprocess import call
import re
import shutil
non_executables = ['.py', '.svg', '.png', '.blend', '.a', '.pak', '.qm', '.pyc', '.txt', '.jpg', '.zip', '.dat', '.conf', '.xml', '.h', '.ui', '.json', '.exe']

def fix_rpath(PATH):
    if False:
        i = 10
        return i + 15
    'FIX broken @rpath on Qt, PyQt, and /usr/local/ dependencies with no @rpath'
    duplicate_path = os.path.join(PATH, 'lib', 'openshot_qt')
    if os.path.exists(duplicate_path):
        shutil.rmtree(duplicate_path)
    for (root, dirs, files) in os.walk(PATH):
        for basename in files:
            file_path = os.path.join(root, basename)
            relative_path = os.path.relpath(root, PATH)
            if os.path.splitext(file_path)[-1] in non_executables or basename.startswith('.') or 'profiles' in file_path:
                continue
            executable_path = os.path.join('@executable_path', relative_path, basename)
            if relative_path == '.':
                executable_path = os.path.join('@executable_path', basename)
            call(['install_name_tool', file_path, '-id', executable_path])
            raw_output = subprocess.Popen(['oTool', '-L', file_path], stdout=subprocess.PIPE).communicate()[0].decode('utf-8')
            for output in raw_output.split('\n')[1:-1]:
                if output and 'is not an object file' not in output and ('.o):' not in output):
                    dependency_path = output.split('\t')[1].split(' ')[0]
                    (dependency_base_path, dependency_name) = os.path.split(dependency_path)
                    if '/usr/local' in dependency_path or '@rpath' in dependency_path:
                        dependency_exe_path = os.path.join('@executable_path', dependency_name)
                        if not os.path.exists(os.path.join(PATH, dependency_name)):
                            print('ERROR: /usr/local PATH not found in EXE folder: %s' % dependency_path)
                        else:
                            call(['install_name_tool', file_path, '-change', dependency_path, dependency_exe_path])

def print_min_versions(PATH):
    if False:
        return 10
    'Print ALL MINIMUM and SDK VERSIONS for files in OpenShot build folder.\n    This does not list all dependent libraries though, and sometimes one of those can cause issues.'
    REGEX_SDK_MATCH = re.compile('.*(LC_VERSION_MIN_MACOSX).*version (\\d+\\.\\d+).*sdk (\\d+\\.\\d+).*(cmd)', re.DOTALL)
    REGEX_SDK_MATCH2 = re.compile('.*sdk\\s(.*)\\s*minos\\s(.*)')
    VERSIONS = {}
    for (root, dirs, files) in os.walk(PATH):
        for basename in files:
            file_path = os.path.join(root, basename)
            file_parts = file_path.split('/')
            if os.path.splitext(file_path)[-1] in non_executables or basename.startswith('.'):
                continue
            raw_output = subprocess.Popen(['oTool', '-l', file_path], stdout=subprocess.PIPE).communicate()[0].decode('utf-8')
            matches = REGEX_SDK_MATCH.findall(raw_output)
            matches2 = REGEX_SDK_MATCH2.findall(raw_output)
            min_version = None
            sdk_version = None
            if matches and len(matches[0]) == 4:
                min_version = matches[0][1]
                sdk_version = matches[0][2]
            elif matches2 and len(matches2[0]) == 2:
                sdk_version = matches2[0][0]
                min_version = matches2[0][1]
            if min_version and sdk_version:
                print('... scanning %s for min version (min: %s, sdk: %s)' % (file_path.replace(PATH, ''), min_version, sdk_version))
                if min_version in VERSIONS:
                    if file_path not in VERSIONS[min_version]:
                        VERSIONS[min_version].append(file_path)
                else:
                    VERSIONS[min_version] = [file_path]
                if min_version in ['11.0']:
                    print('ERROR!!!! Minimum OS X version not met for %s' % file_path)
    print('\nSummary of Minimum Mac SDKs for Dependencies:')
    for key in sorted(VERSIONS.keys()):
        print('\n%s' % key)
        for file_path in VERSIONS[key]:
            print('  %s' % file_path)
    print('\nCount of Minimum Mac SDKs for Dependencies:')
    for key in sorted(VERSIONS.keys()):
        print('%s (%d)' % (key, len(VERSIONS[key])))
if __name__ == '__main__':
    PATH = '/Users/jonathanthomas/apps/openshot-qt/build/exe.macosx-10.15-x86_64-3.7'
    fix_rpath(PATH)
    print_min_versions(PATH)