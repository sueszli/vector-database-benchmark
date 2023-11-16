""" Release tools related to Debian packaging.

"""
import os
import sys
from nuitka.utils.Execution import check_call, getNullInput
from nuitka.utils.FileOperations import copyFile, getFileContentByLine, getFileContents, listDir, openTextFile
from .Release import checkNuitkaChangelog

def _callDebchange(*args):
    if False:
        i = 10
        return i + 15
    args = ['debchange'] + list(args)
    os.environ['EDITOR'] = ''
    check_call(args, stdin=getNullInput())

def _discardDebianChangelogLastEntry():
    if False:
        return 10
    changelog_lines = getFileContents('debian/changelog').splitlines()
    with openTextFile('debian/changelog', 'w') as output:
        first = True
        for line in changelog_lines[1:]:
            if line.startswith('nuitka') and first:
                first = False
            if not first:
                output.write(line + '\n')

def updateDebianChangelog(old_version, new_version, distribution):
    if False:
        i = 10
        return i + 15
    debian_version = new_version.replace('rc', '~rc') + '+ds-1'
    os.environ['DEBEMAIL'] = 'Kay Hayen <kay.hayen@gmail.com>'
    if 'rc' in new_version:
        if 'rc' in old_version:
            _discardDebianChangelogLastEntry()
        message = 'New upstream pre-release.'
        if checkNuitkaChangelog() != 'draft':
            changelog = getFileContents('Changelog.rst')
            title = 'Nuitka Release ' + new_version[:-3] + ' (Draft)'
            found = False
            with openTextFile('Changelog.rst', 'w') as changelog_file:
                for line in changelog.splitlines():
                    if not found:
                        if line.startswith('***') and line.endswith('***'):
                            found = True
                            marker = '*' * (len(title) + 2)
                            changelog_file.write(marker + '\n ' + title + '\n' + marker + '\n\n')
                            changelog_file.write('This release is not done yet.\n\n')
                            changelog_file.write(line + '\n')
                            continue
                    changelog_file.write(line + '\n')
            assert found
    elif 'rc' in old_version:
        _discardDebianChangelogLastEntry()
        message = 'New upstream release.'
    else:
        message = 'New upstream hotfix release.'
    _callDebchange('--newversion=%s' % debian_version, message)
    _callDebchange('-r', '--distribution', distribution, '')

def checkChangeLog(message):
    if False:
        while True:
            i = 10
    'Check debian changelog for given message to be present.'
    for line in getFileContentByLine('debian/changelog'):
        if line.startswith(' --'):
            return False
        if message in line:
            return True
    sys.exit("Error, didn't find in debian/changelog: '%s'" % message)

def shallNotIncludeInlineCopy(codename):
    if False:
        return 10
    return codename not in ('stretch', 'jessie', 'xenial')

def cleanupTarfileForDebian(codename, filename, new_name):
    if False:
        for i in range(10):
            print('nop')
    "Remove files that shouldn't be in Debian.\n\n    The inline copies should definitely not be there. Also remove the\n    PDF files.\n    "
    copyFile(filename, new_name)
    check_call(['gunzip', new_name])
    if shallNotIncludeInlineCopy(codename):
        command = ['tar', '--wildcards', '--delete', '--file', new_name[:-3], 'Nuitka*/build/inline_copy']
        check_call(command)
    check_call(['gzip', '-9', '-n', new_name[:-3]])

def runPy2dsc(filename, new_name):
    if False:
        for i in range(10):
            print('nop')
    check_call(['py2dsc', new_name])
    before_deb_name = filename[:-7].lower().replace('-', '_')
    after_deb_name = before_deb_name.replace('rc', '~rc')
    os.rename('deb_dist/%s.orig.tar.gz' % before_deb_name, 'deb_dist/%s+ds.orig.tar.gz' % after_deb_name)
    check_call(['rm -f deb_dist/*_source*'], shell=True)
    os.unlink(new_name)
    entry = None
    for (fullname, entry) in listDir('deb_dist'):
        if os.path.isdir(fullname) and entry.startswith('nuitka') and (not entry.endswith('.orig')):
            break
    if entry is None:
        assert False
    check_call(['rm -r deb_dist/%s/debian/*' % entry], shell=True)
    check_call(['rsync', '-a', '--exclude', 'pbuilder-hookdir', '../debian/', 'deb_dist/%s/debian/' % entry])
    check_call(['rm deb_dist/*.dsc deb_dist/*.debian.tar.xz'], shell=True)
    return entry