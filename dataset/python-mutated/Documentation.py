""" Generation of Nuitka documentation.

"""
import os
import sys
from nuitka.Tracing import my_print
from nuitka.utils.Execution import check_call
from nuitka.utils.FileOperations import getFileContents, getFileList, openTextFile, putTextFileContents

def _optimizePNGs(pngList):
    if False:
        for i in range(10):
            print('nop')
    for png in pngList:
        check_call(['optipng', '-o2', '%s.png' % png])

def makeLogoImages():
    if False:
        while True:
            i = 10
    basePathLogo = 'doc/Logo/Nuitka-Logo-%s'
    for logo in ('Vertical', 'Symbol', 'Horizontal'):
        cmd = 'convert -background none %s.svg %s.png' % (basePathLogo, basePathLogo)
        check_call((cmd % (logo, logo)).split())
    _optimizePNGs([basePathLogo % item for item in ('Vertical', 'Symbol', 'Horizontal')])
    if os.path.exists('../Nuitka-website'):
        cmd = 'convert -resize %s doc/Logo/Nuitka-Logo-Symbol.svg %s'
        for (icon, size) in {'../Nuitka-website/files/favicon.ico': '32x32', '../Nuitka-website/files/favicon.png': '32x32', '../Nuitka-website/doc/_static/favicon.ico': '32x32', '../Nuitka-website/doc/_static/favicon.png': '32x32', '../Nuitka-website/doc/_static/apple-touch-icon-ipad.png': '72x72', '../Nuitka-website/doc/_static/apple-touch-icon-ipad3.png': '144x144', '../Nuitka-website/doc/_static/apple-touch-icon-iphone.png': '57x57', '../Nuitka-website/doc/_static/apple-touch-icon-iphone4.png': '114x114', '../Nuitka-website/doc/_static/apple-touch-icon-180x180.png': '180x180'}.items():
            check_call((cmd % (icon, size)).split())
extra_rst_keywords = (b'asciinema', b'postlist', b'post', b'youtube', b'grid', b'toctree', b'automodule')

def checkRstLint(document):
    if False:
        i = 10
        return i + 15
    contents = getFileContents(document, mode='rb')
    for keyword in extra_rst_keywords:
        contents = contents.replace(b'.. %s::' % keyword, b'.. raw:: %s' % keyword)
    import restructuredtext_lint
    my_print("Checking '%s' for proper restructured text ..." % document, style='blue')
    lint_results = restructuredtext_lint.lint(contents.decode('utf8'), document)
    lint_error = False
    for lint_result in lint_results:
        if lint_result.message.startswith('Duplicate implicit target name:'):
            continue
        if lint_result.message.startswith('Error in "raw" directive:\nunknown option: "hidden"'):
            continue
        if lint_result.message.startswith('Error in "raw" directive:\nunknown option: "excerpts"'):
            continue
        if lint_result.message.startswith('Error in "raw" directive:\nunknown option: "members"'):
            continue
        my_print(lint_result, style='yellow')
        lint_error = True
    if lint_error:
        sys.exit('Error, no lint clean rest.')
    my_print('OK.', style='blue')

def makeManPages():
    if False:
        for i in range(10):
            print('nop')
    if not os.path.exists('man'):
        os.mkdir('man')

    def makeManPage(python, suffix):
        if False:
            i = 10
            return i + 15
        cmd = ['help2man', '-n', 'the Python compiler', '--no-discard-stderr', '--no-info', '--include', 'doc/nuitka-man-include.txt', '%s ./bin/nuitka' % python]
        with openTextFile('doc/nuitka%s.1' % suffix, 'wb') as output:
            check_call(cmd, stdout=output)
        cmd[-1] += '-run'
        with openTextFile('doc/nuitka%s-run.1' % suffix, 'wb') as output:
            check_call(cmd, stdout=output)
        for manpage in ('doc/nuitka%s.1' % suffix, 'doc/nuitka%s-run.1' % suffix):
            manpage_contents = getFileContents(manpage).splitlines()
            new_contents = []
            mark = False
            for (count, line) in enumerate(manpage_contents):
                if mark:
                    line = '.SS ' + line + '.BR\n'
                    mark = False
                elif line == '.IP\n' and manpage_contents[count + 1].endswith(':\n'):
                    mark = True
                    continue
                if line == '\\fB\\-\\-g\\fR++\\-only' + '\n':
                    line = '\\fB\\-\\-g\\++\\-only\\fR' + '\n'
                new_contents.append(line)
            putTextFileContents(manpage, contents=new_contents)
    makeManPage('python2', '2')
    makeManPage('python3', '3')

def createReleaseDocumentation():
    if False:
        for i in range(10):
            print('nop')
    checkReleaseDocumentation()
    if os.name != 'nt':
        makeManPages()

def checkReleaseDocumentation():
    if False:
        return 10
    documents = [entry for entry in getFileList('.') if entry.endswith('.rst') and (not entry.startswith('web' + os.path.sep)) if 'inline_copy' not in entry]
    for document in ('README.rst', 'Developer_Manual.rst', 'Changelog.rst'):
        assert document in documents, documents
    for document in documents:
        checkRstLint(document)