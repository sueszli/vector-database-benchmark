"""
RunMakefile.py
"""
import os
import sys
import argparse
import subprocess
__prog__ = 'RunMakefile'
__version__ = '%s Version %s' % (__prog__, '1.0')
__copyright__ = 'Copyright (c) 2017, Intel Corporation. All rights reserved.'
__description__ = 'Run a makefile as part of a PREBUILD or POSTBUILD action.\n'
gArgs = None

def Log(Message):
    if False:
        return 10
    if not gArgs.Verbose:
        return
    sys.stdout.write(__prog__ + ': ' + Message + '\n')

def Error(Message, ExitValue=1):
    if False:
        for i in range(10):
            print('nop')
    sys.stderr.write(__prog__ + ': ERROR: ' + Message + '\n')
    sys.exit(ExitValue)

def RelativePath(target):
    if False:
        print('Hello World!')
    return os.path.relpath(target, gWorkspace)

def NormalizePath(target):
    if False:
        for i in range(10):
            print('nop')
    if isinstance(target, tuple):
        return os.path.normpath(os.path.join(*target))
    else:
        return os.path.normpath(target)
if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog=__prog__, version=__version__, description=__description__ + __copyright__, conflict_handler='resolve')
    parser.add_argument('-a', '--arch', dest='Arch', nargs='+', action='append', required=True, help="ARCHS is one of list: IA32, X64, IPF, ARM, AARCH64 or EBC,\n                     which overrides target.txt's TARGET_ARCH definition. To\n                     specify more archs, please repeat this option.")
    parser.add_argument('-t', '--tagname', dest='ToolChain', required=True, help="Using the Tool Chain Tagname to build the platform,\n                     overriding target.txt's TOOL_CHAIN_TAG definition.")
    parser.add_argument('-p', '--platform', dest='PlatformFile', required=True, help="Build the platform specified by the DSC file name argument,\n                     overriding target.txt's ACTIVE_PLATFORM definition.")
    parser.add_argument('-b', '--buildtarget', dest='BuildTarget', required=True, help="Using the TARGET to build the platform, overriding\n                     target.txt's TARGET definition.")
    parser.add_argument('--conf=', dest='ConfDirectory', required=True, help='Specify the customized Conf directory.')
    parser.add_argument('-D', '--define', dest='Define', nargs='*', action='append', help='Macro: "Name [= Value]".')
    parser.add_argument('--makefile', dest='Makefile', required=True, help='Makefile to run passing in arguments as makefile defines.')
    parser.add_argument('-v', '--verbose', dest='Verbose', action='store_true', help='Turn on verbose output with informational messages printed')
    (gArgs, remaining) = parser.parse_known_args()
    gArgs.BuildType = 'all'
    for BuildType in ['all', 'fds', 'genc', 'genmake', 'clean', 'cleanall', 'modules', 'libraries', 'run']:
        if BuildType in remaining:
            gArgs.BuildType = BuildType
            remaining.remove(BuildType)
            break
    gArgs.Remaining = ' '.join(remaining)
    Log('Start')
    PathList = ['']
    try:
        PathList.append(os.environ['WORKSPACE'])
    except:
        Error('WORKSPACE environment variable not set')
    try:
        PathList += os.environ['PACKAGES_PATH'].split(os.pathsep)
    except:
        pass
    for Path in PathList:
        Makefile = NormalizePath((Path, gArgs.Makefile))
        if os.path.exists(Makefile):
            break
    if not os.path.exists(Makefile):
        Error('makefile %s not found' % gArgs.Makefile)
    CommandLine = [Makefile]
    CommandLine.append('TARGET_ARCH="%s"' % ' '.join([Item[0] for Item in gArgs.Arch]))
    CommandLine.append('TOOL_CHAIN_TAG="%s"' % gArgs.ToolChain)
    CommandLine.append('TARGET="%s"' % gArgs.BuildTarget)
    CommandLine.append('ACTIVE_PLATFORM="%s"' % gArgs.PlatformFile)
    CommandLine.append('CONF_DIRECTORY="%s"' % gArgs.ConfDirectory)
    if gArgs.Define:
        for Item in gArgs.Define:
            if '=' not in Item[0]:
                continue
            Item = Item[0].split('=', 1)
            CommandLine.append('%s="%s"' % (Item[0], Item[1]))
    CommandLine.append('EXTRA_FLAGS="%s"' % gArgs.Remaining)
    CommandLine.append(gArgs.BuildType)
    if sys.platform == 'win32':
        CommandLine = 'nmake /f %s' % ' '.join(CommandLine)
    else:
        CommandLine = 'make -f %s' % ' '.join(CommandLine)
    try:
        Process = subprocess.Popen(CommandLine, shell=True)
    except:
        Error('make command not available.  Please verify PATH')
    Process.communicate()
    Log('Done')
    sys.exit(Process.returncode)