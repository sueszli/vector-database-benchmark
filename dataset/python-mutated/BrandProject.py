import os
import sys
import bulkstamp
import vssutil
import win32api

def BrandProject(vssProjectName, descFile, stampPath, filesToSubstitute, buildDesc=None, auto=0, bRebrand=0):
    if False:
        for i in range(10):
            print('nop')
    path = win32api.GetFullPathName(stampPath)
    build = vssutil.MakeNewBuildNo(vssProjectName, buildDesc, auto, bRebrand)
    if build is None:
        print('Cancelled')
        return
    bulkstamp.scan(build, stampPath, descFile)
    for (infile, outfile) in filesToSubstitute:
        SubstituteVSSInFile(vssProjectName, infile, outfile)
    return 1

def usage(msg):
    if False:
        while True:
            i = 10
    print(msg)
    print(f'{os.path.basename(sys.argv[0])} Usage:\n{os.path.basename(sys.argv[0])} [options] vssProject descFile stampPath\n\nAutomatically brand a VSS project with an automatically incremented\nbuild number, and stamp DLL/EXE files with the build number.\n\nChecks that no files are checked out in the project, and finds the last\nbuild number, and suggests the next number.\n\nOptions:\n-a     - Auto increment the build number, and brand (otherwise prompt\n         for the build number after looking for the previous)\n-r     - Restamp the files with the existing build number.\n-d     - A description for the VSS Label.\n-f infile=outfile - Substitute special VSS labels in the specified text\n                    file with the text extracted from VSS.\n')
    sys.exit(1)
if __name__ == '__main__':
    try:
        import getopt
        (opts, args) = getopt.getopt(sys.argv[1:], 'af:d:r')
    except getopts.error as msg:
        usage(msg)
    bAuto = bRebrand = 0
    stampFiles = []
    desc = None
    for (opt, val) in opts:
        if opt == '-a':
            bAuto = 1
        if opt == '-f':
            (infile, outfile) = val.split('=', 2)
            stampFiles.append((infile, outfile))
        if opt == '-d':
            desc = val
        if opt == '-r':
            bRebrand = 1
    if len(args) < 3:
        usage('You must specify the required arguments')
    vssProjectName = '$\\' + args[0]
    descFile = args[1]
    path = args[2]
    try:
        os.stat(descFile)
    except OSError:
        usage("The description file '%s' can not be found" % descFile)
    if not os.path.isdir(path):
        usage("The path to the files to stamp '%s' does not exist" % path)
    BrandProject(vssProjectName, descFile, path, stampFiles, desc, bAuto, bRebrand)