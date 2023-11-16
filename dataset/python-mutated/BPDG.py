from __future__ import print_function
from __future__ import absolute_import
import Common.LongFilePathOs as os
import sys
import encodings.ascii
from optparse import OptionParser
from Common import EdkLogger
from Common.BuildToolError import *
from Common.BuildVersion import gBUILD_VERSION
from . import StringTable as st
from . import GenVpd
PROJECT_NAME = st.LBL_BPDG_LONG_UNI
VERSION = st.LBL_BPDG_VERSION + ' Build ' + gBUILD_VERSION

def main():
    if False:
        while True:
            i = 10
    global Options, Args
    EdkLogger.Initialize()
    (Options, Args) = MyOptionParser()
    ReturnCode = 0
    if Options.opt_verbose:
        EdkLogger.SetLevel(EdkLogger.VERBOSE)
    elif Options.opt_quiet:
        EdkLogger.SetLevel(EdkLogger.QUIET)
    elif Options.debug_level is not None:
        EdkLogger.SetLevel(Options.debug_level + 1)
    else:
        EdkLogger.SetLevel(EdkLogger.INFO)
    if Options.bin_filename is None:
        EdkLogger.error('BPDG', ATTRIBUTE_NOT_AVAILABLE, 'Please use the -o option to specify the file name for the VPD binary file')
    if Options.filename is None:
        EdkLogger.error('BPDG', ATTRIBUTE_NOT_AVAILABLE, 'Please use the -m option to specify the file name for the mapping file')
    Force = False
    if Options.opt_force is not None:
        Force = True
    if Args[0] is not None:
        StartBpdg(Args[0], Options.filename, Options.bin_filename, Force)
    else:
        EdkLogger.error('BPDG', ATTRIBUTE_NOT_AVAILABLE, 'Please specify the file which contain the VPD pcd info.', None)
    return ReturnCode

def MyOptionParser():
    if False:
        while True:
            i = 10
    parser = OptionParser(version='%s - Version %s' % (PROJECT_NAME, VERSION), description='', prog='BPDG', usage=st.LBL_BPDG_USAGE)
    parser.add_option('-d', '--debug', action='store', type='int', dest='debug_level', help=st.MSG_OPTION_DEBUG_LEVEL)
    parser.add_option('-v', '--verbose', action='store_true', dest='opt_verbose', help=st.MSG_OPTION_VERBOSE)
    parser.add_option('-q', '--quiet', action='store_true', dest='opt_quiet', default=False, help=st.MSG_OPTION_QUIET)
    parser.add_option('-o', '--vpd-filename', action='store', dest='bin_filename', help=st.MSG_OPTION_VPD_FILENAME)
    parser.add_option('-m', '--map-filename', action='store', dest='filename', help=st.MSG_OPTION_MAP_FILENAME)
    parser.add_option('-f', '--force', action='store_true', dest='opt_force', help=st.MSG_OPTION_FORCE)
    (options, args) = parser.parse_args()
    if len(args) == 0:
        EdkLogger.info('Please specify the filename.txt file which contain the VPD pcd info!')
        EdkLogger.info(parser.usage)
        sys.exit(1)
    return (options, args)

def StartBpdg(InputFileName, MapFileName, VpdFileName, Force):
    if False:
        print('Hello World!')
    if os.path.exists(VpdFileName) and (not Force):
        print('\nFile %s already exist, Overwrite(Yes/No)?[Y]: ' % VpdFileName)
        choice = sys.stdin.readline()
        if choice.strip().lower() not in ['y', 'yes', '']:
            return
    GenVPD = GenVpd.GenVPD(InputFileName, MapFileName, VpdFileName)
    EdkLogger.info('%-24s = %s' % ('VPD input data file: ', InputFileName))
    EdkLogger.info('%-24s = %s' % ('VPD output map file: ', MapFileName))
    EdkLogger.info('%-24s = %s' % ('VPD output binary file: ', VpdFileName))
    GenVPD.ParserInputFile()
    GenVPD.FormatFileLine()
    GenVPD.FixVpdOffset()
    GenVPD.GenerateVpdFile(MapFileName, VpdFileName)
    EdkLogger.info('- Vpd pcd fixed done! -')
if __name__ == '__main__':
    try:
        r = main()
    except FatalError as e:
        r = e
    if r < 0 or r > 127:
        r = 1
    sys.exit(r)