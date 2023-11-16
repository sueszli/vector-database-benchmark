import argparse
import codecs
try:
    from argcomplete import autocomplete
except ImportError:

    def autocomplete(parser):
        if False:
            return 10
        return None

def run():
    if False:
        print('Hello World!')
    parser = argparse.ArgumentParser(epilog='The printed output may be saved to a file, edited and used as the input for a version resource on any of the executable targets in a PyInstaller .spec file.')
    parser.add_argument('exe_file', metavar='exe-file', help='full pathname of a Windows executable')
    parser.add_argument('out_filename', metavar='out-filename', nargs='?', default='file_version_info.txt', help='filename where the grabbed version info will be saved')
    autocomplete(parser)
    args = parser.parse_args()
    try:
        from PyInstaller.utils.win32 import versioninfo
        info = versioninfo.read_version_info_from_executable(args.exe_file)
        if not info:
            raise SystemExit('Error: VersionInfo resource not found in exe')
        with codecs.open(args.out_filename, 'w', 'utf-8') as fp:
            fp.write(str(info))
        print(f'Version info written to: {args.out_filename!r}')
    except KeyboardInterrupt:
        raise SystemExit('Aborted by user request.')
if __name__ == '__main__':
    run()