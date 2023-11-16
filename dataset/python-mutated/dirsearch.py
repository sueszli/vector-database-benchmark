import sys
from pkg_resources import DistributionNotFound, VersionConflict
from lib.core.data import options
from lib.core.exceptions import FailedDependenciesInstallation
from lib.core.installation import check_dependencies, install_dependencies
from lib.core.settings import OPTIONS_FILE
from lib.parse.config import ConfigParser
if sys.version_info < (3, 7):
    sys.stdout.write('Sorry, dirsearch requires Python 3.7 or higher\n')
    sys.exit(1)

def main():
    if False:
        while True:
            i = 10
    config = ConfigParser()
    config.read(OPTIONS_FILE)
    if config.safe_getboolean('options', 'check-dependencies', False):
        try:
            check_dependencies()
        except (DistributionNotFound, VersionConflict):
            option = input('Missing required dependencies to run.\nDo you want dirsearch to automatically install them? [Y/n] ')
            if option.lower() == 'y':
                print('Installing required dependencies...')
                try:
                    install_dependencies()
                except FailedDependenciesInstallation:
                    print('Failed to install dirsearch dependencies, try doing it manually.')
                    exit(1)
            else:
                config.set('options', 'check-dependencies', 'False')
                with open(OPTIONS_FILE, 'w') as fh:
                    config.write(fh)
    from lib.core.options import parse_options
    options.update(parse_options())
    from lib.controller.controller import Controller
    Controller()
if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        pass