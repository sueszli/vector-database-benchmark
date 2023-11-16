import os
import json
import traceback
from lib.lokilogger import *
from lib.helpers import runProcess

class PESieve(object):
    """
    PESieve class makes use of hasherezade's PE-Sieve tool to scans a given process,
    searching for the modules containing in-memory code modifications
    """
    active = False

    def __init__(self, workingDir, is64bit, logger):
        if False:
            print('Hello World!')
        self.logger = logger
        self.peSieve = os.path.join(workingDir, 'tools/pe-sieve32.exe'.replace('/', os.sep))
        if is64bit:
            self.peSieve = os.path.join(workingDir, 'tools/pe-sieve64.exe'.replace('/', os.sep))
        if self.isAvailable():
            self.active = True
            self.logger.log('NOTICE', 'PESieve', 'PE-Sieve successfully initialized BINARY: {0} SOURCE: https://github.com/hasherezade/pe-sieve'.format(self.peSieve))
        else:
            self.logger.log('NOTICE', 'PESieve', 'Cannot find PE-Sieve in expected location {0} SOURCE: https://github.com/hasherezade/pe-sieve'.format(self.peSieve))

    def isAvailable(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Checks if the PE-Sieve tools are available in a "./tools" sub folder\n        :return:\n        '
        if not os.path.exists(self.peSieve):
            self.logger.log('DEBUG', 'PESieve', "PE-Sieve not found in location '{0}' - feature will not be active".format(self.peSieve))
            return False
        return True

    def scan(self, pid, pesieveshellc=False):
        if False:
            for i in range(10):
                print('nop')
        '\n        Performs a scan on a given process ID\n        :param pid: process id of the process to check\n        :return hooked, replaces, suspicious: number of findings per type\n        '
        results = {'patched': 0, 'replaced': 0, 'unreachable_file': 0, 'implanted_pe': 0, 'implanted_shc': 0}
        command = [self.peSieve, '/pid', str(pid), '/ofilter', '2', '/quiet', '/json'] + (['/shellc'] if pesieveshellc else [])
        (output, returnCode) = runProcess(command)
        if self.logger.debug:
            print('PE-Sieve JSON output: %s' % output)
        if output == '' or not output:
            return results
        try:
            results_raw = json.loads(output)
            results = results_raw['scanned']['modified']
        except ValueError:
            traceback.print_exc()
            self.logger.log('DEBUG', 'PESieve', "Couldn't parse the JSON output.")
        except Exception:
            traceback.print_exc()
            self.logger.log('ERROR', 'PESieve', 'Something went wrong during PE-Sieve scan.')
        return results