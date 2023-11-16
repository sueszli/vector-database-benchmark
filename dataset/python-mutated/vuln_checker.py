import subprocess

class VulnChecker:

    def __init__(self, logger):
        if False:
            return 10
        self.logger = logger
        pass

    def run(self):
        if False:
            i = 10
            return i + 15
        self.logger.log('INFO', 'VulnChecker', 'Starting vulnerability checks ...')
        self.check_sam_readable()

    def check_sam_readable(self):
        if False:
            print('Hello World!')
        '\n        Check if the local SAM is readable by everyone\n        https://twitter.com/wdormann/status/1417447179149533185\n        :return:\n        '
        output = b''
        try:
            output += subprocess.check_output(['icacls.exe', 'C:\\Windows\\System32\\config\\sam'], stderr=subprocess.STDOUT)
        except subprocess.CalledProcessError:
            pass
        try:
            output += subprocess.check_output(['icacls.exe', 'C:\\Windows\\SysNative\\config\\sam'], stderr=subprocess.STDOUT)
        except subprocess.CalledProcessError:
            pass
        try:
            if 'BUILTIN\\Users:(I)(RX)' in output.decode('latin1', errors='ignore'):
                self.logger.log('WARNING', 'VulnChecker', 'The Security Account Manager (SAM) database file C:\\Windows\\System32\\config\\SAM is readable by every user. This is caused by the Hive Permission Bug, which is problematic on systems that have System Protection configured for drive C: (see https://doublepulsar.com/hivenightmare-aka-serioussam-anybody-can-read-the-registry-in-windows-10-7a871c465fa5)')
                return True
            else:
                self.logger.log('DEBUG', 'VulnChecker', "SAM Database isn't readable by every user.")
        except UnicodeDecodeError:
            self.logger.log('ERROR', 'VulnChecker', 'Unicode decode error in SAM check')
        return False