import servicemanager
import win32api
import win32process
import win32service
import win32serviceutil
import subprocess
import sys
from os.path import dirname, join, split
execfile(join(dirname(__file__), '..', 'server', 'odoo', 'release.py'))

class OdooService(win32serviceutil.ServiceFramework):
    _svc_name_ = nt_service_name
    _svc_display_name_ = '%s %s' % (nt_service_name, serie)

    def __init__(self, args):
        if False:
            for i in range(10):
                print('nop')
        win32serviceutil.ServiceFramework.__init__(self, args)
        self.odooprocess = None

    def SvcStop(self):
        if False:
            return 10
        self.ReportServiceStatus(win32service.SERVICE_STOP_PENDING)
        win32api.TerminateProcess(int(self.odooprocess._handle), 0)
        servicemanager.LogInfoMsg('Odoo stopped correctly')

    def SvcDoRun(self):
        if False:
            print('Hello World!')
        service_dir = dirname(sys.argv[0])
        server_dir = split(service_dir)[0]
        server_path = join(server_dir, 'server', 'odoo-bin.exe')
        self.odooprocess = subprocess.Popen([server_path], cwd=server_dir, creationflags=win32process.CREATE_NO_WINDOW)
        servicemanager.LogInfoMsg('Odoo up and running')
        sys.exit(self.odooprocess.wait())

def option_handler(opts):
    if False:
        i = 10
        return i + 15
    subprocess.call(['sc', 'failure', nt_service_name, 'reset=', '0', 'actions=', 'restart/0/restart/0/restart/0'])
if __name__ == '__main__':
    win32serviceutil.HandleCommandLine(OdooService, customOptionHandler=option_handler)