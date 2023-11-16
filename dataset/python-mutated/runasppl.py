class CMEModule:
    name = 'runasppl'
    description = 'Check if the registry value RunAsPPL is set or not'
    supported_protocols = ['smb']
    opsec_safe = True
    multiple_hosts = True

    def __init__(self, context=None, module_options=None):
        if False:
            for i in range(10):
                print('nop')
        self.context = context
        self.module_options = module_options

    def options(self, context, module_options):
        if False:
            for i in range(10):
                print('nop')
        ''

    def on_admin_login(self, context, connection):
        if False:
            i = 10
            return i + 15
        command = 'reg query HKEY_LOCAL_MACHINE\\SYSTEM\\CurrentControlSet\\Control\\Lsa\\ /v RunAsPPL'
        context.log.display('Executing command')
        p = connection.execute(command, True)
        if 'The system was unable to find the specified registry key or value' in p:
            context.log.debug(f'Unable to find RunAsPPL Registry Key')
        else:
            context.log.highlight(p)