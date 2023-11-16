from csv import reader

class CMEModule:
    """
    Search for KeePass-related files and process

    Module by @d3lb3
    Inspired by @harmj0y https://raw.githubusercontent.com/GhostPack/KeeThief/master/PowerShell/KeePassConfig.ps1
    """
    name = 'keepass_discover'
    description = 'Search for KeePass-related files and process.'
    supported_protocols = ['smb']
    opsec_safe = True
    multiple_hosts = True

    def __init__(self):
        if False:
            while True:
                i = 10
        self.search_type = 'ALL'
        self.search_path = "'C:\\Users\\','$env:PROGRAMFILES','env:ProgramFiles(x86)'"

    def options(self, context, module_options):
        if False:
            for i in range(10):
                print('nop')
        "\n        SEARCH_TYPE     Specify what to search, between:\n                          PROCESS     Look for running KeePass.exe process only\n                          FILES       Look for KeePass-related files (KeePass.config.xml, .kdbx, KeePass.exe) only, may take some time\n                          ALL         Look for running KeePass.exe process and KeePass-related files (default)\n\n        SEARCH_PATH     Comma-separated remote locations where to search for KeePass-related files (you must add single quotes around the paths if they include spaces)\n                        Default: 'C:\\Users\\','$env:PROGRAMFILES','env:ProgramFiles(x86)'\n        "
        if 'SEARCH_PATH' in module_options:
            self.search_path = module_options['SEARCH_PATH']
        if 'SEARCH_TYPE' in module_options:
            self.search_type = module_options['SEARCH_TYPE']

    def on_admin_login(self, context, connection):
        if False:
            return 10
        if self.search_type == 'ALL' or self.search_type == 'PROCESS':
            search_keepass_process_command_str = 'powershell.exe "Get-Process kee* -IncludeUserName | Select-Object -Property Id,UserName,ProcessName | ConvertTo-CSV -NoTypeInformation"'
            search_keepass_process_output_csv = connection.execute(search_keepass_process_command_str, True)
            csv_reader = reader(search_keepass_process_output_csv.split('\n'), delimiter=',')
            next(csv_reader)
            row_number = 0
            for row in csv_reader:
                row_number += 1
                keepass_process_id = row[0]
                keepass_process_username = row[1]
                keepass_process_name = row[2]
                context.log.highlight('Found process "{}" with PID {} (user {})'.format(keepass_process_name, keepass_process_id, keepass_process_username))
            if row_number == 0:
                context.log.display('No KeePass-related process was found')
        if self.search_type == 'ALL' or self.search_type == 'FILES':
            search_keepass_files_payload = "Get-ChildItem -Path {} -Recurse -Force -Include ('KeePass.config.xml','KeePass.exe','*.kdbx') -ErrorAction SilentlyContinue | Select FullName -ExpandProperty FullName".format(self.search_path)
            search_keepass_files_cmd = 'powershell.exe "{}"'.format(search_keepass_files_payload)
            search_keepass_files_output = connection.execute(search_keepass_files_cmd, True).split('\r\n')
            found = False
            found_xml = False
            for file in search_keepass_files_output:
                if 'KeePass' in file or 'kdbx' in file:
                    if 'xml' in file:
                        found_xml = True
                    found = True
                    context.log.highlight('Found {}'.format(file))
            if not found:
                context.log.display('No KeePass-related file were found')
            elif not found_xml:
                context.log.fail('No config settings file found !!!')