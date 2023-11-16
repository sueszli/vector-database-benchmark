import frappe

class DbManager:

    def __init__(self, db):
        if False:
            while True:
                i = 10
        '\n\t\tPass root_conn here for access to all databases.\n\t\t'
        if db:
            self.db = db

    def get_current_host(self):
        if False:
            print('Hello World!')
        return self.db.sql('select user()')[0][0].split('@')[1]

    def create_user(self, user, password, host=None):
        if False:
            i = 10
            return i + 15
        host = host or self.get_current_host()
        password_predicate = f" IDENTIFIED BY '{password}'" if password else ''
        self.db.sql(f"CREATE USER '{user}'@'{host}'{password_predicate}")

    def delete_user(self, target, host=None):
        if False:
            while True:
                i = 10
        host = host or self.get_current_host()
        self.db.sql(f"DROP USER IF EXISTS '{target}'@'{host}'")

    def create_database(self, target):
        if False:
            while True:
                i = 10
        if target in self.get_database_list():
            self.drop_database(target)
        self.db.sql(f'CREATE DATABASE `{target}`')

    def drop_database(self, target):
        if False:
            i = 10
            return i + 15
        self.db.sql_ddl(f'DROP DATABASE IF EXISTS `{target}`')

    def grant_all_privileges(self, target, user, host=None):
        if False:
            return 10
        host = host or self.get_current_host()
        permissions = 'SELECT, INSERT, UPDATE, DELETE, CREATE, DROP, INDEX, ALTER, CREATE TEMPORARY TABLES, CREATE VIEW, EVENT, TRIGGER, SHOW VIEW, CREATE ROUTINE, ALTER ROUTINE, EXECUTE, LOCK TABLES' if frappe.conf.rds_db else 'ALL PRIVILEGES'
        self.db.sql(f"GRANT {permissions} ON `{target}`.* TO '{user}'@'{host}'")

    def flush_privileges(self):
        if False:
            return 10
        self.db.sql('FLUSH PRIVILEGES')

    def get_database_list(self):
        if False:
            return 10
        return self.db.sql('SHOW DATABASES', pluck=True)

    @staticmethod
    def restore_database(target, source, user, password):
        if False:
            i = 10
            return i + 15
        import os
        from shutil import which
        from frappe.utils import make_esc
        esc = make_esc('$ ')
        pv = which('pv')
        mariadb_cli = which('mariadb') or which('mysql')
        if pv:
            pipe = f'{pv} {source} |'
            source = ''
        else:
            pipe = ''
            source = f'< {source}'
        if pipe:
            print('Restoring Database file...')
        command = '{pipe} {mariadb_cli} -u {user} -p{password} -h{host} -P{port} {target} {source}'
        command = command.format(pipe=pipe, user=esc(user), password=esc(password), host=esc(frappe.conf.db_host), target=esc(target), source=source, port=frappe.conf.db_port, mariadb_cli=mariadb_cli)
        os.system(command)
        frappe.cache.delete_keys('')