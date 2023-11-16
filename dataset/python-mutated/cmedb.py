import cmd
import configparser
import csv
import os
from os import listdir
from os.path import exists
from os.path import join as path_join
import shutil
from sqlite3 import connect
import sys
from textwrap import dedent
from requests import get, post, ConnectionError
from sqlalchemy import create_engine
from terminaltables import AsciiTable
from cme.loaders.protocolloader import ProtocolLoader
from cme.paths import CONFIG_PATH, WS_PATH, WORKSPACE_DIR

class UserExitedProto(Exception):
    pass

def create_db_engine(db_path):
    if False:
        i = 10
        return i + 15
    db_engine = create_engine(f'sqlite:///{db_path}', isolation_level='AUTOCOMMIT', future=True)
    return db_engine

def print_table(data, title=None):
    if False:
        return 10
    print('')
    table = AsciiTable(data)
    if title:
        table.title = title
    print(table.table)
    print('')

def write_csv(filename, headers, entries):
    if False:
        for i in range(10):
            print('nop')
    '\n    Writes a CSV file with the provided parameters.\n    '
    with open(os.path.expanduser(filename), 'w') as export_file:
        csv_file = csv.writer(export_file, delimiter=';', quoting=csv.QUOTE_ALL, lineterminator='\n', escapechar='\\')
        csv_file.writerow(headers)
        for entry in entries:
            csv_file.writerow(entry)

def write_list(filename, entries):
    if False:
        i = 10
        return i + 15
    '\n    Writes a file with a simple list\n    '
    with open(os.path.expanduser(filename), 'w') as export_file:
        for line in entries:
            export_file.write(line + '\n')
    return

def complete_import(text, line):
    if False:
        while True:
            i = 10
    "\n    Tab-complete 'import' commands\n    "
    commands = ('empire', 'metasploit')
    mline = line.partition(' ')[2]
    offs = len(mline) - len(text)
    return [s[offs:] for s in commands if s.startswith(mline)]

def complete_export(text, line):
    if False:
        i = 10
        return i + 15
    "\n    Tab-complete 'creds' commands.\n    "
    commands = ('creds', 'plaintext', 'hashes', 'shares', 'local_admins', 'signing', 'keys')
    mline = line.partition(' ')[2]
    offs = len(mline) - len(text)
    return [s[offs:] for s in commands if s.startswith(mline)]

def print_help(help_string):
    if False:
        for i in range(10):
            print('nop')
    print(dedent(help_string))

class DatabaseNavigator(cmd.Cmd):

    def __init__(self, main_menu, database, proto):
        if False:
            print('Hello World!')
        cmd.Cmd.__init__(self)
        self.main_menu = main_menu
        self.config = main_menu.config
        self.proto = proto
        self.db = database
        self.prompt = f'cmedb ({main_menu.workspace})({proto}) > '

    def do_exit(self, line):
        if False:
            while True:
                i = 10
        self.db.shutdown_db()
        sys.exit()

    @staticmethod
    def help_exit():
        if False:
            print('Hello World!')
        help_string = '\n        Exits\n        '
        print_help(help_string)

    def do_back(self, line):
        if False:
            for i in range(10):
                print('nop')
        raise UserExitedProto

    def do_export(self, line):
        if False:
            while True:
                i = 10
        if not line:
            print('[-] not enough arguments')
            return
        line = line.split()
        command = line[0].lower()
        if command == 'creds':
            if len(line) < 3:
                print('[-] invalid arguments, export creds <simple|detailed|hashcat> <filename>')
                return
            filename = line[2]
            creds = self.db.get_credentials()
            csv_header = ('id', 'domain', 'username', 'password', 'credtype', 'pillaged_from')
            if line[1].lower() == 'simple':
                write_csv(filename, csv_header, creds)
            elif line[1].lower() == 'detailed':
                formatted_creds = []
                for cred in creds:
                    entry = [cred[0], cred[1], cred[2], cred[3], cred[4]]
                    if cred[5] is None:
                        entry.append('')
                    else:
                        entry.append(self.db.get_hosts(cred[5])[0][2])
                    formatted_creds.append(entry)
                write_csv(filename, csv_header, formatted_creds)
            elif line[1].lower() == 'hashcat':
                usernames = []
                passwords = []
                for cred in creds:
                    if cred[4] == 'hash':
                        usernames.append(cred[2])
                        passwords.append(cred[3])
                output_list = [':'.join(combination) for combination in zip(usernames, passwords)]
                write_list(filename, output_list)
            else:
                print(f'[-] No such export option: {line[1]}')
                return
            print('[+] Creds exported')
        elif command == 'hosts':
            if len(line) < 3:
                print('[-] invalid arguments, export hosts <simple|detailed|signing> <filename>')
                return
            csv_header_simple = ('id', 'ip', 'hostname', 'domain', 'os', 'dc', 'smbv1', 'signing')
            csv_header_detailed = ('id', 'ip', 'hostname', 'domain', 'os', 'dc', 'smbv1', 'signing', 'spooler', 'zerologon', 'petitpotam')
            filename = line[2]
            if line[1].lower() == 'simple':
                hosts = self.db.get_hosts()
                simple_hosts = [host[:8] for host in hosts]
                write_csv(filename, csv_header_simple, simple_hosts)
            elif line[1].lower() == 'detailed':
                hosts = self.db.get_hosts()
                write_csv(filename, csv_header_detailed, hosts)
            elif line[1].lower() == 'signing':
                hosts = self.db.get_hosts('signing')
                signing_hosts = [host[1] for host in hosts]
                write_list(filename, signing_hosts)
            else:
                print(f'[-] No such export option: {line[1]}')
                return
            print('[+] Hosts exported')
        elif command == 'shares':
            if len(line) < 3:
                print('[-] invalid arguments, export shares <simple|detailed> <filename>')
                return
            shares = self.db.get_shares()
            csv_header = ('id', 'host', 'userid', 'name', 'remark', 'read', 'write')
            filename = line[2]
            if line[1].lower() == 'simple':
                write_csv(filename, csv_header, shares)
                print('[+] shares exported')
            elif line[1].lower() == 'detailed':
                formatted_shares = []
                for share in shares:
                    user = self.db.get_users(share[2])[0]
                    if self.db.get_hosts(share[1]):
                        share_host = self.db.get_hosts(share[1])[0][2]
                    else:
                        share_host = 'ERROR'
                    entry = (share[0], share_host, f'{user[1]}\\{user[2]}', share[3], share[4], bool(share[5]), bool(share[6]))
                    formatted_shares.append(entry)
                write_csv(filename, csv_header, formatted_shares)
                print('[+] Shares exported')
            else:
                print(f'[-] No such export option: {line[1]}')
                return
        elif command == 'local_admins':
            if len(line) < 3:
                print('[-] invalid arguments, export local_admins <simple|detailed> <filename>')
                return
            local_admins = self.db.get_admin_relations()
            csv_header = ('id', 'userid', 'host')
            filename = line[2]
            if line[1].lower() == 'simple':
                write_csv(filename, csv_header, local_admins)
            elif line[1].lower() == 'detailed':
                formatted_local_admins = []
                for entry in local_admins:
                    user = self.db.get_users(filter_term=entry[1])[0]
                    formatted_entry = (entry[0], f'{user[1]}/{user[2]}', self.db.get_hosts(filter_term=entry[2])[0][2])
                    formatted_local_admins.append(formatted_entry)
                write_csv(filename, csv_header, formatted_local_admins)
            else:
                print(f'[-] No such export option: {line[1]}')
                return
            print('[+] Local Admins exported')
        elif command == 'dpapi':
            if len(line) < 3:
                print('[-] invalid arguments, export dpapi <simple|detailed> <filename>')
                return
            dpapi_secrets = self.db.get_dpapi_secrets()
            csv_header = ('id', 'host', 'dpapi_type', 'windows_user', 'username', 'password', 'url')
            filename = line[2]
            if line[1].lower() == 'simple':
                write_csv(filename, csv_header, dpapi_secrets)
            elif line[1].lower() == 'detailed':
                formatted_dpapi_secret = []
                for entry in dpapi_secrets:
                    formatted_entry = (entry[0], self.db.get_hosts(filter_term=entry[1])[0][2], entry[2], entry[3], entry[4], entry[5], entry[6])
                    formatted_dpapi_secret.append(formatted_entry)
                write_csv(filename, csv_header, formatted_dpapi_secret)
            else:
                print(f'[-] No such export option: {line[1]}')
                return
            print('[+] DPAPI secrets exported')
        elif command == 'keys':
            if line[1].lower() == 'all':
                keys = self.db.get_keys()
            else:
                keys = self.db.get_keys(key_id=int(line[1]))
            writable_keys = [key[2] for key in keys]
            filename = line[2]
            write_list(filename, writable_keys)
        elif command == 'wcc':
            if len(line) < 3:
                print('[-] invalid arguments, export wcc <simple|detailed> <filename>')
                return
            csv_header_simple = ('id', 'ip', 'hostname', 'check', 'status')
            csv_header_detailed = ('id', 'ip', 'hostname', 'check', 'description', 'status', 'reasons')
            filename = line[2]
            host_mapping = {}
            check_mapping = {}
            hosts = self.db.get_hosts()
            checks = self.db.get_checks()
            check_results = self.db.get_check_results()
            rows = []
            for (result_id, hostid, checkid, secure, reasons) in check_results:
                row = [result_id]
                if hostid in host_mapping:
                    row.extend(host_mapping[hostid])
                else:
                    for (host_id, ip, hostname, _, _, _, _, _, _, _, _) in hosts:
                        if host_id == hostid:
                            row.extend([ip, hostname])
                            host_mapping[hostid] = [ip, hostname]
                            break
                if checkid in check_mapping:
                    row.extend(check_mapping[checkid])
                else:
                    for check in checks:
                        (check_id, name, description) = check
                        if check_id == checkid:
                            row.extend([name, description])
                            check_mapping[checkid] = [name, description]
                            break
                row.append('OK' if secure else 'KO')
                row.append(reasons)
                rows.append(row)
            if line[1].lower() == 'simple':
                simple_rows = list(((row[0], row[1], row[2], row[3], row[5]) for row in rows))
                write_csv(filename, csv_header_simple, simple_rows)
            elif line[1].lower() == 'detailed':
                write_csv(filename, csv_header_detailed, rows)
            elif line[1].lower() == 'signing':
                hosts = self.db.get_hosts('signing')
                signing_hosts = [host[1] for host in hosts]
                write_list(filename, signing_hosts)
            else:
                print(f'[-] No such export option: {line[1]}')
                return
            print('[+] WCC exported')
        else:
            print('[-] Invalid argument, specify creds, hosts, local_admins, shares, wcc or dpapi')

    @staticmethod
    def help_export():
        if False:
            print('Hello World!')
        help_string = '\n        export [creds|hosts|local_admins|shares|signing|keys] [simple|detailed|*] [filename]\n        Exports information to a specified file\n        \n        * hosts has an additional third option from simple and detailed: signing - this simply writes a list of ips of\n        hosts where signing is enabled\n        * keys\' third option is either "all" or an id of a key to export\n            export keys [all|id] [filename]\n        '
        print_help(help_string)

    def do_import(self, line):
        if False:
            i = 10
            return i + 15
        if not line:
            return
        if line == 'empire':
            headers = {'Content-Type': 'application/json'}
            payload = {'username': self.config.get('Empire', 'username'), 'password': self.config.get('Empire', 'password')}
            base_url = f"https://{self.config.get('Empire', 'api_host')}:{self.config.get('Empire', 'api_port')}"
            try:
                r = post(base_url + '/api/admin/login', json=payload, headers=headers, verify=False)
                if r.status_code == 200:
                    token = r.json()['token']
                    url_params = {'token': token}
                    r = get(base_url + '/api/creds', headers=headers, params=url_params, verify=False)
                    creds = r.json()
                    for cred in creds['creds']:
                        if cred['credtype'] == 'token' or cred['credtype'] == 'krbtgt' or cred['username'].endswith('$'):
                            continue
                        self.db.add_credential(cred['credtype'], cred['domain'], cred['username'], cred['password'])
                    print('[+] Empire credential import successful')
                else:
                    print("[-] Error authenticating to Empire's RESTful API server!")
            except ConnectionError as e:
                print(f"[-] Unable to connect to Empire's RESTful API server: {e}")

class CMEDBMenu(cmd.Cmd):

    def __init__(self, config_path):
        if False:
            i = 10
            return i + 15
        cmd.Cmd.__init__(self)
        self.config_path = config_path
        try:
            self.config = configparser.ConfigParser()
            self.config.read(self.config_path)
        except Exception as e:
            print(f'[-] Error reading cme.conf: {e}')
            sys.exit(1)
        self.conn = None
        self.p_loader = ProtocolLoader()
        self.protocols = self.p_loader.get_protocols()
        self.workspace = self.config.get('CME', 'workspace')
        self.do_workspace(self.workspace)
        self.db = self.config.get('CME', 'last_used_db')
        if self.db:
            self.do_proto(self.db)

    def write_configfile(self):
        if False:
            return 10
        with open(self.config_path, 'w') as configfile:
            self.config.write(configfile)

    def do_proto(self, proto):
        if False:
            while True:
                i = 10
        if not proto:
            return
        proto_db_path = path_join(WORKSPACE_DIR, self.workspace, f'{proto}.db')
        if exists(proto_db_path):
            self.conn = create_db_engine(proto_db_path)
            db_nav_object = self.p_loader.load_protocol(self.protocols[proto]['nvpath'])
            db_object = self.p_loader.load_protocol(self.protocols[proto]['dbpath'])
            self.config.set('CME', 'last_used_db', proto)
            self.write_configfile()
            try:
                proto_menu = getattr(db_nav_object, 'navigator')(self, getattr(db_object, 'database')(self.conn), proto)
                proto_menu.cmdloop()
            except UserExitedProto:
                pass

    @staticmethod
    def help_proto():
        if False:
            for i in range(10):
                print('nop')
        help_string = '\n        proto [smb|mssql|winrm]\n            *unimplemented protocols: ftp, rdp, ldap, ssh\n        Changes cmedb to the specified protocol\n        '
        print_help(help_string)

    def do_workspace(self, line):
        if False:
            while True:
                i = 10
        line = line.strip()
        if not line:
            subcommand = ''
            self.help_workspace()
        else:
            subcommand = line.split()[0]
        if subcommand == 'create':
            new_workspace = line.split()[1].strip()
            print(f"[*] Creating workspace '{new_workspace}'")
            self.create_workspace(new_workspace, self.p_loader, self.protocols)
            self.do_workspace(new_workspace)
        elif subcommand == 'list':
            print('[*] Enumerating Workspaces')
            for workspace in listdir(path_join(WORKSPACE_DIR)):
                if workspace == self.workspace:
                    print('==> ' + workspace)
                else:
                    print(workspace)
        elif exists(path_join(WORKSPACE_DIR, line)):
            self.config.set('CME', 'workspace', line)
            self.write_configfile()
            self.workspace = line
            self.prompt = f'cmedb ({line}) > '

    @staticmethod
    def help_workspace():
        if False:
            i = 10
            return i + 15
        help_string = '\n        workspace [create <targetName> | workspace list | workspace <targetName>]\n        '
        print_help(help_string)

    @staticmethod
    def do_exit(line):
        if False:
            for i in range(10):
                print('nop')
        sys.exit()

    @staticmethod
    def help_exit():
        if False:
            for i in range(10):
                print('nop')
        help_string = '\n        Exits\n        '
        print_help(help_string)

    @staticmethod
    def create_workspace(workspace_name, p_loader, protocols):
        if False:
            print('Hello World!')
        os.mkdir(path_join(WORKSPACE_DIR, workspace_name))
        for protocol in protocols.keys():
            protocol_object = p_loader.load_protocol(protocols[protocol]['dbpath'])
            proto_db_path = path_join(WORKSPACE_DIR, workspace_name, f'{protocol}.db')
            if not exists(proto_db_path):
                print(f'[*] Initializing {protocol.upper()} protocol database')
                conn = connect(proto_db_path)
                c = conn.cursor()
                c.execute('PRAGMA journal_mode = OFF')
                c.execute('PRAGMA foreign_keys = 1')
                getattr(protocol_object, 'database').db_schema(c)
                conn.commit()
                conn.close()

def delete_workspace(workspace_name):
    if False:
        for i in range(10):
            print('nop')
    shutil.rmtree(path_join(WORKSPACE_DIR, workspace_name))

def initialize_db(logger):
    if False:
        return 10
    if not exists(path_join(WS_PATH, 'default')):
        logger.debug('Creating default workspace')
        os.mkdir(path_join(WS_PATH, 'default'))
    p_loader = ProtocolLoader()
    protocols = p_loader.get_protocols()
    for protocol in protocols.keys():
        protocol_object = p_loader.load_protocol(protocols[protocol]['dbpath'])
        proto_db_path = path_join(WS_PATH, 'default', f'{protocol}.db')
        if not exists(proto_db_path):
            logger.debug(f'Initializing {protocol.upper()} protocol database')
            conn = connect(proto_db_path)
            c = conn.cursor()
            c.execute('PRAGMA journal_mode = OFF')
            c.execute('PRAGMA foreign_keys = 1')
            c.execute('PRAGMA busy_timeout = 5000')
            getattr(protocol_object, 'database').db_schema(c)
            conn.commit()
            conn.close()

def main():
    if False:
        return 10
    if not exists(CONFIG_PATH):
        print('[-] Unable to find config file')
        sys.exit(1)
    try:
        cmedbnav = CMEDBMenu(CONFIG_PATH)
        cmedbnav.cmdloop()
    except KeyboardInterrupt:
        pass
if __name__ == '__main__':
    main()