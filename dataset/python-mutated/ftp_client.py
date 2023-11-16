import ftplib
import io
from routersploit.core.exploit.exploit import Exploit
from routersploit.core.exploit.exploit import Protocol
from routersploit.core.exploit.option import OptBool
from routersploit.core.exploit.printer import print_error
from routersploit.core.exploit.printer import print_success
FTP_TIMEOUT = 8.0

class FTPCli(object):
    """ FTP Client provides methods to handle communication with FTP server """

    def __init__(self, ftp_target: str, ftp_port: int, ssl: bool=False, verbosity: bool=False) -> None:
        if False:
            for i in range(10):
                print('nop')
        ' FTP client constructor\n\n        :param str ftp_target: target FTP server ip address\n        :param int ftp_port: target FTP server port\n        :param bool ssl: target FTP ssl enabled\n        :param bool verbosity: display verbose output\n        :return None:\n        '
        self.ftp_target = ftp_target
        self.ftp_port = ftp_port
        self.verbosity = verbosity
        self.peer = '{}:{}'.format(self.ftp_target, ftp_port)
        if ssl:
            self.ftp_client = ftplib.FTP_TLS()
        else:
            self.ftp_client = ftplib.FTP()

    def connect(self, retries: int=1) -> bool:
        if False:
            while True:
                i = 10
        ' Connect to FTP server\n\n        :param int retries: number of retry attempts\n        :return bool: True if connection was successful, False otherwise\n        '
        for _ in range(retries):
            try:
                self.ftp_client.connect(self.ftp_target, self.ftp_port, timeout=FTP_TIMEOUT)
                return True
            except Exception as err:
                print_error(self.peer, 'FTP Error while connecting to the server', err, verbose=self.verbosity)
            self.ftp_client.close()
        return False

    def login(self, username: str, password: str) -> bool:
        if False:
            i = 10
            return i + 15
        ' Login to FTP server\n\n        :param str username: FTP account username\n        :param str password: FTP account password\n        :return bool: True if login was successful, False otherwise\n        '
        try:
            self.ftp_client.login(username, password)
            print_success(self.peer, "FTP Authentication Successful - Username: '{}' Password: '{}'".format(username, password), verbose=self.verbosity)
            return True
        except Exception:
            print_error(self.peer, "FTP Authentication Failed - Username: '{}' Password: '{}'".format(username, password), verbose=self.verbosity)
        self.ftp_client.close()
        return False

    def test_connect(self) -> bool:
        if False:
            return 10
        ' Test connection to FTP server\n\n        :return bool: True if connection was successful, False otherwise\n        '
        if self.connect():
            self.ftp_client.close()
            return True
        return False

    def get_content(self, remote_file: str) -> str:
        if False:
            while True:
                i = 10
        ' Get remote file from FTP server\n\n        :param str remote_file: remote file name\n        :return str: remote file content\n        '
        try:
            fp_content = io.BytesIO()
            self.ftp_client.retrbinary('RETR {}'.format(remote_file), fp_content.write)
            return fp_content.getvalue()
        except Exception as err:
            print_error(self.peer, 'FTP Error while retrieving content', err, verbose=self.verbosity)
        return None

    def close(self) -> bool:
        if False:
            print('Hello World!')
        ' Close FTP connection\n\n        :return bool: True if closing connection was successful, False otherwise\n        '
        try:
            self.ftp_client.close()
            return True
        except Exception as err:
            print_error(self.peer, 'FTP Error while closing connection', err, verbose=self.verbosity)
        return False

class FTPClient(Exploit):
    """ FTP Client exploit """
    target_protocol = Protocol.FTP
    ssl = OptBool(False, 'SSL enabled: true/false')
    verbosity = OptBool(True, 'Enable verbose output: true/false')

    def ftp_create(self, target: str=None, port: int=None) -> FTPCli:
        if False:
            i = 10
            return i + 15
        ' Create FTP client\n\n        :param str target: target FTP server ip address\n        :param int port: target FTP server port\n        :return FTPCli: FTP client object\n        '
        ftp_target = target if target else self.target
        ftp_port = port if port else self.port
        ftp_client = FTPCli(ftp_target, ftp_port, ssl=self.ssl, verbosity=self.verbosity)
        return ftp_client