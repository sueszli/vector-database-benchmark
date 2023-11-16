from cme.protocols.smb.remotefile import RemoteFile
from impacket import nt_errors
from impacket.smb3structs import FILE_READ_DATA
from impacket.smbconnection import SessionError

class CMEModule:
    """
    Enumerate whether the WebClient service is running on the target by looking for the
    DAV RPC Service pipe. This technique was first suggested by Lee Christensen (@tifkin_)

    Module by Tobias Neitzel (@qtc_de)
    """
    name = 'webdav'
    description = 'Checks whether the WebClient service is running on the target'
    supported_protocols = ['smb']
    opsec_safe = True
    multiple_hosts = True

    def options(self, context, module_options):
        if False:
            while True:
                i = 10
        "\n        MSG     Info message when the WebClient service is running. '{}' is replaced by the target.\n        "
        self.output = 'WebClient Service enabled on: {}'
        if 'MSG' in module_options:
            self.output = module_options['MSG']

    def on_login(self, context, connection):
        if False:
            i = 10
            return i + 15
        "\n        Check whether the 'DAV RPC Service' pipe exists within the 'IPC$' share. This indicates\n        that the WebClient service is running on the target.\n        "
        try:
            remote_file = RemoteFile(connection.conn, 'DAV RPC Service', 'IPC$', access=FILE_READ_DATA)
            remote_file.open()
            remote_file.close()
            context.log.highlight(self.output.format(connection.conn.getRemoteHost()))
        except SessionError as e:
            if e.getErrorCode() == nt_errors.STATUS_OBJECT_NAME_NOT_FOUND:
                pass
            else:
                raise e