from impacket.smb3structs import FILE_READ_DATA, FILE_WRITE_DATA

class RemoteFile:

    def __init__(self, smbConnection, fileName, share='ADMIN$', access=FILE_READ_DATA | FILE_WRITE_DATA):
        if False:
            return 10
        self.__smbConnection = smbConnection
        self.__share = share
        self.__access = access
        self.__fileName = fileName
        self.__tid = self.__smbConnection.connectTree(share)
        self.__fid = None
        self.__currentOffset = 0

    def open(self):
        if False:
            print('Hello World!')
        self.__fid = self.__smbConnection.openFile(self.__tid, self.__fileName, desiredAccess=self.__access)

    def seek(self, offset, whence):
        if False:
            i = 10
            return i + 15
        if whence == 0:
            self.__currentOffset = offset

    def read(self, bytesToRead):
        if False:
            while True:
                i = 10
        if bytesToRead > 0:
            data = self.__smbConnection.readFile(self.__tid, self.__fid, self.__currentOffset, bytesToRead)
            self.__currentOffset += len(data)
            return data
        return ''

    def close(self):
        if False:
            print('Hello World!')
        if self.__fid is not None:
            self.__smbConnection.closeFile(self.__tid, self.__fid)
            self.__fid = None

    def delete(self):
        if False:
            return 10
        self.__smbConnection.deleteFile(self.__share, self.__fileName)

    def tell(self):
        if False:
            while True:
                i = 10
        return self.__currentOffset

    def __str__(self):
        if False:
            return 10
        return f'\\\\{self.__smbConnection.getRemoteHost()}\\{self.__share}\\{self.__fileName}'