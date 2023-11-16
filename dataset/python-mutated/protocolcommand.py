from __future__ import absolute_import
from __future__ import print_function
import pprint
from buildbot_worker.base import ProtocolCommandBase

class FakeProtocolCommand(ProtocolCommandBase):
    debug = False

    def __init__(self, basedir):
        if False:
            i = 10
            return i + 15
        self.unicode_encoding = 'utf-8'
        self.updates = []
        self.worker_basedir = basedir
        self.basedir = basedir

    def show(self):
        if False:
            for i in range(10):
                print('nop')
        return pprint.pformat(self.updates)

    def send_update(self, status):
        if False:
            i = 10
            return i + 15
        if self.debug:
            print('FakeWorkerForBuilder.sendUpdate', status)
        for st in status:
            self.updates.append(st)

    def protocol_update_upload_file_close(self, writer):
        if False:
            for i in range(10):
                print('nop')
        return writer.callRemote('close')

    def protocol_update_upload_file_utime(self, writer, access_time, modified_time):
        if False:
            return 10
        return writer.callRemote('utime', (access_time, modified_time))

    def protocol_update_upload_file_write(self, writer, data):
        if False:
            return 10
        return writer.callRemote('write', data)

    def protocol_update_upload_directory(self, writer):
        if False:
            print('Hello World!')
        return writer.callRemote('unpack')

    def protocol_update_upload_directory_write(self, writer, data):
        if False:
            while True:
                i = 10
        return writer.callRemote('write', data)

    def protocol_update_read_file_close(self, reader):
        if False:
            print('Hello World!')
        return reader.callRemote('close')

    def protocol_update_read_file(self, reader, length):
        if False:
            i = 10
            return i + 15
        return reader.callRemote('read', length)