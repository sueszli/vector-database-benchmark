from bigdl.ppml.fl import *

class FLServer(JavaValue):

    def __init__(self, jvalue=None, *args):
        if False:
            print('Hello World!')
        self.bigdl_type = 'float'
        super().__init__(jvalue, self.bigdl_type, *args)

    def build(self):
        if False:
            print('Hello World!')
        callBigDlFunc(self.bigdl_type, 'flServerBuild', self.value)

    def start(self):
        if False:
            i = 10
            return i + 15
        callBigDlFunc(self.bigdl_type, 'flServerStart', self.value)

    def stop(self):
        if False:
            while True:
                i = 10
        callBigDlFunc(self.bigdl_type, 'flServerStop', self.value)

    def set_client_num(self, client_num):
        if False:
            i = 10
            return i + 15
        callBigDlFunc(self.bigdl_type, 'flServerSetClientNum', self.value, client_num)

    def set_port(self, port):
        if False:
            return 10
        callBigDlFunc(self.bigdl_type, 'flServerSetPort', self.value, port)

    def block_until_shutdown(self):
        if False:
            for i in range(10):
                print('nop')
        callBigDlFunc(self.bigdl_type, 'flServerBlockUntilShutdown', self.value)
if __name__ == '__main__':
    fl_server = FLServer()
    fl_server.build()
    fl_server.start()
    fl_server.block_until_shutdown()