from bigdl.ppml.fl import *
from bigdl.dllib.utils.common import JavaValue

def set_psi_salt(psi_salt):
    if False:
        return 10
    callBigDlFunc('float', 'setPsiSalt', psi_salt)

class PSI(JavaValue):

    def __init__(self, jvalue=None, *args):
        if False:
            return 10
        self.bigdl_type = 'float'
        super().__init__(jvalue, self.bigdl_type, *args)

    def get_salt(self, secure_code=''):
        if False:
            while True:
                i = 10
        return callBigDlFunc(self.bigdl_type, 'psiGetSalt', self.value, secure_code)

    def upload_set(self, ids, salt):
        if False:
            for i in range(10):
                print('nop')
        callBigDlFunc(self.bigdl_type, 'psiUploadSet', self.value, ids, salt)

    def download_intersection(self, max_try=100, retry=3000):
        if False:
            i = 10
            return i + 15
        return callBigDlFunc(self.bigdl_type, 'psiDownloadIntersection', self.value, max_try, retry)

    def get_intersection(self, ids, max_try=100, retry=3000):
        if False:
            return 10
        return callBigDlFunc(self.bigdl_type, 'psiGetIntersection', self.value, ids, max_try, retry)