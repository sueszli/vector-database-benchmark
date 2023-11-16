from array import array

class KeyManager:

    def __init__(self):
        if False:
            i = 10
            return i + 15
        self.keys = {}

    def __get_bssid_hasheable_type(self, bssid):
        if False:
            return 10
        if not isinstance(bssid, (list, tuple, array)):
            raise Exception('BSSID datatype must be a tuple, list or array')
        return tuple(bssid)

    def add_key(self, bssid, key):
        if False:
            print('Hello World!')
        bssid = self.__get_bssid_hasheable_type(bssid)
        if bssid not in self.keys:
            self.keys[bssid] = key
            return True
        else:
            return False

    def replace_key(self, bssid, key):
        if False:
            print('Hello World!')
        bssid = self.__get_bssid_hasheable_type(bssid)
        self.keys[bssid] = key
        return True

    def get_key(self, bssid):
        if False:
            while True:
                i = 10
        bssid = self.__get_bssid_hasheable_type(bssid)
        if bssid in self.keys:
            return self.keys[bssid]
        else:
            return False

    def delete_key(self, bssid):
        if False:
            for i in range(10):
                print('nop')
        bssid = self.__get_bssid_hasheable_type(bssid)
        if not isinstance(bssid, list):
            raise Exception('BSSID datatype must be a list')
        if bssid in self.keys:
            del self.keys[bssid]
            return True
        return False