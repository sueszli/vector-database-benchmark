hash_alg_id = {'sm3': 1, 'sha1': 2, 'sha256': 4, 'sha512': 8}

class Digest:

    def __init__(self, session, alg_name='sm3'):
        if False:
            return 10
        if hash_alg_id[alg_name] is None:
            raise Exception('unsupported hash alg {}'.format(alg_name))
        self._alg_name = alg_name
        self._session = session
        self.__init_hash()

    def __init_hash(self):
        if False:
            return 10
        self._session.hash_init(hash_alg_id[self._alg_name])

    def update(self, data):
        if False:
            print('Hello World!')
        self._session.hash_update(data)

    def final(self):
        if False:
            for i in range(10):
                print('nop')
        return self._session.hash_final()

    def reset(self):
        if False:
            while True:
                i = 10
        self.__init_hash()

    def destroy(self):
        if False:
            print('Hello World!')
        self._session.close()