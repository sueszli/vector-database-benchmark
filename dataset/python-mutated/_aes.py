class AES:

    def __init__(self, backend, fallback=None):
        if False:
            for i in range(10):
                print('nop')
        self._backend = backend
        self._fallback = fallback

    def get_algo_key_length(self, algo):
        if False:
            return 10
        if algo.count('-') != 2:
            raise ValueError('Invalid algorithm name')
        try:
            return int(algo.split('-')[1]) // 8
        except ValueError:
            raise ValueError('Invalid algorithm name') from None

    def new_key(self, algo='aes-256-cbc'):
        if False:
            while True:
                i = 10
        if not self._backend.is_algo_supported(algo):
            if self._fallback is None:
                raise ValueError('This algorithm is not supported')
            return self._fallback.new_key(algo)
        return self._backend.random(self.get_algo_key_length(algo))

    def encrypt(self, data, key, algo='aes-256-cbc'):
        if False:
            return 10
        if not self._backend.is_algo_supported(algo):
            if self._fallback is None:
                raise ValueError('This algorithm is not supported')
            return self._fallback.encrypt(data, key, algo)
        key_length = self.get_algo_key_length(algo)
        if len(key) != key_length:
            raise ValueError('Expected key to be {} bytes, got {} bytes'.format(key_length, len(key)))
        return self._backend.encrypt(data, key, algo)

    def decrypt(self, ciphertext, iv, key, algo='aes-256-cbc'):
        if False:
            print('Hello World!')
        if not self._backend.is_algo_supported(algo):
            if self._fallback is None:
                raise ValueError('This algorithm is not supported')
            return self._fallback.decrypt(ciphertext, iv, key, algo)
        key_length = self.get_algo_key_length(algo)
        if len(key) != key_length:
            raise ValueError('Expected key to be {} bytes, got {} bytes'.format(key_length, len(key)))
        return self._backend.decrypt(ciphertext, iv, key, algo)

    def get_backend(self):
        if False:
            return 10
        return self._backend.get_backend()