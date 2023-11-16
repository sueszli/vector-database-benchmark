import pickle
import hmac
import hashlib
from bigdl.ppml.utils.log4Error import invalidInputError

class SafePickle:
    key = b'shared-key'
    "\n    Example:\n        >>> from bigdl.ppml.utils.safepickle import SafePickle\n        >>> with open(file_path, 'wb') as file:\n        >>>     signature = SafePickle.dump(data, file, return_digest=True)\n        >>> with open(file_path, 'rb') as file:\n        >>>     data = SafePickle.load(file, signature)\n    "

    @classmethod
    def dump(self, obj, file, return_digest=False, *args, **kwargs):
        if False:
            while True:
                i = 10
        if return_digest:
            pickled_data = pickle.dumps(obj)
            file.write(pickled_data)
            digest = hmac.new(self.key, pickled_data, hashlib.sha1).hexdigest()
            return digest
        else:
            pickle.dump(obj, file, *args, **kwargs)

    @classmethod
    def load(self, file, digest=None, *args, **kwargs):
        if False:
            print('Hello World!')
        if digest:
            content = file.read()
            new_digest = hmac.new(self.key, content, hashlib.sha1).hexdigest()
            if digest != new_digest:
                invalidInputError(False, 'Pickle safe check failed')
            file.seek(0)
        return pickle.load(file, *args, **kwargs)

    @classmethod
    def dumps(self, obj, *args, **kwargs):
        if False:
            print('Hello World!')
        return pickle.dumps(obj, *args, **kwargs)

    @classmethod
    def loads(self, data, *args, **kwargs):
        if False:
            while True:
                i = 10
        return pickle.loads(data, *args, **kwargs)