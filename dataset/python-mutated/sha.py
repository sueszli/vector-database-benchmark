__all__ = ('SHA1', 'SHA256', 'SHA384', 'SHA3_256', 'SHA3_512')
try:
    from Crypto.Hash import SHA1, SHA256, SHA384, SHA3_256, SHA3_512
except ImportError:
    SHA3_256 = None
    SHA3_512 = None
    from hashlib import sha1, sha256, sha384

    class SHA1(object):
        __slots__ = ()

        @staticmethod
        def new(*args):
            if False:
                i = 10
                return i + 15
            return sha1(*args)

    class SHA256(object):
        __slots__ = ()

        @staticmethod
        def new(*args):
            if False:
                i = 10
                return i + 15
            return sha256(*args)

    class SHA384(object):
        __slots__ = ()

        @staticmethod
        def new(*args):
            if False:
                print('Hello World!')
            return sha384(*args)