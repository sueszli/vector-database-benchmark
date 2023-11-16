import random
from datetime import datetime
from ipv8.keyvault.crypto import default_eccrypto
from pony import orm
from pony.orm.core import DEFAULT, db_session
from tribler.core.components.metadata_store.db.orm_bindings.discrete_clock import clock
from tribler.core.components.metadata_store.db.serialization import CHANNEL_NODE, ChannelNodePayload, DELETED, DeletedMetadataPayload
from tribler.core.exceptions import InvalidChannelNodeException, InvalidSignatureException
from tribler.core.utilities.path_util import Path
from tribler.core.utilities.unicode import hexlify
NEW = 0
TODELETE = 1
COMMITTED = 2
UPDATED = 6
LEGACY_ENTRY = 1000
DIRTY_STATUSES = (NEW, TODELETE, UPDATED)
PUBLIC_KEY_LEN = 64
CHANNEL_DESCRIPTION_FLAG = 1
CHANNEL_THUMBNAIL_FLAG = 2

def generate_dict_from_pony_args(cls, skip_list=None, **kwargs):
    if False:
        while True:
            i = 10
    '\n    Note: this is a way to manually define Pony entity default attributes in case we\n    have to generate the signature before creating an object\n    '
    d = {}
    skip_list = skip_list or []
    for attr in cls._attrs_:
        val = kwargs.get(attr.name, DEFAULT)
        if attr.name in skip_list:
            continue
        d[attr.name] = attr.validate(val, entity=cls)
    return d

def define_binding(db, logger=None, key=None):
    if False:
        while True:
            i = 10

    class ChannelNode(db.Entity):
        """
        This is the base class of our ORM bindings. It implements methods for signing and serialization of ORM objects.
        All other GigaChannel-related ORM classes are derived from it. It is not intended for direct use.
        Instead, other classes should derive from it.
        """
        _discriminator_ = CHANNEL_NODE
        rowid = orm.PrimaryKey(int, size=64, auto=True)
        metadata_type = orm.Discriminator(int, size=16)
        reserved_flags = orm.Optional(int, size=16, default=0)
        origin_id = orm.Optional(int, size=64, default=0, index=True)
        public_key = orm.Required(bytes)
        id_ = orm.Required(int, size=64)
        orm.composite_key(public_key, id_)
        orm.composite_index(public_key, origin_id)
        timestamp = orm.Required(int, size=64, default=0)
        signature = orm.Optional(bytes, unique=True, nullable=True, default=None)
        added_on = orm.Optional(datetime, default=datetime.utcnow)
        status = orm.Optional(int, default=COMMITTED)
        _payload_class = ChannelNodePayload
        _my_key = key
        _logger = logger
        payload_arguments = _payload_class.__init__.__code__.co_varnames[:_payload_class.__init__.__code__.co_argcount][1:]
        nonpersonal_attributes = ('metadata_type',)

        def __init__(self, *args, **kwargs):
            if False:
                i = 10
                return i + 15
            '\n            Initialize a metadata object.\n            All this dance is required to ensure that the signature is there and it is correct.\n            '
            skip_key_check = False
            private_key_override = None
            if 'sign_with' in kwargs:
                kwargs['public_key'] = kwargs['sign_with'].pub().key_to_bin()[10:]
                private_key_override = kwargs.pop('sign_with')
            if 'public_key' in kwargs and kwargs['public_key'] == b'':
                if 'id_' in kwargs:
                    kwargs['signature'] = None
                    skip_key_check = True
                else:
                    raise InvalidChannelNodeException('Attempted to create %s free-for-all (unsigned) object without specifying id_ : ' % str(self.__class__.__name__))
            skip_key_check = kwargs.pop('skip_key_check', skip_key_check)
            if 'timestamp' not in kwargs:
                kwargs['timestamp'] = clock.tick()
            if 'id_' not in kwargs:
                kwargs['id_'] = int(random.getrandbits(63))
            if not private_key_override and (not skip_key_check):
                if 'signature' not in kwargs and ('public_key' not in kwargs or kwargs['public_key'] == self._my_key.pub().key_to_bin()[10:]):
                    private_key_override = self._my_key
                elif 'public_key' in kwargs and 'signature' in kwargs:
                    try:
                        self._payload_class(**kwargs)
                    except InvalidSignatureException as e:
                        raise InvalidSignatureException(f'Attempted to create {str(self.__class__.__name__)} object with invalid signature/PK: ' + (hexlify(kwargs['signature']) if 'signature' in kwargs else 'empty signature ') + ' / ' + (hexlify(kwargs['public_key']) if 'public_key' in kwargs else ' empty PK')) from e
            if private_key_override:
                kwargs = generate_dict_from_pony_args(self.__class__, skip_list=['signature', 'public_key'], **kwargs)
                payload = self._payload_class(**dict(kwargs, public_key=private_key_override.pub().key_to_bin()[10:], key=private_key_override, metadata_type=self.metadata_type))
                kwargs['public_key'] = payload.public_key
                kwargs['signature'] = payload.signature
            super().__init__(*args, **kwargs)

        def _serialized(self, key=None):
            if False:
                while True:
                    i = 10
            '\n            Serializes the object and returns the result with added signature (tuple output)\n            :param key: private key to sign object with\n            :return: (serialized_data, signature) tuple\n            '
            return self._payload_class(key=key, unsigned=self.signature is None, **self.to_dict())._serialized()

        def serialized(self, key=None):
            if False:
                while True:
                    i = 10
            '\n            Serializes the object and returns the result with added signature (blob output)\n            :param key: private key to sign object with\n            :return: serialized_data+signature binary string\n            '
            return b''.join(self._serialized(key))

        def _serialized_delete(self):
            if False:
                for i in range(10):
                    print('nop')
            '\n            Create a special command to delete this metadata and encode it for transfer (tuple output).\n            :return: (serialized_data, signature) tuple\n            '
            my_dict = ChannelNode.to_dict(self)
            my_dict.update({'metadata_type': DELETED, 'delete_signature': self.signature})
            return DeletedMetadataPayload(key=self._my_key, **my_dict)._serialized()

        def serialized_delete(self):
            if False:
                for i in range(10):
                    print('nop')
            '\n            Create a special command to delete this metadata and encode it for transfer (blob output).\n            :return: serialized_data+signature binary string\n            '
            return b''.join(self._serialized_delete())

        def serialized_health(self) -> bytes:
            if False:
                i = 10
                return i + 15
            return b';'

        def to_file(self, filename, key=None):
            if False:
                while True:
                    i = 10
            with open(Path.fix_win_long_file(filename), 'wb') as output_file:
                output_file.write(self.serialized(key))

        def to_delete_file(self, filename):
            if False:
                print('Hello World!')
            with open(Path.fix_win_long_file(filename), 'wb') as output_file:
                output_file.write(self.serialized_delete())

        def sign(self, key=None):
            if False:
                i = 10
                return i + 15
            if not key:
                key = self._my_key
            self.public_key = key.pub().key_to_bin()[10:]
            (_, self.signature) = self._serialized(key)

        def has_valid_signature(self):
            if False:
                while True:
                    i = 10
            crypto = default_eccrypto
            signature_correct = False
            key_correct = crypto.is_valid_public_bin(b'LibNaCLPK:' + bytes(self.public_key))
            if key_correct:
                try:
                    self._payload_class(**self.to_dict())
                except InvalidSignatureException:
                    signature_correct = False
                else:
                    signature_correct = True
            return key_correct and signature_correct

        @classmethod
        def from_payload(cls, payload):
            if False:
                for i in range(10):
                    print('nop')
            return cls(**payload.to_dict())

        @classmethod
        def from_dict(cls, dct):
            if False:
                i = 10
                return i + 15
            return cls(**dct)

        @property
        @db_session
        def is_personal(self):
            if False:
                while True:
                    i = 10
            return self._my_key.pub().key_to_bin()[10:] == self.public_key

        @db_session
        def soft_delete(self):
            if False:
                for i in range(10):
                    print('nop')
            if self.status == NEW:
                self.delete()
            else:
                self.status = TODELETE

        def update_properties(self, update_dict):
            if False:
                print('Hello World!')
            signed_attribute_changed = False
            for (k, value) in update_dict.items():
                if getattr(self, k) != value:
                    setattr(self, k, value)
                    signed_attribute_changed = signed_attribute_changed or k in self.payload_arguments
            if signed_attribute_changed:
                if self.status != NEW:
                    self.status = UPDATED
                self.timestamp = clock.tick()
                self.sign()
            return self

        def get_parent_nodes(self):
            if False:
                i = 10
                return i + 15
            full_path = {self: True}
            node = self
            while node:
                node = db.CollectionNode.get(public_key=self.public_key, id_=node.origin_id)
                if node is None:
                    break
                if node in full_path:
                    break
                full_path[node] = True
                if node.origin_id == 0:
                    break
            return tuple(reversed(list(full_path)))

        def make_copy(self, tgt_parent_id, attributes_override=None):
            if False:
                print('Hello World!')
            dst_dict = attributes_override or {}
            for k in self.nonpersonal_attributes:
                dst_dict[k] = getattr(self, k)
            dst_dict.update({'origin_id': tgt_parent_id, 'status': NEW})
            return self.__class__(**dst_dict)

        def get_type(self) -> int:
            if False:
                for i in range(10):
                    print('nop')
            return self._discriminator_

        def to_simple_dict(self):
            if False:
                i = 10
                return i + 15
            '\n            Return a basic dictionary with information about the node\n            '
            simple_dict = {'type': self.get_type(), 'id': self.id_, 'origin_id': self.origin_id, 'public_key': hexlify(self.public_key), 'status': self.status}
            return simple_dict
    return ChannelNode