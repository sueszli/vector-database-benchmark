from pony.orm import Optional
from tribler.core.components.metadata_store.db.serialization import BINARY_NODE, BinaryNodePayload

def define_binding(db, db_version: int):
    if False:
        print('Hello World!')

    class BinaryNode(db.ChannelNode):
        """
        This ORM class represents channel descriptions.
        """
        _discriminator_ = BINARY_NODE
        if db_version >= 12:
            binary_data = Optional(bytes, default=b'')
            data_type = Optional(str, default='')
        _payload_class = BinaryNodePayload
        payload_arguments = _payload_class.__init__.__code__.co_varnames[:_payload_class.__init__.__code__.co_argcount][1:]
        nonpersonal_attributes = db.ChannelNode.nonpersonal_attributes + ('binary_data', 'data_type')
    return BinaryNode