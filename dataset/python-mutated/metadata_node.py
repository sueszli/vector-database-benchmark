from pony import orm
from tribler.core.components.metadata_store.db.serialization import METADATA_NODE, MetadataNodePayload

def define_binding(db):
    if False:
        for i in range(10):
            print('nop')

    class MetadataNode(db.ChannelNode):
        """
        This ORM class extends ChannelNode by adding metadata-storing attributes such as "title" and "tags".
        It implements methods for indexed text search based on the "title" field.
        It is not intended for direct use. Instead, other classes should derive from it.
        """
        _discriminator_ = METADATA_NODE
        title = orm.Optional(str, default='')
        tags = orm.Optional(str, default='')
        num_entries = orm.Optional(int, size=64, default=0)
        _payload_class = MetadataNodePayload
        payload_arguments = _payload_class.__init__.__code__.co_varnames[:_payload_class.__init__.__code__.co_argcount][1:]
        nonpersonal_attributes = db.ChannelNode.nonpersonal_attributes + ('title', 'tags')

        def to_simple_dict(self):
            if False:
                i = 10
                return i + 15
            '\n            Return a basic dictionary with information about the channel.\n            '
            simple_dict = super().to_simple_dict()
            simple_dict.update({'name': self.title, 'category': self.tags})
            return simple_dict
    return MetadataNode