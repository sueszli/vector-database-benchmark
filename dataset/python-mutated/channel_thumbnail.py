from tribler.core.components.metadata_store.db.serialization import CHANNEL_THUMBNAIL

def define_binding(db):
    if False:
        return 10

    class ChannelThumbnail(db.BinaryNode):
        """
        This ORM class represents channel descriptions.
        """
        _discriminator_ = CHANNEL_THUMBNAIL
    return ChannelThumbnail