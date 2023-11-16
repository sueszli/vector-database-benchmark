import enum
from dataclasses import dataclass, field
from pony.orm import db_session
from tribler.core.components.metadata_store.category_filter.l2_filter import is_forbidden
from tribler.core.components.metadata_store.db.serialization import CHANNEL_DESCRIPTION, CHANNEL_THUMBNAIL, CHANNEL_TORRENT, COLLECTION_NODE, DELETED, NULL_KEY, REGULAR_TORRENT
from tribler.core.utilities.sentinels import sentinel
from tribler.core.utilities.unicode import hexlify

class ObjState(enum.Enum):
    UPDATED_LOCAL_VERSION = enum.auto()
    LOCAL_VERSION_NEWER = enum.auto()
    LOCAL_VERSION_SAME = enum.auto()
    NEW_OBJECT = enum.auto()
CONTINUE = sentinel('CONTINUE')

@dataclass
class ProcessingResult:
    md_obj: object = None
    obj_state: object = None
    missing_deps: list = field(default_factory=list)

class PayloadChecker:

    def __init__(self, mds, payload, skip_personal_metadata_payload=True, channel_public_key=None):
        if False:
            print('Hello World!')
        self.mds = mds
        self.payload = payload
        self.skip_personal_metadata_payload = skip_personal_metadata_payload
        self.channel_public_key = channel_public_key
        self._logger = self.mds._logger

    def reject_payload_with_nonmatching_public_key(self, channel_public_key):
        if False:
            while True:
                i = 10
        '\n        This check rejects payloads that do not match the given public key. It is used during authoritative\n        updates of channels from disk (serialized and downloaded in the torrent form) to prevent\n        channel creators from injecting random garbage into local database.\n        '
        if self.payload.public_key != channel_public_key:
            self._logger.warning('Tried to push metadata entry with foreign public key.             Expected public key: %s, entry public key / id: %s / %i', hexlify(channel_public_key), self.payload.public_key, self.payload.id_)
            return []
        return CONTINUE

    def process_delete_node_command(self):
        if False:
            while True:
                i = 10
        '\n        Check if the payload is a command to delete an existing node. If it is, delete the node\n        and return empty list. Otherwise, CONTINUE control to further checks.\n        '
        if self.payload.metadata_type == DELETED:
            node = self.mds.ChannelNode.get_for_update(signature=self.payload.delete_signature, public_key=self.payload.public_key)
            if node:
                node.delete()
                return []
        return CONTINUE

    def reject_unknown_payload_type(self):
        if False:
            print('Hello World!')
        '\n        Check if the payload contains metadata of a known type.\n        If it does not, stop processing and return empty list.\n        Otherwise, CONTINUE control to further checks.\n        '
        if self.payload.metadata_type not in [CHANNEL_TORRENT, REGULAR_TORRENT, COLLECTION_NODE, CHANNEL_DESCRIPTION, CHANNEL_THUMBNAIL]:
            return []
        return CONTINUE

    def reject_payload_with_offending_words(self):
        if False:
            print('Hello World!')
        '\n        Check if the payload contains strong offending words.\n        If it does, stop processing and return empty list.\n        Otherwise, CONTINUE control to further checks.\n        '
        if is_forbidden(' '.join((getattr(self.payload, attr) for attr in ('title', 'tags', 'text') if hasattr(self.payload, attr)))):
            return []
        return CONTINUE

    def add_ffa_node(self):
        if False:
            return 10
        '\n        Check if the payload contains metadata of Free-For-All (FFA) type, which is just a REGULAR_TORRENT payload\n        without signature. If it does, create a corresponding node in the local database.\n        Otherwise, CONTINUE control to further checks.\n        '
        if self.payload.public_key == NULL_KEY:
            if self.payload.metadata_type == REGULAR_TORRENT:
                node = self.mds.TorrentMetadata.add_ffa_from_dict(self.payload.to_dict())
                if node:
                    return [ProcessingResult(md_obj=node, obj_state=ObjState.NEW_OBJECT)]
            return []
        return CONTINUE

    def add_node(self):
        if False:
            i = 10
            return i + 15
        '\n        Try to create a local node from the payload.\n        If it is impossible, CONTINUE control to further checks (there should not be any more, really).\n        '
        for orm_class in (self.mds.TorrentMetadata, self.mds.ChannelMetadata, self.mds.CollectionNode, self.mds.ChannelThumbnail, self.mds.ChannelDescription):
            if orm_class._discriminator_ == self.payload.metadata_type:
                obj = orm_class.from_payload(self.payload)
                return [ProcessingResult(md_obj=obj, obj_state=ObjState.NEW_OBJECT)]
        return CONTINUE

    def reject_personal_metadata(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Check if the payload contains metadata signed by our private key. This could happen in a situation where\n        someone else tries to push us our old channel data, for example.\n        Since we are the only authoritative source of information about our own channel, we reject\n        such payloads and thus return empty list.\n        Otherwise, CONTINUE control to further checks.\n        '
        if self.payload.public_key == self.mds.my_public_key_bin:
            return []
        return CONTINUE

    def reject_obsolete_metadata(self):
        if False:
            print('Hello World!')
        '\n        Check if the received payload contains older deleted metadata for a channel we are subscribed to.\n        In that case, we reject the metadata and return an empty list.\n        Otherwise, CONTINUE control to further checks.\n        '
        parent = self.mds.CollectionNode.get(public_key=self.payload.public_key, id_=self.payload.origin_id)
        if parent is None:
            return CONTINUE
        parent = parent.get_parent_nodes()[0] if parent.metadata_type != CHANNEL_TORRENT else parent
        if parent.metadata_type == CHANNEL_TORRENT and self.payload.timestamp <= parent.local_version:
            return []
        return CONTINUE

    def update_local_node(self):
        if False:
            for i in range(10):
                print('nop')
        "\n        Check if the received payload contains an updated version of metadata node we already have\n        in the local database (e.g. a newer version of channel entry gossiped to us).\n        We try to update the local metadata node in that case, returning UPDATED_LOCAL_VERSION status.\n        Conversely, if we got a newer version of the metadata node, we return it to higher level\n        with a LOCAL_VERSION_NEWER mark, so the higher level can possibly push an update back to the sender.\n        If we don't have some version of the node locally, CONTINUE control to further checks.\n        "
        node = self.mds.ChannelNode.get_for_update(public_key=self.payload.public_key, id_=self.payload.id_)
        if not node:
            return CONTINUE
        node.to_simple_dict()
        if node.timestamp == self.payload.timestamp:
            return [ProcessingResult(md_obj=node, obj_state=ObjState.LOCAL_VERSION_SAME)]
        if node.timestamp > self.payload.timestamp:
            return [ProcessingResult(md_obj=node, obj_state=ObjState.LOCAL_VERSION_NEWER)]
        if node.timestamp < self.payload.timestamp:
            return self.update_channel_node(node)
        return CONTINUE

    def update_channel_node(self, node):
        if False:
            while True:
                i = 10
        if node.metadata_type == self.payload.metadata_type:
            node.set(**self.payload.to_dict())
            return [ProcessingResult(md_obj=node, obj_state=ObjState.UPDATED_LOCAL_VERSION)]
        for orm_class in (self.mds.ChannelMetadata, self.mds.CollectionNode):
            if orm_class._discriminator_ == self.payload.metadata_type:
                node.delete()
                obj = orm_class.from_payload(self.payload)
                return [ProcessingResult(md_obj=obj, obj_state=ObjState.UPDATED_LOCAL_VERSION)]
        self._logger.warning(f'Tried to update channel node to illegal type:  original type: {node.metadata_type} updated type: {self.payload.metadata_type} {hexlify(self.payload.public_key)}, {self.payload.id_} ')
        return []

    def request_missing_dependencies(self, node_list):
        if False:
            i = 10
            return i + 15
        '\n        Scan the results for entries with locally missing dependencies, such as thumbnail and description nodes,\n        and modify the results by adding a dict with request for missing nodes in the get_entries format.\n        '
        for r in node_list:
            updated_local_channel_node = r.obj_state == ObjState.UPDATED_LOCAL_VERSION and r.md_obj.metadata_type == CHANNEL_TORRENT
            r.missing_deps.extend(self.requests_for_child_dependencies(r.md_obj, include_newer=updated_local_channel_node))
        return node_list

    def perform_checks(self):
        if False:
            print('Hello World!')
        '\n        This method runs checks on the received payload. Essentially, it acts like a firewall, rejecting\n        incorrect or conflicting entries. Individual checks can return either CONTINUE, an empty list or a list\n        of ProcessingResult objects. If CONTINUE sentinel object is returned, checks will proceed further.\n        If non-CONTINUE result is returned by a check, the checking process stops.\n        '
        if self.channel_public_key:
            yield self.reject_payload_with_nonmatching_public_key(self.channel_public_key)
        if self.skip_personal_metadata_payload:
            yield self.reject_personal_metadata()
        if self.channel_public_key:
            yield self.process_delete_node_command()
        yield self.reject_unknown_payload_type()
        yield self.reject_payload_with_offending_words()
        yield self.reject_obsolete_metadata()
        yield self.add_ffa_node()
        yield self.update_local_node()
        yield self.add_node()
        self._logger.warning(f'Payload processing ended without actions, this should not happen normally. Payload type: {self.payload.metadata_type} {hexlify(self.payload.public_key)}, {self.payload.id_}  {self.payload.timestamp}')
        yield []

    def requests_for_child_dependencies(self, node, include_newer=False):
        if False:
            print('Hello World!')
        '\n        This method checks the given ORM node (object) for missing dependencies, such as thumbnails and/or\n        descriptions. To do so, it checks for existence of special dependency flags in the object\'s\n        "reserved_flags" field and checks for existence of the corresponding dependencies in the local database.\n        '
        if node.metadata_type not in (CHANNEL_TORRENT, COLLECTION_NODE):
            return []
        result = []
        if node.description_flag:
            result.extend(self.check_and_request_child_dependency(node, CHANNEL_DESCRIPTION, include_newer))
        if node.thumbnail_flag:
            result.extend(self.check_and_request_child_dependency(node, CHANNEL_THUMBNAIL, include_newer))
        return result

    def check_and_request_child_dependency(self, node, dep_type, include_newer=False):
        if False:
            print('Hello World!')
        '\n        For each missing dependency it will generate a query in the "get_entry" format that should be addressed to the\n        peer that sent the original payload/node/object.\n        If include_newer argument is true, it will generate a query even if the dependencies exist in the local\n        database. However, this query will limit the selection to dependencies with a higher timestamp than that\n        of the local versions. Effectively, this query asks the remote peer for updates on dependencies. Thus,\n        it should only be issued when it is known that the parent object was updated.\n        '
        dep_node = self.mds.ChannelNode.select(lambda g: g.origin_id == node.id_ and g.public_key == node.public_key and (g.metadata_type == dep_type)).first()
        request_dict = {'metadata_type': [dep_type], 'channel_pk': node.public_key, 'origin_id': node.id_, 'first': 0, 'last': 1}
        if not dep_node:
            return [request_dict]
        if include_newer:
            request_dict['attribute_ranges'] = (('timestamp', dep_node.timestamp + 1, None),)
            return [request_dict]
        return []

    @db_session
    def process_payload(self):
        if False:
            print('Hello World!')
        result = []
        for result in self.perform_checks():
            if result is not CONTINUE:
                break
        if self.channel_public_key is None:
            result = self.request_missing_dependencies(result)
        return result

def process_payload(metadata_store, payload, skip_personal_metadata_payload=True, channel_public_key=None):
    if False:
        print('Hello World!')
    '\n    This routine decides what to do with a given payload and executes the necessary actions.\n    To do so, it looks into the database, compares version numbers, etc.\n    It returns a list of tuples each of which contain the corresponding new/old object and the actions\n    that were performed on that object.\n    :param metadata_store: Metadata Store object serving the database\n    :param payload: payload to work on\n    :param skip_personal_metadata_payload: if this is set to True, personal torrent metadata payload received\n            through gossip will be ignored. The default value is True.\n    :param channel_public_key: rejects payloads that do not belong to this key.\n           Enabling this allows to skip some costly checks during e.g. channel processing.\n\n    :return: a list of ProcessingResult objects\n    '
    return PayloadChecker(metadata_store, payload, skip_personal_metadata_payload=skip_personal_metadata_payload, channel_public_key=channel_public_key).process_payload()