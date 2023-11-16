import os
from pathlib import Path
from pony import orm
from pony.orm import db_session, select
from tribler.core.components.libtorrent.torrentdef import TorrentDef
from tribler.core.components.metadata_store.db.orm_bindings.channel_metadata import chunks
from tribler.core.components.metadata_store.db.orm_bindings.channel_node import CHANNEL_DESCRIPTION_FLAG, CHANNEL_THUMBNAIL_FLAG, COMMITTED, DIRTY_STATUSES, NEW, TODELETE, UPDATED
from tribler.core.components.metadata_store.db.orm_bindings.discrete_clock import clock
from tribler.core.components.metadata_store.db.orm_bindings.torrent_metadata import tdef_to_metadata_dict
from tribler.core.components.metadata_store.db.serialization import CHANNEL_TORRENT, COLLECTION_NODE, CollectionNodePayload
from tribler.core.utilities.simpledefs import CHANNEL_STATE
from tribler.core.utilities.utilities import random_infohash

def define_binding(db):
    if False:
        return 10

    class CollectionNode(db.MetadataNode):
        """
        This ORM class represents a generic named container, i.e. a folder. It is used as an intermediary node
        in building the nested channels tree.
        Methods for copying stuff recursively are bound to it.
        """
        _discriminator_ = COLLECTION_NODE
        _payload_class = CollectionNodePayload
        payload_arguments = _payload_class.__init__.__code__.co_varnames[:_payload_class.__init__.__code__.co_argcount][1:]
        nonpersonal_attributes = db.MetadataNode.nonpersonal_attributes + ('num_entries',)

        @property
        @db_session
        def state(self):
            if False:
                i = 10
                return i + 15
            if self.is_personal:
                return CHANNEL_STATE.PERSONAL.value
            toplevel_parent = self.get_parent_nodes()[0]
            if toplevel_parent.metadata_type == CHANNEL_TORRENT and toplevel_parent.local_version == toplevel_parent.timestamp:
                return CHANNEL_STATE.COMPLETE.value
            return CHANNEL_STATE.PREVIEW.value

        def to_simple_dict(self):
            if False:
                for i in range(10):
                    print('nop')
            result = super().to_simple_dict()
            result.update({'torrents': self.num_entries, 'state': self.state, 'description_flag': self.description_flag, 'thumbnail_flag': self.thumbnail_flag})
            return result

        def make_copy(self, tgt_parent_id, recursion_depth=15, **kwargs):
            if False:
                print('Hello World!')
            new_node = db.MetadataNode.make_copy(self, tgt_parent_id, **kwargs)
            if recursion_depth:
                for node in self.actual_contents:
                    if issubclass(type(node), CollectionNode):
                        node.make_copy(new_node.id_, recursion_depth=recursion_depth - 1)
                    else:
                        node.make_copy(new_node.id_)
            return new_node

        @db_session
        def copy_torrent_from_infohash(self, infohash):
            if False:
                for i in range(10):
                    print('nop')
            '\n            Search the database for a given infohash and create a copy of the matching entry in the current channel\n            :param infohash:\n            :return: New TorrentMetadata signed with your key.\n            '
            existing = db.TorrentMetadata.select(lambda g: g.infohash == infohash).first()
            if not existing:
                return None
            new_entry_dict = {'origin_id': self.id_, 'infohash': existing.infohash, 'title': existing.title, 'tags': existing.tags, 'size': existing.size, 'torrent_date': existing.torrent_date, 'tracker_info': existing.tracker_info, 'status': NEW}
            return db.TorrentMetadata.from_dict(new_entry_dict)

        @property
        def dirty(self):
            if False:
                return 10
            return self.contents.where(lambda g: g.status in DIRTY_STATUSES).exists()

        @property
        def contents(self):
            if False:
                while True:
                    i = 10
            return db.ChannelNode.select(lambda g: g.public_key == self.public_key and g.origin_id == self.id_ and (g != self))

        @property
        def actual_contents(self):
            if False:
                for i in range(10):
                    print('nop')
            return self.contents.where(lambda g: g.status != TODELETE)

        @property
        @db_session
        def contents_list(self):
            if False:
                while True:
                    i = 10
            return list(self.contents)

        @property
        def contents_len(self):
            if False:
                for i in range(10):
                    print('nop')
            return orm.count(self.contents)

        @property
        def thumbnail_flag(self):
            if False:
                return 10
            return bool(self.reserved_flags & CHANNEL_THUMBNAIL_FLAG)

        @property
        def description_flag(self):
            if False:
                while True:
                    i = 10
            return bool(self.reserved_flags & CHANNEL_DESCRIPTION_FLAG)

        @db_session
        def add_torrent_to_channel(self, tdef, extra_info=None):
            if False:
                while True:
                    i = 10
            '\n            Add a torrent to your channel.\n            :param tdef: The torrent definition file of the torrent to add\n            :param extra_info: Optional extra info to add to the torrent\n            '
            new_entry_dict = dict(tdef_to_metadata_dict(tdef), status=NEW)
            if extra_info:
                new_entry_dict['tags'] = extra_info.get('description', '')
            old_torrent = db.TorrentMetadata.get(public_key=self.public_key, infohash=tdef.get_infohash())
            torrent_metadata = old_torrent
            if old_torrent:
                if old_torrent.status == TODELETE:
                    new_timestamp = clock.tick()
                    old_torrent.set(timestamp=new_timestamp, origin_id=self.id_, **new_entry_dict)
                    old_torrent.sign()
                    old_torrent.status = UPDATED
            else:
                torrent_metadata = db.TorrentMetadata.from_dict(dict(origin_id=self.id_, **new_entry_dict))
            return torrent_metadata

        @db_session
        def pprint_tree(self, file=None, _prefix='', _last=True):
            if False:
                return 10
            print(_prefix, '`- ' if _last else '|- ', (self.num_entries, self.metadata_type), sep='', file=file)
            _prefix += '   ' if _last else '|  '
            child_count = self.actual_contents.count()
            for (i, child) in enumerate(list(self.actual_contents)):
                if issubclass(type(child), CollectionNode):
                    _last = i == child_count - 1
                    child.pprint_tree(file, _prefix, _last)
                else:
                    print(_prefix, '`- ' if _last else '|- ', child.metadata_type, sep='', file=file)

        @db_session
        def get_contents_recursive(self):
            if False:
                return 10
            results_stack = []
            for subnode in self.contents:
                if issubclass(type(subnode), CollectionNode):
                    results_stack.extend(subnode.get_contents_recursive())
                results_stack.append(subnode)
            return results_stack

        async def add_torrents_from_dir(self, torrents_dir, recursive=False):
            torrents_list = []
            errors_list = []

            def rec_gen(dir_):
                if False:
                    while True:
                        i = 10
                for (root, _, filenames) in os.walk(dir_):
                    for fn in filenames:
                        yield (Path(root) / fn)
            filename_generator = rec_gen(torrents_dir) if recursive else os.listdir(torrents_dir)
            torrents_list_generator = (Path(torrents_dir, f) for f in filename_generator)
            torrents_list = [f for f in torrents_list_generator if f.is_file() and f.suffix == '.torrent']
            torrent_defs = []
            for filename in torrents_list:
                try:
                    torrent_defs.append(await TorrentDef.load(filename))
                except Exception:
                    errors_list.append(filename)
            with db_session:
                for chunk in chunks(torrent_defs, 100):
                    for tdef in chunk:
                        self.add_torrent_to_channel(tdef)
                    orm.commit()
            return (torrents_list, errors_list)

        @staticmethod
        @db_session
        def commit_all_channels():
            if False:
                for i in range(10):
                    print('nop')
            committed_channels = []
            commit_queues_list = db.ChannelMetadata.get_commit_forest()
            for (_, queue) in commit_queues_list.items():
                channel = queue[-1]
                if len(queue) == 1:
                    if channel.status == TODELETE:
                        channel.delete()
                    else:
                        channel.status = COMMITTED
                    continue
                queue_prepared = db.ChannelMetadata.prepare_commit_queue_for_channel(queue)
                if isinstance(channel, db.ChannelMetadata):
                    committed_channels.append(channel.commit_channel_torrent(commit_list=queue_prepared))
                elif isinstance(channel, db.CollectionNode):
                    for g in queue:
                        if g.status in [NEW, UPDATED]:
                            g.status = COMMITTED
                        elif g.status == TODELETE:
                            g.delete()
            return committed_channels

        @staticmethod
        @db_session
        def get_children_dict_to_commit():
            if False:
                print('Hello World!')
            db.CollectionNode.collapse_deleted_subtrees()
            upd_dict = {}
            children = {}

            def update_node_info(n):
                if False:
                    print('Hello World!')
                if n.origin_id not in children:
                    children[n.origin_id] = {n}
                else:
                    children[n.origin_id].add(n)
                upd_dict[n.id_] = n
            dead_parents = set()
            for node in db.ChannelNode.select(lambda g: g.public_key == db.ChannelNode._my_key.pub().key_to_bin()[10:] and g.status in DIRTY_STATUSES):
                update_node_info(node)
                while node and node.origin_id not in upd_dict:
                    update_node_info(node)
                    parent = db.CollectionNode.get(public_key=node.public_key, id_=node.origin_id)
                    if not parent:
                        dead_parents.add(node.origin_id)
                    node = parent
            if 0 in dead_parents:
                dead_parents.remove(0)
            db.ChannelNode.select(lambda g: db.ChannelNode._my_key.pub().key_to_bin()[10:] == g.public_key and g.origin_id in dead_parents).delete()
            orm.flush()
            if not children or 0 not in children:
                return {}
            return children

        @staticmethod
        @db_session
        def get_commit_forest():
            if False:
                return 10
            children = db.CollectionNode.get_children_dict_to_commit()
            if not children:
                return {}
            forest = {}
            toplevel_nodes = children.pop(0)
            for root_node in toplevel_nodes:
                commit_queue = []
                tree_stack = [root_node]
                while tree_stack and children.get(tree_stack[-1].id_, None):
                    while children.get(tree_stack[-1].id_, None):
                        node = children[tree_stack[-1].id_].pop()
                        tree_stack.append(node)
                    while not issubclass(type(tree_stack[-1]), db.CollectionNode):
                        commit_queue.append(tree_stack.pop())
                    while tree_stack and (not children.get(tree_stack[-1].id_, None)):
                        while not issubclass(type(tree_stack[-1]), db.CollectionNode):
                            commit_queue.append(tree_stack.pop())
                        collection = tree_stack.pop()
                        commit_queue.append(collection)
                if not commit_queue or commit_queue[-1] != root_node:
                    commit_queue.append(root_node)
                forest[root_node.id_] = tuple(commit_queue)
            return forest

        @staticmethod
        def prepare_commit_queue_for_channel(commit_queue):
            if False:
                i = 10
                return i + 15
            "\n            This routine prepares the raw commit queue for commit by updating the elements' properties and\n            re-signing them. Also, it removes the channel entry itself from the queue [:-1], because its\n            meaningless to put it in the blobs, as it must be updated with the new infohash after commit.\n\n            :param commit_queue:\n            :return:\n            "
            for node in commit_queue:
                if issubclass(type(node), db.CollectionNode) and node.status != TODELETE:
                    node.num_entries = select((g.num_entries if g.metadata_type == COLLECTION_NODE else 1 for g in node.actual_contents)).sum()
                    node.timestamp = clock.tick()
                    node.sign()
            return sorted(commit_queue[:-1], key=lambda x: int(x.status == TODELETE) - 1 / x.timestamp)

        def delete(self, *args, **kwargs):
            if False:
                print('Hello World!')
            if kwargs.pop('recursive', True):
                for node in self.contents:
                    node.delete(*args, **kwargs)
            super().delete(*args, **kwargs)

        @staticmethod
        @db_session
        def collapse_deleted_subtrees():
            if False:
                return 10
            '\n            This procedure scans personal channels for collection nodes marked TODELETE and recursively removes\n            their contents. The top-level nodes themselves are left intact so soft delete entries can be generated\n            in the future.\n            This procedure should be always run _before_ committing personal channels.\n            '

            def get_highest_deleted_parent(node, highest_deleted_parent=None):
                if False:
                    print('Hello World!')
                if node.origin_id == 0:
                    return highest_deleted_parent
                parent = db.CollectionNode.get(public_key=node.public_key, id_=node.origin_id)
                if not parent:
                    return highest_deleted_parent
                if parent.status == TODELETE:
                    highest_deleted_parent = parent
                return get_highest_deleted_parent(parent, highest_deleted_parent)
            deletion_set = {get_highest_deleted_parent(node, highest_deleted_parent=node).rowid for node in db.CollectionNode.select(lambda g: g.public_key == db.CollectionNode._my_key.pub().key_to_bin()[10:] and g.status == TODELETE) if node}
            for node in [db.CollectionNode[rowid] for rowid in deletion_set]:
                for subnode in node.contents:
                    subnode.delete()

        @db_session
        def get_contents_to_commit(self):
            if False:
                while True:
                    i = 10
            return db.ChannelMetadata.prepare_commit_queue_for_channel(self.get_commit_forest().get(self.id_, []))

        def update_properties(self, update_dict):
            if False:
                return 10
            new_origin_id = update_dict.get('origin_id', self.origin_id)
            if new_origin_id not in (0, self.origin_id):
                new_parent = CollectionNode.get(public_key=self.public_key, id_=new_origin_id)
                if not new_parent:
                    raise ValueError('Target collection does not exists')
                root_path = new_parent.get_parent_nodes()
                if new_origin_id == self.id_ or self in root_path[:-1]:
                    raise ValueError("Can't move collection into itself or its descendants!")
                if root_path[0].origin_id != 0:
                    raise ValueError('Tried to move collection into an orphaned hierarchy!')
            updated_self = super().update_properties(update_dict)
            if updated_self.origin_id == 0 and self.metadata_type == COLLECTION_NODE:
                self_dict = updated_self.to_dict()
                updated_self.delete(recursive=False)
                self_dict.pop('rowid')
                self_dict.pop('metadata_type')
                self_dict.pop('timestamp')
                self_dict['infohash'] = random_infohash()
                self_dict['sign_with'] = self._my_key
                updated_self = db.ChannelMetadata.from_dict(self_dict)
            return updated_self
    return CollectionNode