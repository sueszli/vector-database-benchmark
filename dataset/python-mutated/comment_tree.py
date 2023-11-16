from collections import defaultdict
from pylons import app_globals as g
from r2.lib.utils import SimpleSillyStub
from r2.lib.utils.comment_tree_utils import get_tree_details, calc_num_children
from r2.models.link import Comment
'Storage for comment trees\n\nCommentTree is a class that provides an interface to the actual storage.\nWhatever the underlying storage is, it must be able to generate the following\nstructures:\n* tree: dict of comment id -> list of child comment ids. The `None` entry is\n  top level comments\n* cids: list of all comment ids in the comment tree\n* depth: dict of comment id -> depth\n* parents: dict of comment id -> parent comment id\n* num_children: dict of comment id -> number of descendant comments, not just\n  direct children\n\nCommentTreePermacache uses permacache as the storage, and stores just the tree\nstructure. The cids, depth, parents and num_children are generated on the fly\nfrom the tree.\n\nAttempts were made to move to a different data model that would take advantage\nof the column based storage of Cassandra and eliminate the need for locking when\nadding a comment to the comment tree.\n\nCommentTreeStorageV2: for each comment, write a column where the column name is\n(parent_comment id, comment_id) and the column value is a counter giving the\nsize of the subtree rooted at the comment. This data model was abandoned because\ncounters ended up being unreliable and the shards put too much GC pressure on\nthe Cassandra JVM.\n\nCommentTreeStorageV3: for each comment, write a column where the column name is\n(depth, parent_comment_id, comment_id) and the column value is not used. This\ndata model was abandoned because of more unexpected GC problems after longer\ntime periods and generally insufficient regular-case performance.\n\n'

class CommentTreePermacache(object):

    @classmethod
    def _permacache_key(cls, link):
        if False:
            for i in range(10):
                print('nop')
        return 'comments_' + str(link._id)

    @classmethod
    def _mutation_context(cls, link):
        if False:
            for i in range(10):
                print('nop')
        'Return a lock for use during read-modify-write operations'
        key = 'comment_lock_' + str(link._id)
        return g.make_lock('comment_tree', key)

    @classmethod
    def prepare_new_storage(cls, link):
        if False:
            i = 10
            return i + 15
        'Write an empty tree to permacache'
        with cls._mutation_context(link) as lock:
            existing_tree = cls._load_tree(link)
            if not existing_tree:
                tree = {}
                cls._write_tree(link, tree, lock)

    @classmethod
    def _load_tree(cls, link):
        if False:
            print('Hello World!')
        key = cls._permacache_key(link)
        tree = g.permacache.get(key)
        return tree or {}

    @classmethod
    def _write_tree(cls, link, tree, lock):
        if False:
            return 10
        assert lock.have_lock
        key = cls._permacache_key(link)
        g.permacache.set(key, tree)

    @classmethod
    def get_tree_pieces(cls, link, timer):
        if False:
            while True:
                i = 10
        tree = cls._load_tree(link)
        timer.intermediate('load')
        (cids, depth, parents) = get_tree_details(tree)
        num_children = calc_num_children(tree)
        num_children = defaultdict(int, num_children)
        timer.intermediate('calculate')
        return (cids, tree, depth, parents, num_children)

    @classmethod
    def add_comments(cls, link, comments):
        if False:
            for i in range(10):
                print('nop')
        with cls._mutation_context(link) as lock:
            tree = cls._load_tree(link)
            (cids, _, _) = get_tree_details(tree)
            comments = {comment for comment in comments if comment._id not in cids}
            if not comments:
                return
            parent_ids = set(cids) | {comment._id for comment in comments}
            possible_orphan_comments = {comment for comment in comments if comment.parent_id and comment.parent_id not in parent_ids}
            if possible_orphan_comments:
                g.log.error('comment_tree_inconsistent: %s %s', link, possible_orphan_comments)
                g.stats.simple_event('comment_tree_inconsistent')
            for comment in comments:
                tree.setdefault(comment.parent_id, []).append(comment._id)
            cls._write_tree(link, tree, lock)

    @classmethod
    def rebuild(cls, link, comments):
        if False:
            return 10
        'Generate a tree from comments and overwrite any existing tree.'
        with cls._mutation_context(link) as lock:
            tree = {}
            for comment in comments:
                tree.setdefault(comment.parent_id, []).append(comment._id)
            cls._write_tree(link, tree, lock)

class CommentTree:

    def __init__(self, link, cids, tree, depth, parents, num_children):
        if False:
            i = 10
            return i + 15
        self.link = link
        self.cids = cids
        self.tree = tree
        self.depth = depth
        self.parents = parents
        self.num_children = num_children

    @classmethod
    def by_link(cls, link, timer=None):
        if False:
            print('Hello World!')
        if timer is None:
            timer = SimpleSillyStub()
        pieces = CommentTreePermacache.get_tree_pieces(link, timer)
        (cids, tree, depth, parents, num_children) = pieces
        comment_tree = cls(link, cids, tree, depth, parents, num_children)
        return comment_tree

    @classmethod
    def on_new_link(cls, link):
        if False:
            return 10
        CommentTreePermacache.prepare_new_storage(link)

    @classmethod
    def add_comments(cls, link, comments):
        if False:
            return 10
        CommentTreePermacache.add_comments(link, comments)

    @classmethod
    def rebuild(cls, link):
        if False:
            print('Hello World!')
        q = Comment._query(Comment.c.link_id == link._id, Comment.c._deleted == (True, False), Comment.c._spam == (True, False), optimize_rules=True)
        comments = list(q)
        comment_ids = {comment._id for comment in comments}
        comments = [comment for comment in comments if not comment.parent_id or comment.parent_id in comment_ids]
        CommentTreePermacache.rebuild(link, comments)
        link.num_comments = sum((1 for c in comments if not c._deleted))
        link._commit()