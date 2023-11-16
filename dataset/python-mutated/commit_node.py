from datetime import datetime
from deeplake.client.utils import get_user_name
from typing import List, Optional

class CommitNode:
    """Contains all the Version Control information about a particular commit."""

    def __init__(self, branch: str, commit_id: str, total_samples_processed: int=0):
        if False:
            i = 10
            return i + 15
        self.commit_id = commit_id
        self.branch = branch
        self.children: List['CommitNode'] = []
        self.parent: Optional['CommitNode'] = None
        self.commit_message: Optional[str] = None
        self.commit_time: Optional[datetime] = None
        self.commit_user_name: Optional[str] = None
        self.merge_parent: Optional['CommitNode'] = None
        self._info_updated: bool = False
        self.is_checkpoint: bool = False
        self.total_samples_processed: int = total_samples_processed

    def add_child(self, node: 'CommitNode'):
        if False:
            for i in range(10):
                print('nop')
        'Adds a child to the node, used for branching.'
        node.parent = self
        self.children.append(node)

    def copy(self):
        if False:
            while True:
                i = 10
        node = CommitNode(self.branch, self.commit_id)
        node.commit_message = self.commit_message
        node.commit_user_name = self.commit_user_name
        node.commit_time = self.commit_time
        node.is_checkpoint = self.is_checkpoint
        node.total_samples_processed = self.total_samples_processed
        return node

    def add_successor(self, node: 'CommitNode', message: Optional[str]=None):
        if False:
            for i in range(10):
                print('nop')
        'Adds a successor (a type of child) to the node, used for commits.'
        node.parent = self
        self.children.append(node)
        self.commit_message = message
        self.commit_user_name = get_user_name()
        self.commit_time = datetime.utcnow()

    def merge_from(self, node: 'CommitNode'):
        if False:
            while True:
                i = 10
        'Merges the given node into this node.'
        self.merge_parent = node

    @property
    def is_merge_node(self):
        if False:
            for i in range(10):
                print('nop')
        'Returns True if the node is a merge node.'
        return self.merge_parent is not None

    def __repr__(self) -> str:
        if False:
            for i in range(10):
                print('nop')
        return f'Commit : {self.commit_id} ({self.branch}) \nAuthor : {self.commit_user_name}\nTime   : {str(self.commit_time)[:-7]}\nMessage: {self.commit_message}' + (f'\nTotal samples processed in transform: {self.total_samples_processed}' if self.is_checkpoint else '')

    @property
    def is_head_node(self) -> bool:
        if False:
            i = 10
            return i + 15
        'Returns True if the node is the head node of the branch.'
        return self.commit_time is None
    __str__ = __repr__

    def to_json(self):
        if False:
            print('Hello World!')
        return {'branch': self.branch, 'children': [node.commit_id for node in self.children], 'parent': self.parent.commit_id if self.parent else None, 'commit_message': self.commit_message, 'commit_time': self.commit_time.timestamp() if self.commit_time else None, 'commit_user_name': self.commit_user_name, 'is_checkpoint': self.is_checkpoint, 'total_samples_processed': self.total_samples_processed}