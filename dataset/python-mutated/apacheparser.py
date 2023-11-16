""" apacheconfig implementation of the ParserNode interfaces """
from typing import Any
from typing import Iterable
from typing import List
from typing import Optional
from typing import Tuple
from certbot_apache._internal import assertions
from certbot_apache._internal import interfaces
from certbot_apache._internal import parsernode_util as util
from certbot_apache._internal.interfaces import ParserNode

class ApacheParserNode(interfaces.ParserNode):
    """ apacheconfig implementation of ParserNode interface.

        Expects metadata `ac_ast` to be passed in, where `ac_ast` is the AST provided
        by parsing the equivalent configuration text using the apacheconfig library.
    """

    def __init__(self, **kwargs: Any) -> None:
        if False:
            while True:
                i = 10
        (ancestor, dirty, filepath, metadata) = util.parsernode_kwargs(kwargs)
        super().__init__(**kwargs)
        self.ancestor = ancestor
        self.filepath = filepath
        self.dirty = dirty
        self.metadata = metadata
        self._raw: Any = self.metadata['ac_ast']

    def save(self, msg: str) -> None:
        if False:
            for i in range(10):
                print('nop')
        pass

    def find_ancestors(self, name: str) -> List['ApacheParserNode']:
        if False:
            for i in range(10):
                print('nop')
        'Find ancestor BlockNodes with a given name'
        return [ApacheBlockNode(name=assertions.PASS, parameters=assertions.PASS, ancestor=self, filepath=assertions.PASS, metadata=self.metadata)]

class ApacheCommentNode(ApacheParserNode):
    """ apacheconfig implementation of CommentNode interface """

    def __init__(self, **kwargs: Any) -> None:
        if False:
            while True:
                i = 10
        (comment, kwargs) = util.commentnode_kwargs(kwargs)
        super().__init__(**kwargs)
        self.comment = comment

    def __eq__(self, other: Any) -> bool:
        if False:
            print('Hello World!')
        if isinstance(other, self.__class__):
            return self.comment == other.comment and self.dirty == other.dirty and (self.ancestor == other.ancestor) and (self.metadata == other.metadata) and (self.filepath == other.filepath)
        return False

class ApacheDirectiveNode(ApacheParserNode):
    """ apacheconfig implementation of DirectiveNode interface """

    def __init__(self, **kwargs: Any) -> None:
        if False:
            for i in range(10):
                print('nop')
        (name, parameters, enabled, kwargs) = util.directivenode_kwargs(kwargs)
        super().__init__(**kwargs)
        self.name = name
        self.parameters = parameters
        self.enabled = enabled
        self.include: Optional[str] = None

    def __eq__(self, other: Any) -> bool:
        if False:
            print('Hello World!')
        if isinstance(other, self.__class__):
            return self.name == other.name and self.filepath == other.filepath and (self.parameters == other.parameters) and (self.enabled == other.enabled) and (self.dirty == other.dirty) and (self.ancestor == other.ancestor) and (self.metadata == other.metadata)
        return False

    def set_parameters(self, _parameters: Iterable[str]) -> None:
        if False:
            for i in range(10):
                print('nop')
        'Sets the parameters for DirectiveNode'
        return

class ApacheBlockNode(ApacheDirectiveNode):
    """ apacheconfig implementation of BlockNode interface """

    def __init__(self, **kwargs: Any) -> None:
        if False:
            while True:
                i = 10
        super().__init__(**kwargs)
        self.children: Tuple[ApacheParserNode, ...] = ()

    def __eq__(self, other: Any) -> bool:
        if False:
            return 10
        if isinstance(other, self.__class__):
            return self.name == other.name and self.filepath == other.filepath and (self.parameters == other.parameters) and (self.children == other.children) and (self.enabled == other.enabled) and (self.dirty == other.dirty) and (self.ancestor == other.ancestor) and (self.metadata == other.metadata)
        return False

    def add_child_block(self, name: str, parameters: Optional[List[str]]=None, position: Optional[int]=None) -> 'ApacheBlockNode':
        if False:
            print('Hello World!')
        'Adds a new BlockNode to the sequence of children'
        new_block = ApacheBlockNode(name=assertions.PASS, parameters=assertions.PASS, ancestor=self, filepath=assertions.PASS, metadata=self.metadata)
        self.children += (new_block,)
        return new_block

    def add_child_directive(self, name: str, parameters: Optional[List[str]]=None, position: Optional[int]=None) -> ApacheDirectiveNode:
        if False:
            while True:
                i = 10
        'Adds a new DirectiveNode to the sequence of children'
        new_dir = ApacheDirectiveNode(name=assertions.PASS, parameters=assertions.PASS, ancestor=self, filepath=assertions.PASS, metadata=self.metadata)
        self.children += (new_dir,)
        return new_dir

    def add_child_comment(self, name: str, parameters: Optional[int]=None, position: Optional[int]=None) -> ApacheCommentNode:
        if False:
            i = 10
            return i + 15
        'Adds a new CommentNode to the sequence of children'
        new_comment = ApacheCommentNode(comment=assertions.PASS, ancestor=self, filepath=assertions.PASS, metadata=self.metadata)
        self.children += (new_comment,)
        return new_comment

    def find_blocks(self, name: str, exclude: bool=True) -> List['ApacheBlockNode']:
        if False:
            i = 10
            return i + 15
        'Recursive search of BlockNodes from the sequence of children'
        return [ApacheBlockNode(name=assertions.PASS, parameters=assertions.PASS, ancestor=self, filepath=assertions.PASS, metadata=self.metadata)]

    def find_directives(self, name: str, exclude: bool=True) -> List[ApacheDirectiveNode]:
        if False:
            print('Hello World!')
        'Recursive search of DirectiveNodes from the sequence of children'
        return [ApacheDirectiveNode(name=assertions.PASS, parameters=assertions.PASS, ancestor=self, filepath=assertions.PASS, metadata=self.metadata)]

    def find_comments(self, comment: str, exact: bool=False) -> List[ApacheCommentNode]:
        if False:
            while True:
                i = 10
        'Recursive search of DirectiveNodes from the sequence of children'
        return [ApacheCommentNode(comment=assertions.PASS, ancestor=self, filepath=assertions.PASS, metadata=self.metadata)]

    def delete_child(self, child: ParserNode) -> None:
        if False:
            for i in range(10):
                print('nop')
        'Deletes a ParserNode from the sequence of children'
        return

    def unsaved_files(self) -> List[str]:
        if False:
            return 10
        'Returns a list of unsaved filepaths'
        return [assertions.PASS]

    def parsed_paths(self) -> List[str]:
        if False:
            while True:
                i = 10
        'Returns a list of parsed configuration file paths'
        return [assertions.PASS]
interfaces.CommentNode.register(ApacheCommentNode)
interfaces.DirectiveNode.register(ApacheDirectiveNode)
interfaces.BlockNode.register(ApacheBlockNode)