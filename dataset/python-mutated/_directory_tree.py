from __future__ import annotations
from asyncio import Queue
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Callable, ClassVar, Iterable, Iterator
from ..await_complete import AwaitComplete
if TYPE_CHECKING:
    from typing_extensions import Self
from rich.style import Style
from rich.text import Text, TextType
from .. import work
from ..message import Message
from ..reactive import var
from ..worker import Worker, WorkerCancelled, WorkerFailed, get_current_worker
from ._tree import TOGGLE_STYLE, Tree, TreeNode

@dataclass
class DirEntry:
    """Attaches directory information to a node."""
    path: Path
    'The path of the directory entry.'
    loaded: bool = False
    'Has this been loaded?'

class DirectoryTree(Tree[DirEntry]):
    """A Tree widget that presents files and directories."""
    COMPONENT_CLASSES: ClassVar[set[str]] = {'directory-tree--extension', 'directory-tree--file', 'directory-tree--folder', 'directory-tree--hidden'}
    '\n    | Class | Description |\n    | :- | :- |\n    | `directory-tree--extension` | Target the extension of a file name. |\n    | `directory-tree--file` | Target files in the directory structure. |\n    | `directory-tree--folder` | Target folders in the directory structure. |\n    | `directory-tree--hidden` | Target hidden items in the directory structure. |\n\n    See also the [component classes for `Tree`][textual.widgets.Tree.COMPONENT_CLASSES].\n    '
    DEFAULT_CSS = '\n    DirectoryTree > .directory-tree--folder {\n        text-style: bold;\n    }\n\n    DirectoryTree > .directory-tree--extension {\n        text-style: italic;\n    }\n\n    DirectoryTree > .directory-tree--hidden {\n        color: $text 50%;\n    }\n    '
    PATH: Callable[[str | Path], Path] = Path
    'Callable that returns a fresh path object.'

    class FileSelected(Message):
        """Posted when a file is selected.

        Can be handled using `on_directory_tree_file_selected` in a subclass of
        `DirectoryTree` or in a parent widget in the DOM.
        """

        def __init__(self, node: TreeNode[DirEntry], path: Path) -> None:
            if False:
                for i in range(10):
                    print('nop')
            'Initialise the FileSelected object.\n\n            Args:\n                node: The tree node for the file that was selected.\n                path: The path of the file that was selected.\n            '
            super().__init__()
            self.node: TreeNode[DirEntry] = node
            'The tree node of the file that was selected.'
            self.path: Path = path
            'The path of the file that was selected.'

        @property
        def control(self) -> Tree[DirEntry]:
            if False:
                while True:
                    i = 10
            'The `Tree` that had a file selected.'
            return self.node.tree

    class DirectorySelected(Message):
        """Posted when a directory is selected.

        Can be handled using `on_directory_tree_directory_selected` in a
        subclass of `DirectoryTree` or in a parent widget in the DOM.
        """

        def __init__(self, node: TreeNode[DirEntry], path: Path) -> None:
            if False:
                print('Hello World!')
            'Initialise the DirectorySelected object.\n\n            Args:\n                node: The tree node for the directory that was selected.\n                path: The path of the directory that was selected.\n            '
            super().__init__()
            self.node: TreeNode[DirEntry] = node
            'The tree node of the directory that was selected.'
            self.path: Path = path
            'The path of the directory that was selected.'

        @property
        def control(self) -> Tree[DirEntry]:
            if False:
                return 10
            'The `Tree` that had a directory selected.'
            return self.node.tree
    path: var[str | Path] = var['str | Path'](PATH('.'), init=False, always_update=True)
    'The path that is the root of the directory tree.\n\n    Note:\n        This can be set to either a `str` or a `pathlib.Path` object, but\n        the value will always be a `pathlib.Path` object.\n    '

    def __init__(self, path: str | Path, *, name: str | None=None, id: str | None=None, classes: str | None=None, disabled: bool=False) -> None:
        if False:
            for i in range(10):
                print('nop')
        'Initialise the directory tree.\n\n        Args:\n            path: Path to directory.\n            name: The name of the widget, or None for no name.\n            id: The ID of the widget in the DOM, or None for no ID.\n            classes: A space-separated list of classes, or None for no classes.\n            disabled: Whether the directory tree is disabled or not.\n        '
        self._load_queue: Queue[TreeNode[DirEntry]] = Queue()
        super().__init__(str(path), data=DirEntry(self.PATH(path)), name=name, id=id, classes=classes, disabled=disabled)
        self.path = path

    def _add_to_load_queue(self, node: TreeNode[DirEntry]) -> AwaitComplete:
        if False:
            return 10
        'Add the given node to the load queue.\n\n        The return value can optionally be awaited until the queue is empty.\n\n        Args:\n            node: The node to add to the load queue.\n\n        Returns:\n            An optionally awaitable object that can be awaited until the\n            load queue has finished processing.\n        '
        assert node.data is not None
        if not node.data.loaded:
            node.data.loaded = True
            self._load_queue.put_nowait(node)
        return AwaitComplete(self._load_queue.join())

    def reload(self) -> AwaitComplete:
        if False:
            while True:
                i = 10
        'Reload the `DirectoryTree` contents.'
        self.reset(str(self.path), DirEntry(self.PATH(self.path)))
        self._load_queue = Queue()
        self._loader()
        queue_processed = self._add_to_load_queue(self.root)
        return queue_processed

    def clear_node(self, node: TreeNode[DirEntry]) -> Self:
        if False:
            return 10
        'Clear all nodes under the given node.\n\n        Returns:\n            The `Tree` instance.\n        '
        self._clear_line_cache()
        node_label = node._label
        node_data = node.data
        node_parent = node.parent
        node = TreeNode(self, node_parent, self._new_id(), node_label, node_data, expanded=True)
        self._updates += 1
        self.refresh()
        return self

    def reset_node(self, node: TreeNode[DirEntry], label: TextType, data: DirEntry | None=None) -> Self:
        if False:
            while True:
                i = 10
        'Clear the subtree and reset the given node.\n\n        Args:\n            node: The node to reset.\n            label: The label for the node.\n            data: Optional data for the node.\n\n        Returns:\n            The `Tree` instance.\n        '
        self.clear_node(node)
        node.label = label
        node.data = data
        return self

    def reload_node(self, node: TreeNode[DirEntry]) -> AwaitComplete:
        if False:
            while True:
                i = 10
        "Reload the given node's contents.\n\n        The return value may be awaited to ensure the DirectoryTree has reached\n        a stable state and is no longer performing any node reloading (of this node\n        or any other nodes).\n\n        Args:\n            node: The node to reload.\n        "
        self.reset_node(node, str(node.data.path.name), DirEntry(self.PATH(node.data.path)))
        return self._add_to_load_queue(node)

    def validate_path(self, path: str | Path) -> Path:
        if False:
            i = 10
            return i + 15
        'Ensure that the path is of the `Path` type.\n\n        Args:\n            path: The path to validate.\n\n        Returns:\n            The validated Path value.\n\n        Note:\n            The result will always be a Python `Path` object, regardless of\n            the value given.\n        '
        return self.PATH(path)

    async def watch_path(self) -> None:
        """Watch for changes to the `path` of the directory tree.

        If the path is changed the directory tree will be repopulated using
        the new value as the root.
        """
        await self.reload()

    def process_label(self, label: TextType) -> Text:
        if False:
            for i in range(10):
                print('nop')
        'Process a str or Text into a label. Maybe overridden in a subclass to modify how labels are rendered.\n\n        Args:\n            label: Label.\n\n        Returns:\n            A Rich Text object.\n        '
        if isinstance(label, str):
            text_label = Text(label)
        else:
            text_label = label
        first_line = text_label.split()[0]
        return first_line

    def render_label(self, node: TreeNode[DirEntry], base_style: Style, style: Style) -> Text:
        if False:
            i = 10
            return i + 15
        'Render a label for the given node.\n\n        Args:\n            node: A tree node.\n            base_style: The base style of the widget.\n            style: The additional style for the label.\n\n        Returns:\n            A Rich Text object containing the label.\n        '
        node_label = node._label.copy()
        node_label.stylize(style)
        if node._allow_expand:
            prefix = ('📂 ' if node.is_expanded else '📁 ', base_style + TOGGLE_STYLE)
            node_label.stylize_before(self.get_component_rich_style('directory-tree--folder', partial=True))
        else:
            prefix = ('📄 ', base_style)
            node_label.stylize_before(self.get_component_rich_style('directory-tree--file', partial=True))
            node_label.highlight_regex('\\..+$', self.get_component_rich_style('directory-tree--extension', partial=True))
        if node_label.plain.startswith('.'):
            node_label.stylize_before(self.get_component_rich_style('directory-tree--hidden'))
        text = Text.assemble(prefix, node_label)
        return text

    def filter_paths(self, paths: Iterable[Path]) -> Iterable[Path]:
        if False:
            i = 10
            return i + 15
        'Filter the paths before adding them to the tree.\n\n        Args:\n            paths: The paths to be filtered.\n\n        Returns:\n            The filtered paths.\n\n        By default this method returns all of the paths provided. To create\n        a filtered `DirectoryTree` inherit from it and implement your own\n        version of this method.\n        '
        return paths

    @staticmethod
    def _safe_is_dir(path: Path) -> bool:
        if False:
            for i in range(10):
                print('nop')
        'Safely check if a path is a directory.\n\n        Args:\n            path: The path to check.\n\n        Returns:\n            `True` if the path is for a directory, `False` if not.\n        '
        try:
            return path.is_dir()
        except PermissionError:
            return False

    def _populate_node(self, node: TreeNode[DirEntry], content: Iterable[Path]) -> None:
        if False:
            i = 10
            return i + 15
        'Populate the given tree node with the given directory content.\n\n        Args:\n            node: The Tree node to populate.\n            content: The collection of `Path` objects to populate the node with.\n        '
        node.remove_children()
        for path in content:
            node.add(path.name, data=DirEntry(path), allow_expand=self._safe_is_dir(path))
        node.expand()

    def _directory_content(self, location: Path, worker: Worker) -> Iterator[Path]:
        if False:
            while True:
                i = 10
        'Load the content of a given directory.\n\n        Args:\n            location: The location to load from.\n            worker: The worker that the loading is taking place in.\n\n        Yields:\n            Path: An entry within the location.\n        '
        try:
            for entry in location.iterdir():
                if worker.is_cancelled:
                    break
                yield entry
        except PermissionError:
            pass

    @work(thread=True)
    def _load_directory(self, node: TreeNode[DirEntry]) -> list[Path]:
        if False:
            return 10
        'Load the directory contents for a given node.\n\n        Args:\n            node: The node to load the directory contents for.\n\n        Returns:\n            The list of entries within the directory associated with the node.\n        '
        assert node.data is not None
        return sorted(self.filter_paths(self._directory_content(node.data.path, get_current_worker())), key=lambda path: (not self._safe_is_dir(path), path.name.lower()))

    @work(exclusive=True)
    async def _loader(self) -> None:
        """Background loading queue processor."""
        worker = get_current_worker()
        while not worker.is_cancelled:
            node = await self._load_queue.get()
            content: list[Path] = []
            try:
                content = await self._load_directory(node).wait()
            except WorkerCancelled:
                break
            except WorkerFailed:
                pass
            else:
                if content:
                    self._populate_node(node, content)
            finally:
                self._load_queue.task_done()

    async def _on_tree_node_expanded(self, event: Tree.NodeExpanded) -> None:
        event.stop()
        dir_entry = event.node.data
        if dir_entry is None:
            return
        if self._safe_is_dir(dir_entry.path):
            await self._add_to_load_queue(event.node)
        else:
            self.post_message(self.FileSelected(event.node, dir_entry.path))

    def _on_tree_node_selected(self, event: Tree.NodeSelected) -> None:
        if False:
            i = 10
            return i + 15
        event.stop()
        dir_entry = event.node.data
        if dir_entry is None:
            return
        if self._safe_is_dir(dir_entry.path):
            self.post_message(self.DirectorySelected(event.node, dir_entry.path))
        else:
            self.post_message(self.FileSelected(event.node, dir_entry.path))