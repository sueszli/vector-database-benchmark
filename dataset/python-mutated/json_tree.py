import json
from pathlib import Path
from rich.text import Text
from textual.app import App, ComposeResult
from textual.widgets import Header, Footer, Tree
from textual.widgets.tree import TreeNode

class TreeApp(App):
    BINDINGS = [('a', 'add', 'Add node'), ('c', 'clear', 'Clear'), ('t', 'toggle_root', 'Toggle root')]

    def compose(self) -> ComposeResult:
        if False:
            while True:
                i = 10
        yield Header()
        yield Footer()
        yield Tree('Root')

    @classmethod
    def add_json(cls, node: TreeNode, json_data: object) -> None:
        if False:
            return 10
        'Adds JSON data to a node.\n\n        Args:\n            node (TreeNode): A Tree node.\n            json_data (object): An object decoded from JSON.\n        '
        from rich.highlighter import ReprHighlighter
        highlighter = ReprHighlighter()

        def add_node(name: str, node: TreeNode, data: object) -> None:
            if False:
                return 10
            'Adds a node to the tree.\n\n            Args:\n                name (str): Name of the node.\n                node (TreeNode): Parent node.\n                data (object): Data associated with the node.\n            '
            if isinstance(data, dict):
                node.set_label(Text(f'{{}} {name}'))
                for (key, value) in data.items():
                    new_node = node.add('')
                    add_node(key, new_node, value)
            elif isinstance(data, list):
                node.set_label(Text(f'[] {name}'))
                for (index, value) in enumerate(data):
                    new_node = node.add('')
                    add_node(str(index), new_node, value)
            else:
                node.allow_expand = False
                if name:
                    label = Text.assemble(Text.from_markup(f'[b]{name}[/b]='), highlighter(repr(data)))
                else:
                    label = Text(repr(data))
                node.set_label(label)
        add_node('JSON', node, json_data)

    def on_mount(self) -> None:
        if False:
            while True:
                i = 10
        'Load some JSON when the app starts.'
        file_path = Path(__file__).parent / 'food.json'
        with open(file_path) as data_file:
            self.json_data = json.load(data_file)

    def action_add(self) -> None:
        if False:
            print('Hello World!')
        'Add a node to the tree.'
        tree = self.query_one(Tree)
        json_node = tree.root.add('JSON')
        self.add_json(json_node, self.json_data)
        tree.root.expand()

    def action_clear(self) -> None:
        if False:
            print('Hello World!')
        'Clear the tree (remove all nodes).'
        tree = self.query_one(Tree)
        tree.clear()

    def action_toggle_root(self) -> None:
        if False:
            print('Hello World!')
        'Toggle the root node.'
        tree = self.query_one(Tree)
        tree.show_root = not tree.show_root
if __name__ == '__main__':
    app = TreeApp()
    app.run()