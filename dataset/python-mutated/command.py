"""
Command pattern decouples the object invoking a job from the one who knows
how to do it. As mentioned in the GoF book, a good example is in menu items.
You have a menu that has lots of items. Each item is responsible for doing a
special thing and you want your menu item just call the execute method when
it is pressed. To achieve this you implement a command object with the execute
method for each menu item and pass to it.

*About the example
We have a menu containing two items. Each item accepts a file name, one hides the file
and the other deletes it. Both items have an undo option.
Each item is a MenuItem class that accepts the corresponding command as input and executes
it's execute method when it is pressed.

*TL;DR
Object oriented implementation of callback functions.

*Examples in Python ecosystem:
Django HttpRequest (without execute method):
https://docs.djangoproject.com/en/2.1/ref/request-response/#httprequest-objects
"""
from typing import List, Union

class HideFileCommand:
    """
    A command to hide a file given its name
    """

    def __init__(self) -> None:
        if False:
            i = 10
            return i + 15
        self._hidden_files: List[str] = []

    def execute(self, filename: str) -> None:
        if False:
            i = 10
            return i + 15
        print(f'hiding {filename}')
        self._hidden_files.append(filename)

    def undo(self) -> None:
        if False:
            return 10
        filename = self._hidden_files.pop()
        print(f'un-hiding {filename}')

class DeleteFileCommand:
    """
    A command to delete a file given its name
    """

    def __init__(self) -> None:
        if False:
            i = 10
            return i + 15
        self._deleted_files: List[str] = []

    def execute(self, filename: str) -> None:
        if False:
            print('Hello World!')
        print(f'deleting {filename}')
        self._deleted_files.append(filename)

    def undo(self) -> None:
        if False:
            i = 10
            return i + 15
        filename = self._deleted_files.pop()
        print(f'restoring {filename}')

class MenuItem:
    """
    The invoker class. Here it is items in a menu.
    """

    def __init__(self, command: Union[HideFileCommand, DeleteFileCommand]) -> None:
        if False:
            i = 10
            return i + 15
        self._command = command

    def on_do_press(self, filename: str) -> None:
        if False:
            return 10
        self._command.execute(filename)

    def on_undo_press(self) -> None:
        if False:
            while True:
                i = 10
        self._command.undo()

def main():
    if False:
        for i in range(10):
            print('nop')
    "\n    >>> item1 = MenuItem(DeleteFileCommand())\n\n    >>> item2 = MenuItem(HideFileCommand())\n\n    # create a file named `test-file` to work with\n    >>> test_file_name = 'test-file'\n\n    # deleting `test-file`\n    >>> item1.on_do_press(test_file_name)\n    deleting test-file\n\n    # restoring `test-file`\n    >>> item1.on_undo_press()\n    restoring test-file\n\n    # hiding `test-file`\n    >>> item2.on_do_press(test_file_name)\n    hiding test-file\n\n    # un-hiding `test-file`\n    >>> item2.on_undo_press()\n    un-hiding test-file\n    "
if __name__ == '__main__':
    import doctest
    doctest.testmod()