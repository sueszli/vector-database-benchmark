from ..element import Element

class ContextMenu(Element):

    def __init__(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        "Context Menu\n\n        Creates a context menu based on Quasar's `QMenu <https://quasar.dev/vue-components/menu>`_ component.\n        The context menu should be placed inside the element where it should be shown.\n        It is automatically opened when the user right-clicks on the element and appears at the mouse position.\n        "
        super().__init__('q-menu')
        self._props['context-menu'] = True
        self._props['touch-position'] = True

    def open(self) -> None:
        if False:
            return 10
        'Open the context menu.'
        self.run_method('show')

    def close(self) -> None:
        if False:
            print('Hello World!')
        'Close the context menu.'
        self.run_method('hide')