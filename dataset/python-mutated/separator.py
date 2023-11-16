from ..element import Element

class Separator(Element):

    def __init__(self) -> None:
        if False:
            print('Hello World!')
        "Separator\n\n        This element is based on Quasar's `QSeparator <https://quasar.dev/vue-components/separator>`_ component.\n\n        It serves as a separator for cards, menus and other component containers and is similar to HTML's <hr> tag.\n        "
        super().__init__('q-separator')
        self._classes.append('nicegui-separator')