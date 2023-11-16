from .mixins.text_element import TextElement

class Tooltip(TextElement):

    def __init__(self, text: str) -> None:
        if False:
            print('Hello World!')
        "Tooltip\n\n        This element is based on Quasar's `QTooltip <https://quasar.dev/vue-components/tooltip>`_ component.\n        It be placed in another element to show additional information on hover.\n\n        :param text: the content of the tooltip\n        "
        super().__init__(tag='q-tooltip', text=text)