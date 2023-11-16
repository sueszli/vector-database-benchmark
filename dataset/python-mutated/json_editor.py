from typing import Callable, Dict, Optional
from ..element import Element
from ..events import GenericEventArguments, JsonEditorChangeEventArguments, JsonEditorSelectEventArguments, handle_event

class JsonEditor(Element, component='json_editor.js', exposed_libraries=['lib/vanilla-jsoneditor/index.js']):

    def __init__(self, properties: Dict, *, on_select: Optional[Callable]=None, on_change: Optional[Callable]=None) -> None:
        if False:
            i = 10
            return i + 15
        'JSONEditor\n\n        An element to create a JSON editor using `JSONEditor <https://github.com/josdejong/svelte-jsoneditor>`_.\n        Updates can be pushed to the editor by changing the `properties` property.\n        After data has changed, call the `update` method to refresh the editor.\n\n        :param properties: dictionary of JSONEditor properties\n        :param on_select: callback function that is called when some of the content has been selected\n        :param on_change: callback function that is called when the content has changed\n        '
        super().__init__()
        self._props['properties'] = properties
        if on_select:

            def handle_on_select(e: GenericEventArguments) -> None:
                if False:
                    while True:
                        i = 10
                handle_event(on_select, JsonEditorSelectEventArguments(sender=self, client=self.client, **e.args))
            self.on('select', handle_on_select, ['selection'])
        if on_change:

            def handle_on_change(e: GenericEventArguments) -> None:
                if False:
                    print('Hello World!')
                handle_event(on_change, JsonEditorChangeEventArguments(sender=self, client=self.client, **e.args))
            self.on('change', handle_on_change, ['content', 'errors'])

    @property
    def properties(self) -> Dict:
        if False:
            print('Hello World!')
        'The property dictionary.'
        return self._props['properties']

    def update(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        super().update()
        self.run_method('update_editor')