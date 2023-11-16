from typing import Optional, Union, TYPE_CHECKING
from metaflow.datastore import FlowDataStore
from metaflow.metaflow_config import CARD_SUFFIX
from .card_resolver import resolve_paths_from_task, resumed_info
from .card_datastore import CardDatastore
from .exception import UnresolvableDatastoreException, IncorrectArguementException, IncorrectPathspecException
import os
import tempfile
import uuid
if TYPE_CHECKING:
    from metaflow.client.core import Task
_TYPE = type
_ID_FUNC = id

class Card:
    """
    `Card` represents an individual Metaflow Card, a single HTML file, produced by
    the card `@card` decorator. `Card`s are contained by `CardContainer`, returned by
    `get_cards`.

    Note that the contents of the card, an HTML file, is retrieved lazily when you call
    `Card.get` for the first time or when the card is rendered in a notebook.
    """

    def __init__(self, card_ds, type, path, hash, id=None, html=None, created_on=None, from_resumed=False, origin_pathspec=None):
        if False:
            while True:
                i = 10
        self._path = path
        self._html = html
        self._created_on = created_on
        self._card_ds = card_ds
        self._card_id = id
        self.hash = hash
        self.type = type
        self.from_resumed = from_resumed
        self.origin_pathspec = origin_pathspec
        self._temp_file = None

    def get(self) -> str:
        if False:
            i = 10
            return i + 15
        '\n        Retrieves the HTML contents of the card from the\n        Metaflow datastore.\n\n        Returns\n        -------\n        str\n            HTML contents of the card.\n        '
        if self._html is not None:
            return self._html
        self._html = self._card_ds.get_card_html(self.path)
        return self._html

    @property
    def path(self) -> str:
        if False:
            for i in range(10):
                print('nop')
        '\n        The path of the card in the datastore which uniquely\n        identifies the card.\n\n        Returns\n        -------\n        str\n            Path to the card\n        '
        return self._path

    @property
    def id(self) -> Optional[str]:
        if False:
            while True:
                i = 10
        '\n        The ID of the card, if specified with `@card(id=ID)`.\n        '
        return self._card_id

    def __str__(self):
        if False:
            i = 10
            return i + 15
        return "<Card at '%s'>" % self._path

    def view(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        "\n        Opens the card in a local web browser.\n\n        This call uses Python's built-in [`webbrowser`](https://docs.python.org/3/library/webbrowser.html)\n        module to open the card.\n        "
        import webbrowser
        self._temp_file = tempfile.NamedTemporaryFile(suffix='.html')
        html = self.get()
        self._temp_file.write(html.encode())
        self._temp_file.seek(0)
        url = 'file://' + os.path.abspath(self._temp_file.name)
        webbrowser.open(url)

    def _repr_html_(self):
        if False:
            print('Hello World!')
        main_html = []
        container_id = uuid.uuid4()
        main_html.append("<script type='text/javascript'>var mfContainerId = '%s';</script>" % container_id)
        main_html.append("<div class='embed' data-container='%s'>%s</div>" % (container_id, self.get()))
        return '\n'.join(main_html)

class CardContainer:
    """
    `CardContainer` is an immutable list-like object, returned by `get_cards`,
    which contains individual `Card`s.

    Notably, `CardContainer` contains a special
    `_repr_html_` function which renders cards automatically in an output
    cell of a notebook.

    The following operations are supported:
    ```
    cards = get_cards(MyTask)

    # retrieve by index
    first_card = cards[0]

    # check length
    if len(cards) > 1:
        print('many cards present!')

    # iteration
    list_of_cards = list(cards)
    ```
    """

    def __init__(self, card_paths, card_ds, from_resumed=False, origin_pathspec=None):
        if False:
            i = 10
            return i + 15
        self._card_paths = card_paths
        self._card_ds = card_ds
        self._current = 0
        self._high = len(card_paths)
        self.from_resumed = from_resumed
        self.origin_pathspec = origin_pathspec

    def __len__(self):
        if False:
            i = 10
            return i + 15
        return self._high

    def __iter__(self):
        if False:
            for i in range(10):
                print('nop')
        for idx in range(self._high):
            yield self._get_card(idx)

    def __getitem__(self, index):
        if False:
            for i in range(10):
                print('nop')
        return self._get_card(index)

    def _get_card(self, index):
        if False:
            for i in range(10):
                print('nop')
        if index >= self._high:
            raise IndexError
        path = self._card_paths[index]
        card_info = self._card_ds.card_info_from_path(path)
        return Card(self._card_ds, card_info.type, path, card_info.hash, id=card_info.id, html=None, created_on=None)

    def _make_heading(self, type):
        if False:
            i = 10
            return i + 15
        return '<h1>Displaying Card Of Type : %s</h1>' % type.title()

    def _repr_html_(self):
        if False:
            return 10
        main_html = []
        for (idx, _) in enumerate(self._card_paths):
            card = self._get_card(idx)
            main_html.append(self._make_heading(card.type))
            container_id = uuid.uuid4()
            main_html.append("<script type='text/javascript'>var mfContainerId = '%s';</script>" % container_id)
            main_html.append("<div class='embed' data-container='%s'>%s</div>" % (container_id, card.get()))
        return '\n'.join(main_html)

def get_cards(task: Union[str, 'Task'], id: Optional[str]=None, type: Optional[str]=None, follow_resumed: bool=True) -> CardContainer:
    if False:
        for i in range(10):
            print('nop')
    "\n    Get cards related to a `Task`.\n\n    Note that `get_cards` resolves the cards contained by the task, but it doesn't actually\n    retrieve them from the datastore. Actual card contents are retrieved lazily either when\n    the card is rendered in a notebook to when you call `Card.get`. This means that\n    `get_cards` is a fast call even when individual cards contain a lot of data.\n\n    Parameters\n    ----------\n    task : str or `Task`\n        A `Task` object or pathspec `{flow_name}/{run_id}/{step_name}/{task_id}` that\n        uniquely identifies a task.\n    id : str, optional\n        The ID of card to retrieve if multiple cards are present.\n    type : str, optional\n        The type of card to retrieve if multiple cards are present.\n    follow_resumed : bool, default: True\n        If the task has been resumed, then setting this flag will resolve the card for\n        the origin task.\n\n    Returns\n    -------\n    CardContainer\n        A list-like object that holds `Card` objects.\n    "
    from metaflow.client import Task
    from metaflow import namespace
    card_id = id
    if isinstance(task, str):
        task_str = task
        if len(task_str.split('/')) != 4:
            raise IncorrectPathspecException(task_str)
        namespace(None)
        task = Task(task_str)
    elif not isinstance(task, Task):
        raise IncorrectArguementException(_TYPE(task))
    if follow_resumed:
        origin_taskpathspec = resumed_info(task)
        if origin_taskpathspec:
            task = Task(origin_taskpathspec)
    (card_paths, card_ds) = resolve_paths_from_task(_get_flow_datastore(task), pathspec=task.pathspec, type=type, card_id=card_id)
    return CardContainer(card_paths, card_ds, from_resumed=origin_taskpathspec is not None, origin_pathspec=origin_taskpathspec)

def _get_flow_datastore(task):
    if False:
        for i in range(10):
            print('nop')
    flow_name = task.pathspec.split('/')[0]
    ds_type = None
    meta_dict = task.metadata_dict
    ds_type = meta_dict.get('ds-type', None)
    if ds_type is None:
        raise UnresolvableDatastoreException(task)
    ds_root = meta_dict.get('ds-root', None)
    if ds_root:
        ds_root = os.path.join(ds_root, CARD_SUFFIX)
    else:
        ds_root = CardDatastore.get_storage_root(ds_type)
    from metaflow.plugins import DATASTORES
    storage_impl = [d for d in DATASTORES if d.TYPE == ds_type][0]
    return FlowDataStore(flow_name=flow_name, environment=None, storage_impl=storage_impl, ds_root=ds_root)