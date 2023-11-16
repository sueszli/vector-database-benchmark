import json
from abc import ABC, abstractmethod
from typing import Any, Generic, Optional, TypeVar
from streamlit.runtime.secrets import AttrDict, secrets_singleton
from streamlit.util import calc_md5
RawConnectionT = TypeVar('RawConnectionT')

class BaseConnection(ABC, Generic[RawConnectionT]):
    """The abstract base class that all Streamlit Connections must inherit from.

    This base class provides connection authors with a standardized way to hook into the
    ``st.connection()`` factory function: connection authors are required to provide an
    implementation for the abstract method ``_connect`` in their subclasses.

    Additionally, it also provides a few methods/properties designed to make
    implementation of connections more convenient. See the docstrings for each of the
    methods of this class for more information

    .. note::
        While providing an implementation of ``_connect`` is technically all that's
        required to define a valid connection, connections should also provide the user
        with context-specific ways of interacting with the underlying connection object.
        For example, the first-party SQLConnection provides a ``query()`` method for
        reads and a ``session`` property for more complex operations.
    """

    def __init__(self, connection_name: str, **kwargs) -> None:
        if False:
            return 10
        "Create a BaseConnection.\n\n        This constructor is called by the connection factory machinery when a user\n        script calls ``st.connection()``.\n\n        Subclasses of BaseConnection that want to overwrite this method should take care\n        to also call the base class' implementation.\n\n        Parameters\n        ----------\n        connection_name : str\n            The name of this connection. This corresponds to the\n            ``[connections.<connection_name>]`` config section in ``st.secrets``.\n        kwargs : dict\n            Any other kwargs to pass to this connection class' ``_connect`` method.\n\n        Returns\n        -------\n        None\n        "
        self._connection_name = connection_name
        self._kwargs = kwargs
        self._config_section_hash = calc_md5(json.dumps(self._secrets.to_dict()))
        secrets_singleton.file_change_listener.connect(self._on_secrets_changed)
        self._raw_instance: Optional[RawConnectionT] = self._connect(**kwargs)

    def __del__(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        secrets_singleton.file_change_listener.disconnect(self._on_secrets_changed)

    def __getattribute__(self, name: str) -> Any:
        if False:
            print('Hello World!')
        try:
            return object.__getattribute__(self, name)
        except AttributeError as e:
            if hasattr(self._instance, name):
                raise AttributeError(f"`{name}` doesn't exist here, but you can call `._instance.{name}` instead")
            raise e

    def _repr_html_(self) -> str:
        if False:
            for i in range(10):
                print('nop')
        'Return a human-friendly markdown string describing this connection.\n\n        This is the string that will be written to the app if a user calls\n        ``st.write(this_connection)``. Subclasses of BaseConnection can freely overwrite\n        this method if desired.\n\n        Returns\n        -------\n        str\n        '
        module_name = getattr(self, '__module__', None)
        class_name = type(self).__name__
        cfg = f'- Configured from `[connections.{self._connection_name}]`' if len(self._secrets) else ''
        return f'\n---\n**st.connection {self._connection_name} built from `{module_name}.{class_name}`**\n{cfg}\n- Learn more using `st.help()`\n---\n'

    def _on_secrets_changed(self, _) -> None:
        if False:
            while True:
                i = 10
        "Reset the raw connection object when this connection's secrets change.\n\n        We don't expect either user scripts or connection authors to have to use or\n        overwrite this method.\n        "
        new_hash = calc_md5(json.dumps(self._secrets.to_dict()))
        if new_hash != self._config_section_hash:
            self._config_section_hash = new_hash
            self.reset()

    @property
    def _secrets(self) -> AttrDict:
        if False:
            while True:
                i = 10
        "Get the secrets for this connection from the corresponding st.secrets section.\n\n        We expect this property to be used primarily by connection authors when they\n        are implementing their class' ``_connect`` method. User scripts should, for the\n        most part, have no reason to use this property.\n        "
        connections_section = None
        if secrets_singleton.load_if_toml_exists():
            connections_section = secrets_singleton.get('connections')
        if type(connections_section) is not AttrDict:
            return AttrDict({})
        return connections_section.get(self._connection_name, AttrDict({}))

    def reset(self) -> None:
        if False:
            while True:
                i = 10
        'Reset this connection so that it gets reinitialized the next time it\'s used.\n\n        This method can be useful when a connection has become stale, an auth token has\n        expired, or in similar scenarios where a broken connection might be fixed by\n        reinitializing it. Note that some connection methods may already use ``reset()``\n        in their error handling code.\n\n        Example\n        -------\n        >>> import streamlit as st\n        >>>\n        >>> conn = st.connection("my_conn")\n        >>>\n        >>> # Reset the connection before using it if it isn\'t healthy\n        >>> # Note: is_healthy() isn\'t a real method and is just shown for example here.\n        >>> if not conn.is_healthy():\n        ...     conn.reset()\n        ...\n        >>> # Do stuff with conn...\n        '
        self._raw_instance = None

    @property
    def _instance(self) -> RawConnectionT:
        if False:
            i = 10
            return i + 15
        'Get an instance of the underlying connection, creating a new one if needed.'
        if self._raw_instance is None:
            self._raw_instance = self._connect(**self._kwargs)
        return self._raw_instance

    @abstractmethod
    def _connect(self, **kwargs) -> RawConnectionT:
        if False:
            print('Hello World!')
        'Create an instance of an underlying connection object.\n\n        This abstract method is the one method that we require subclasses of\n        BaseConnection to provide an implementation for. It is called when first\n        creating a connection and when reconnecting after a connection is reset.\n\n        Parameters\n        ----------\n        kwargs : dict\n\n        Returns\n        -------\n        RawConnectionT\n            The underlying connection object.\n        '
        raise NotImplementedError