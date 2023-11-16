"""Base class for Telegram Objects."""
import datetime
import inspect
import json
from collections.abc import Sized
from contextlib import contextmanager
from copy import deepcopy
from itertools import chain
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, ClassVar, Dict, Iterator, List, Mapping, Optional, Set, Tuple, Type, TypeVar, Union, cast
from telegram._utils.datetime import to_timestamp
from telegram._utils.types import JSONDict
from telegram._utils.warnings import warn
if TYPE_CHECKING:
    from telegram import Bot
Tele_co = TypeVar('Tele_co', bound='TelegramObject', covariant=True)

class TelegramObject:
    """Base class for most Telegram objects.

    Objects of this type are subscriptable with strings. See :meth:`__getitem__` for more details.
    The :mod:`pickle` and :func:`~copy.deepcopy` behavior of objects of this type are defined by
    :meth:`__getstate__`, :meth:`__setstate__` and :meth:`__deepcopy__`.

    Tip:
        Objects of this type can be serialized via Python's :mod:`pickle` module and pickled
        objects from one version of PTB are usually loadable in future versions. However, we can
        not guarantee that this compatibility will always be provided. At least a manual one-time
        conversion of the data may be needed on major updates of the library.

    .. versionchanged:: 20.0

        * Removed argument and attribute ``bot`` for several subclasses. Use
          :meth:`set_bot` and :meth:`get_bot` instead.
        * Removed the possibility to pass arbitrary keyword arguments for several subclasses.
        * String representations objects of this type was overhauled. See :meth:`__repr__` for
          details. As this class doesn't implement :meth:`object.__str__`, the default
          implementation will be used, which is equivalent to :meth:`__repr__`.
        * Objects of this class (or subclasses) are now immutable. This means that you can't set
          or delete attributes anymore. Moreover, attributes that were formerly of type
          :obj:`list` are now of type :obj:`tuple`.

    Arguments:
        api_kwargs (Dict[:obj:`str`, any], optional): |toapikwargsarg|

            .. versionadded:: 20.0

    Attributes:
        api_kwargs (:obj:`types.MappingProxyType` [:obj:`str`, any]): |toapikwargsattr|

            .. versionadded:: 20.0

    """
    __slots__ = ('_id_attrs', '_bot', '_frozen', 'api_kwargs')
    __INIT_PARAMS: ClassVar[Set[str]] = set()
    __INIT_PARAMS_CHECK: Optional[Type['TelegramObject']] = None

    def __init__(self, *, api_kwargs: Optional[JSONDict]=None) -> None:
        if False:
            while True:
                i = 10
        self._frozen: bool = False
        self._id_attrs: Tuple[object, ...] = ()
        self._bot: Optional[Bot] = None
        self.api_kwargs: Mapping[str, Any] = MappingProxyType(api_kwargs or {})

    def __eq__(self, other: object) -> bool:
        if False:
            return 10
        "Compares this object with :paramref:`other` in terms of equality.\n        If this object and :paramref:`other` are `not` objects of the same class,\n        this comparison will fall back to Python's default implementation of :meth:`object.__eq__`.\n        Otherwise, both objects may be compared in terms of equality, if the corresponding\n        subclass of :class:`TelegramObject` has defined a set of attributes to compare and\n        the objects are considered to be equal, if all of these attributes are equal.\n        If the subclass has not defined a set of attributes to compare, a warning will be issued.\n\n        Tip:\n            If instances of a class in the :mod:`telegram` module are comparable in terms of\n            equality, the documentation of the class will state the attributes that will be used\n            for this comparison.\n\n        Args:\n            other (:obj:`object`): The object to compare with.\n\n        Returns:\n            :obj:`bool`\n\n        "
        if isinstance(other, self.__class__):
            if not self._id_attrs:
                warn(f'Objects of type {self.__class__.__name__} can not be meaningfully tested for equivalence.', stacklevel=2)
            if not other._id_attrs:
                warn(f'Objects of type {other.__class__.__name__} can not be meaningfully tested for equivalence.', stacklevel=2)
            return self._id_attrs == other._id_attrs
        return super().__eq__(other)

    def __hash__(self) -> int:
        if False:
            i = 10
            return i + 15
        'Builds a hash value for this object such that the hash of two objects is equal if and\n        only if the objects are equal in terms of :meth:`__eq__`.\n\n        Returns:\n            :obj:`int`\n        '
        if self._id_attrs:
            return hash((self.__class__, self._id_attrs))
        return super().__hash__()

    def __setattr__(self, key: str, value: object) -> None:
        if False:
            return 10
        'Overrides :meth:`object.__setattr__` to prevent the overriding of attributes.\n\n        Raises:\n            :exc:`AttributeError`\n        '
        if key[0] == '_' or not getattr(self, '_frozen', True):
            super().__setattr__(key, value)
            return
        raise AttributeError(f"Attribute `{key}` of class `{self.__class__.__name__}` can't be set!")

    def __delattr__(self, key: str) -> None:
        if False:
            return 10
        'Overrides :meth:`object.__delattr__` to prevent the deletion of attributes.\n\n        Raises:\n            :exc:`AttributeError`\n        '
        if key[0] == '_' or not getattr(self, '_frozen', True):
            super().__delattr__(key)
            return
        raise AttributeError(f"Attribute `{key}` of class `{self.__class__.__name__}` can't be deleted!")

    def __repr__(self) -> str:
        if False:
            i = 10
            return i + 15
        "Gives a string representation of this object in the form\n        ``ClassName(attr_1=value_1, attr_2=value_2, ...)``, where attributes are omitted if they\n        have the value :obj:`None` or are empty instances of :class:`collections.abc.Sized` (e.g.\n        :class:`list`, :class:`dict`, :class:`set`, :class:`str`, etc.).\n\n        As this class doesn't implement :meth:`object.__str__`, the default implementation\n        will be used, which is equivalent to :meth:`__repr__`.\n\n        Returns:\n            :obj:`str`\n        "
        as_dict = self._get_attrs(recursive=False, include_private=False)
        if not self.api_kwargs:
            as_dict.pop('api_kwargs', None)
        else:
            as_dict['api_kwargs'] = dict(self.api_kwargs)
        contents = ', '.join((f'{k}={as_dict[k]!r}' for k in sorted(as_dict.keys()) if as_dict[k] is not None and (not (isinstance(as_dict[k], Sized) and len(as_dict[k]) == 0))))
        return f'{self.__class__.__name__}({contents})'

    def __getitem__(self, item: str) -> object:
        if False:
            return 10
        '\n        Objects of this type are subscriptable with strings, where\n        ``telegram_object["attribute_name"]`` is equivalent to ``telegram_object.attribute_name``.\n\n        Tip:\n            This is useful for dynamic attribute lookup, i.e. ``telegram_object[arg]`` where the\n            value of ``arg`` is determined at runtime.\n            In all other cases, it\'s recommended to use the dot notation instead, i.e.\n            ``telegram_object.attribute_name``.\n\n        .. versionchanged:: 20.0\n\n            ``telegram_object[\'from\']`` will look up the key ``from_user``. This is to account for\n            special cases like :attr:`Message.from_user` that deviate from the official Bot API.\n\n        Args:\n            item (:obj:`str`): The name of the attribute to look up.\n\n        Returns:\n            :obj:`object`\n\n        Raises:\n            :exc:`KeyError`: If the object does not have an attribute with the appropriate name.\n        '
        if item == 'from':
            item = 'from_user'
        try:
            return getattr(self, item)
        except AttributeError as exc:
            raise KeyError(f"Objects of type {self.__class__.__name__} don't have an attribute called `{item}`.") from exc

    def __getstate__(self) -> Dict[str, Union[str, object]]:
        if False:
            return 10
        "\n        Overrides :meth:`object.__getstate__` to customize the pickling process of objects of this\n        type.\n        The returned state does `not` contain the :class:`telegram.Bot` instance set with\n        :meth:`set_bot` (if any), as it can't be pickled.\n\n        Returns:\n            state (Dict[:obj:`str`, :obj:`object`]): The state of the object.\n        "
        out = self._get_attrs(include_private=True, recursive=False, remove_bot=True)
        out['api_kwargs'] = dict(self.api_kwargs)
        return out

    def __setstate__(self, state: Dict[str, object]) -> None:
        if False:
            for i in range(10):
                print('nop')
        '\n        Overrides :meth:`object.__setstate__` to customize the unpickling process of objects of\n        this type. Modifies the object in-place.\n\n        If any data was stored in the :attr:`api_kwargs` of the pickled object, this method checks\n        if the class now has dedicated attributes for those keys and moves the values from\n        :attr:`api_kwargs` to the dedicated attributes.\n        This can happen, if serialized data is loaded with a new version of this library, where\n        the new version was updated to account for updates of the Telegram Bot API.\n\n        If on the contrary an attribute was removed from the class, the value is not discarded but\n        made available via :attr:`api_kwargs`.\n\n        Args:\n            state (:obj:`dict`): The data to set as attributes of this object.\n        '
        self._unfreeze()
        self._bot = None
        api_kwargs = cast(Dict[str, object], state.pop('api_kwargs', {}))
        frozen = state.pop('_frozen', False)
        for (key, val) in state.items():
            try:
                setattr(self, key, val)
            except AttributeError:
                api_kwargs[key] = val
        self._apply_api_kwargs(api_kwargs)
        self.api_kwargs = MappingProxyType(api_kwargs)
        if frozen:
            self._freeze()

    def __deepcopy__(self: Tele_co, memodict: Dict[int, object]) -> Tele_co:
        if False:
            print('Hello World!')
        '\n        Customizes how :func:`copy.deepcopy` processes objects of this type.\n        The only difference to the default implementation is that the :class:`telegram.Bot`\n        instance set via :meth:`set_bot` (if any) is not copied, but shared between the original\n        and the copy, i.e.::\n\n            assert telegram_object.get_bot() is copy.deepcopy(telegram_object).get_bot()\n\n        Args:\n            memodict (:obj:`dict`): A dictionary that maps objects to their copies.\n\n        Returns:\n            :obj:`telegram.TelegramObject`: The copied object.\n        '
        bot = self._bot
        self.set_bot(None)
        cls = self.__class__
        result = cls.__new__(cls)
        memodict[id(self)] = result
        result._frozen = False
        for k in self._get_attrs_names(include_private=True):
            if k == '_frozen':
                continue
            if k == 'api_kwargs':
                setattr(result, k, MappingProxyType(deepcopy(dict(self.api_kwargs), memodict)))
                continue
            try:
                setattr(result, k, deepcopy(getattr(self, k), memodict))
            except AttributeError:
                continue
        if self._frozen:
            result._freeze()
        result.set_bot(bot)
        self.set_bot(bot)
        return result

    @staticmethod
    def _parse_data(data: Optional[JSONDict]) -> Optional[JSONDict]:
        if False:
            i = 10
            return i + 15
        'Should be called by subclasses that override de_json to ensure that the input\n        is not altered. Whoever calls de_json might still want to use the original input\n        for something else.\n        '
        return None if data is None else data.copy()

    @classmethod
    def _de_json(cls: Type[Tele_co], data: Optional[JSONDict], bot: 'Bot', api_kwargs: Optional[JSONDict]=None) -> Optional[Tele_co]:
        if False:
            i = 10
            return i + 15
        if data is None:
            return None
        try:
            obj = cls(**data, api_kwargs=api_kwargs)
        except TypeError as exc:
            if '__init__() got an unexpected keyword argument' not in str(exc):
                raise exc
            if cls.__INIT_PARAMS_CHECK is not cls:
                signature = inspect.signature(cls)
                cls.__INIT_PARAMS = set(signature.parameters.keys())
                cls.__INIT_PARAMS_CHECK = cls
            api_kwargs = api_kwargs or {}
            existing_kwargs: JSONDict = {}
            for (key, value) in data.items():
                (existing_kwargs if key in cls.__INIT_PARAMS else api_kwargs)[key] = value
            obj = cls(api_kwargs=api_kwargs, **existing_kwargs)
        obj.set_bot(bot=bot)
        return obj

    @classmethod
    def de_json(cls: Type[Tele_co], data: Optional[JSONDict], bot: 'Bot') -> Optional[Tele_co]:
        if False:
            i = 10
            return i + 15
        'Converts JSON data to a Telegram object.\n\n        Args:\n            data (Dict[:obj:`str`, ...]): The JSON data.\n            bot (:class:`telegram.Bot`): The bot associated with this object.\n\n        Returns:\n            The Telegram object.\n\n        '
        return cls._de_json(data=data, bot=bot)

    @classmethod
    def de_list(cls: Type[Tele_co], data: Optional[List[JSONDict]], bot: 'Bot') -> Tuple[Tele_co, ...]:
        if False:
            return 10
        'Converts a list of JSON objects to a tuple of Telegram objects.\n\n        .. versionchanged:: 20.0\n\n           * Returns a tuple instead of a list.\n           * Filters out any :obj:`None` values.\n\n        Args:\n            data (List[Dict[:obj:`str`, ...]]): The JSON data.\n            bot (:class:`telegram.Bot`): The bot associated with these objects.\n\n        Returns:\n            A tuple of Telegram objects.\n\n        '
        if not data:
            return ()
        return tuple((obj for obj in (cls.de_json(d, bot) for d in data) if obj is not None))

    @contextmanager
    def _unfrozen(self: Tele_co) -> Iterator[Tele_co]:
        if False:
            print('Hello World!')
        'Context manager to temporarily unfreeze the object. For internal use only.\n\n        Note:\n            with to._unfrozen() as other_to:\n                assert to is other_to\n        '
        self._unfreeze()
        yield self
        self._freeze()

    def _freeze(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        self._frozen = True

    def _unfreeze(self) -> None:
        if False:
            i = 10
            return i + 15
        self._frozen = False

    def _apply_api_kwargs(self, api_kwargs: JSONDict) -> None:
        if False:
            i = 10
            return i + 15
        'Loops through the api kwargs and for every key that exists as attribute of the\n        object (and is None), it moves the value from `api_kwargs` to the attribute.\n        *Edits `api_kwargs` in place!*\n\n        This method is currently only called in the unpickling process, i.e. not on "normal" init.\n        This is because\n        * automating this is tricky to get right: It should be called at the *end* of the __init__,\n          preferably only once at the end of the __init__ of the last child class. This could be\n          done via __init_subclass__, but it\'s hard to not destroy the signature of __init__ in the\n          process.\n        * calling it manually in every __init__ is tedious\n        * There probably is no use case for it anyway. If you manually initialize a TO subclass,\n          then you can pass everything as proper argument.\n        '
        for key in list(api_kwargs.keys()):
            if getattr(self, key, True) is None:
                setattr(self, key, api_kwargs.pop(key))

    def _get_attrs_names(self, include_private: bool) -> Iterator[str]:
        if False:
            i = 10
            return i + 15
        '\n        Returns the names of the attributes of this object. This is used to determine which\n        attributes should be serialized when pickling the object.\n\n        Args:\n            include_private (:obj:`bool`): Whether to include private attributes.\n\n        Returns:\n            Iterator[:obj:`str`]: An iterator over the names of the attributes of this object.\n        '
        all_slots = (s for c in self.__class__.__mro__[:-1] for s in c.__slots__)
        all_attrs = chain(all_slots, self.__dict__.keys()) if hasattr(self, '__dict__') else all_slots
        if include_private:
            return all_attrs
        return (attr for attr in all_attrs if not attr.startswith('_'))

    def _get_attrs(self, include_private: bool=False, recursive: bool=False, remove_bot: bool=False) -> Dict[str, Union[str, object]]:
        if False:
            print('Hello World!')
        'This method is used for obtaining the attributes of the object.\n\n        Args:\n            include_private (:obj:`bool`): Whether the result should include private variables.\n            recursive (:obj:`bool`): If :obj:`True`, will convert any ``TelegramObjects`` (if\n                found) in the attributes to a dictionary. Else, preserves it as an object itself.\n            remove_bot (:obj:`bool`): Whether the bot should be included in the result.\n\n        Returns:\n            :obj:`dict`: A dict where the keys are attribute names and values are their values.\n        '
        data = {}
        for key in self._get_attrs_names(include_private=include_private):
            value = getattr(self, key, None)
            if value is not None:
                if recursive and hasattr(value, 'to_dict'):
                    data[key] = value.to_dict(recursive=True)
                else:
                    data[key] = value
            elif not recursive:
                data[key] = value
        if recursive and data.get('from_user'):
            data['from'] = data.pop('from_user', None)
        if remove_bot:
            data.pop('_bot', None)
        return data

    def to_json(self) -> str:
        if False:
            while True:
                i = 10
        'Gives a JSON representation of object.\n\n        .. versionchanged:: 20.0\n            Now includes all entries of :attr:`api_kwargs`.\n\n        Returns:\n            :obj:`str`\n        '
        return json.dumps(self.to_dict())

    def to_dict(self, recursive: bool=True) -> JSONDict:
        if False:
            while True:
                i = 10
        'Gives representation of object as :obj:`dict`.\n\n        .. versionchanged:: 20.0\n\n            * Now includes all entries of :attr:`api_kwargs`.\n            * Attributes whose values are empty sequences are no longer included.\n\n        Args:\n            recursive (:obj:`bool`, optional): If :obj:`True`, will convert any TelegramObjects\n                (if found) in the attributes to a dictionary. Else, preserves it as an object\n                itself. Defaults to :obj:`True`.\n\n                .. versionadded:: 20.0\n\n        Returns:\n            :obj:`dict`\n        '
        out = self._get_attrs(recursive=recursive)
        pop_keys: Set[str] = set()
        for (key, value) in out.items():
            if isinstance(value, (tuple, list)):
                if not value:
                    pop_keys.add(key)
                    continue
                val = []
                for item in value:
                    if hasattr(item, 'to_dict'):
                        val.append(item.to_dict(recursive=recursive))
                    elif isinstance(item, (tuple, list)):
                        val.append([i.to_dict(recursive=recursive) if hasattr(i, 'to_dict') else i for i in item])
                    else:
                        val.append(item)
                out[key] = val
            elif isinstance(value, datetime.datetime):
                out[key] = to_timestamp(value)
        for key in pop_keys:
            out.pop(key)
        out.update(out.pop('api_kwargs', {}))
        return out

    def get_bot(self) -> 'Bot':
        if False:
            for i in range(10):
                print('nop')
        'Returns the :class:`telegram.Bot` instance associated with this object.\n\n        .. seealso:: :meth:`set_bot`\n\n        .. versionadded: 20.0\n\n        Raises:\n            RuntimeError: If no :class:`telegram.Bot` instance was set for this object.\n        '
        if self._bot is None:
            raise RuntimeError('This object has no bot associated with it. Shortcuts cannot be used.')
        return self._bot

    def set_bot(self, bot: Optional['Bot']) -> None:
        if False:
            while True:
                i = 10
        'Sets the :class:`telegram.Bot` instance associated with this object.\n\n        .. seealso:: :meth:`get_bot`\n\n        .. versionadded: 20.0\n\n        Arguments:\n            bot (:class:`telegram.Bot` | :obj:`None`): The bot instance.\n        '
        self._bot = bot