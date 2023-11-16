"""This module contains the auxiliary class ContextTypes."""
from typing import Any, Dict, Generic, Type, overload
from telegram.ext._callbackcontext import CallbackContext
from telegram.ext._extbot import ExtBot
from telegram.ext._utils.types import BD, CCT, CD, UD
ADict = Dict[Any, Any]

class ContextTypes(Generic[CCT, UD, CD, BD]):
    """
    Convenience class to gather customizable types of the :class:`telegram.ext.CallbackContext`
    interface.

    Examples:
        :any:`ContextTypes Bot <examples.contexttypesbot>`

    .. seealso:: :wiki:`Architecture Overview <Architecture>`,
        :wiki:`Storing Bot, User and Chat Related Data <Storing-bot%2C-user-and-chat-related-data>`

    .. versionadded:: 13.6

    Args:
        context (:obj:`type`, optional): Determines the type of the ``context`` argument of all
            (error-)handler callbacks and job callbacks. Must be a subclass of
            :class:`telegram.ext.CallbackContext`. Defaults to
            :class:`telegram.ext.CallbackContext`.
        bot_data (:obj:`type`, optional): Determines the type of
            :attr:`context.bot_data <CallbackContext.bot_data>` of all (error-)handler callbacks
            and job callbacks. Defaults to :obj:`dict`. Must support instantiating without
            arguments.
        chat_data (:obj:`type`, optional): Determines the type of
            :attr:`context.chat_data <CallbackContext.chat_data>` of all (error-)handler callbacks
            and job callbacks. Defaults to :obj:`dict`. Must support instantiating without
            arguments.
        user_data (:obj:`type`, optional): Determines the type of
            :attr:`context.user_data <CallbackContext.user_data>` of all (error-)handler callbacks
            and job callbacks. Defaults to :obj:`dict`. Must support instantiating without
            arguments.

    """
    DEFAULT_TYPE = CallbackContext[ExtBot[None], ADict, ADict, ADict]
    "Shortcut for the type annotation for the ``context`` argument that's correct for the\n    default settings, i.e. if :class:`telegram.ext.ContextTypes` is not used.\n\n    Example:\n        .. code:: python\n\n            async def callback(update: Update, context: ContextTypes.DEFAULT_TYPE):\n                ...\n\n    .. versionadded: 20.0\n    "
    __slots__ = ('_context', '_bot_data', '_chat_data', '_user_data')

    @overload
    def __init__(self: 'ContextTypes[CallbackContext[ExtBot[Any], ADict, ADict, ADict], ADict, ADict, ADict]'):
        if False:
            print('Hello World!')
        ...

    @overload
    def __init__(self: 'ContextTypes[CCT, ADict, ADict, ADict]', context: Type[CCT]):
        if False:
            return 10
        ...

    @overload
    def __init__(self: 'ContextTypes[CallbackContext[ExtBot[Any], UD, ADict, ADict], UD, ADict, ADict]', user_data: Type[UD]):
        if False:
            for i in range(10):
                print('nop')
        ...

    @overload
    def __init__(self: 'ContextTypes[CallbackContext[ExtBot[Any], ADict, CD, ADict], ADict, CD, ADict]', chat_data: Type[CD]):
        if False:
            print('Hello World!')
        ...

    @overload
    def __init__(self: 'ContextTypes[CallbackContext[ExtBot[Any], ADict, ADict, BD], ADict, ADict, BD]', bot_data: Type[BD]):
        if False:
            print('Hello World!')
        ...

    @overload
    def __init__(self: 'ContextTypes[CCT, UD, ADict, ADict]', context: Type[CCT], user_data: Type[UD]):
        if False:
            return 10
        ...

    @overload
    def __init__(self: 'ContextTypes[CCT, ADict, CD, ADict]', context: Type[CCT], chat_data: Type[CD]):
        if False:
            return 10
        ...

    @overload
    def __init__(self: 'ContextTypes[CCT, ADict, ADict, BD]', context: Type[CCT], bot_data: Type[BD]):
        if False:
            while True:
                i = 10
        ...

    @overload
    def __init__(self: 'ContextTypes[CallbackContext[ExtBot[Any], UD, CD, ADict], UD, CD, ADict]', user_data: Type[UD], chat_data: Type[CD]):
        if False:
            while True:
                i = 10
        ...

    @overload
    def __init__(self: 'ContextTypes[CallbackContext[ExtBot[Any], UD, ADict, BD], UD, ADict, BD]', user_data: Type[UD], bot_data: Type[BD]):
        if False:
            return 10
        ...

    @overload
    def __init__(self: 'ContextTypes[CallbackContext[ExtBot[Any], ADict, CD, BD], ADict, CD, BD]', chat_data: Type[CD], bot_data: Type[BD]):
        if False:
            i = 10
            return i + 15
        ...

    @overload
    def __init__(self: 'ContextTypes[CCT, UD, CD, ADict]', context: Type[CCT], user_data: Type[UD], chat_data: Type[CD]):
        if False:
            for i in range(10):
                print('nop')
        ...

    @overload
    def __init__(self: 'ContextTypes[CCT, UD, ADict, BD]', context: Type[CCT], user_data: Type[UD], bot_data: Type[BD]):
        if False:
            return 10
        ...

    @overload
    def __init__(self: 'ContextTypes[CCT, ADict, CD, BD]', context: Type[CCT], chat_data: Type[CD], bot_data: Type[BD]):
        if False:
            while True:
                i = 10
        ...

    @overload
    def __init__(self: 'ContextTypes[CallbackContext[ExtBot[Any], UD, CD, BD], UD, CD, BD]', user_data: Type[UD], chat_data: Type[CD], bot_data: Type[BD]):
        if False:
            for i in range(10):
                print('nop')
        ...

    @overload
    def __init__(self: 'ContextTypes[CCT, UD, CD, BD]', context: Type[CCT], user_data: Type[UD], chat_data: Type[CD], bot_data: Type[BD]):
        if False:
            print('Hello World!')
        ...

    def __init__(self, context: 'Type[CallbackContext[ExtBot[Any], ADict, ADict, ADict]]'=CallbackContext, bot_data: Type[ADict]=dict, chat_data: Type[ADict]=dict, user_data: Type[ADict]=dict):
        if False:
            while True:
                i = 10
        if not issubclass(context, CallbackContext):
            raise ValueError('context must be a subclass of CallbackContext.')
        self._context = context
        self._bot_data = bot_data
        self._chat_data = chat_data
        self._user_data = user_data

    @property
    def context(self) -> Type[CCT]:
        if False:
            return 10
        'The type of the ``context`` argument of all (error-)handler callbacks and job\n        callbacks.\n        '
        return self._context

    @property
    def bot_data(self) -> Type[BD]:
        if False:
            for i in range(10):
                print('nop')
        'The type of :attr:`context.bot_data <CallbackContext.bot_data>` of all (error-)handler\n        callbacks and job callbacks.\n        '
        return self._bot_data

    @property
    def chat_data(self) -> Type[CD]:
        if False:
            for i in range(10):
                print('nop')
        'The type of :attr:`context.chat_data <CallbackContext.chat_data>` of all (error-)handler\n        callbacks and job callbacks.\n        '
        return self._chat_data

    @property
    def user_data(self) -> Type[UD]:
        if False:
            i = 10
            return i + 15
        'The type of :attr:`context.user_data <CallbackContext.user_data>` of all (error-)handler\n        callbacks and job callbacks.\n        '
        return self._user_data