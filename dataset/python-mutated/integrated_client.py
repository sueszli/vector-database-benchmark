from .client import SAMPClient
from .hub_proxy import SAMPHubProxy
__all__ = ['SAMPIntegratedClient']
__doctest_skip__ = ['SAMPIntegratedClient.*']

class SAMPIntegratedClient:
    """
    A Simple SAMP client.

    This class is meant to simplify the client usage providing a proxy class
    that merges the :class:`~astropy.samp.SAMPClient` and
    :class:`~astropy.samp.SAMPHubProxy` functionalities in a
    simplified API.

    Parameters
    ----------
    name : str, optional
        Client name (corresponding to ``samp.name`` metadata keyword).

    description : str, optional
        Client description (corresponding to ``samp.description.text`` metadata
        keyword).

    metadata : dict, optional
        Client application metadata in the standard SAMP format.

    addr : str, optional
        Listening address (or IP). This defaults to 127.0.0.1 if the internet
        is not reachable, otherwise it defaults to the host name.

    port : int, optional
        Listening XML-RPC server socket port. If left set to 0 (the default),
        the operating system will select a free port.

    callable : bool, optional
        Whether the client can receive calls and notifications. If set to
        `False`, then the client can send notifications and calls, but can not
        receive any.
    """

    def __init__(self, name=None, description=None, metadata=None, addr=None, port=0, callable=True):
        if False:
            return 10
        self.hub = SAMPHubProxy()
        self.client_arguments = {'name': name, 'description': description, 'metadata': metadata, 'addr': addr, 'port': port, 'callable': callable}
        '\n        Collected arguments that should be passed on to the SAMPClient below.\n        The SAMPClient used to be instantiated in __init__; however, this\n        caused problems with disconnecting and reconnecting to the HUB.\n        The client_arguments is used to maintain backwards compatibility.\n        '
        self.client = None
        'The client will be instantiated upon connect().'

    @property
    def is_connected(self):
        if False:
            return 10
        '\n        Testing method to verify the client connection with a running Hub.\n\n        Returns\n        -------\n        is_connected : bool\n            True if the client is connected to a Hub, False otherwise.\n        '
        return self.hub.is_connected and self.client.is_running

    def connect(self, hub=None, hub_params=None, pool_size=20):
        if False:
            print('Hello World!')
        '\n        Connect with the current or specified SAMP Hub, start and register the\n        client.\n\n        Parameters\n        ----------\n        hub : `~astropy.samp.SAMPHubServer`, optional\n            The hub to connect to.\n\n        hub_params : dict, optional\n            Optional dictionary containing the lock-file content of the Hub\n            with which to connect. This dictionary has the form\n            ``{<token-name>: <token-string>, ...}``.\n\n        pool_size : int, optional\n            The number of socket connections opened to communicate with the\n            Hub.\n        '
        self.hub.connect(hub, hub_params, pool_size)
        self.client = SAMPClient(self.hub, **self.client_arguments)
        self.client.start()
        self.client.register()

    def disconnect(self):
        if False:
            while True:
                i = 10
        '\n        Unregister the client from the current SAMP Hub, stop the client and\n        disconnect from the Hub.\n        '
        if self.is_connected:
            try:
                self.client.unregister()
            finally:
                if self.client.is_running:
                    self.client.stop()
                self.hub.disconnect()

    def ping(self):
        if False:
            print('Hello World!')
        '\n        Proxy to ``ping`` SAMP Hub method (Standard Profile only).\n        '
        return self.hub.ping()

    def declare_metadata(self, metadata):
        if False:
            while True:
                i = 10
        '\n        Proxy to ``declareMetadata`` SAMP Hub method.\n        '
        return self.client.declare_metadata(metadata)

    def get_metadata(self, client_id):
        if False:
            while True:
                i = 10
        '\n        Proxy to ``getMetadata`` SAMP Hub method.\n        '
        return self.hub.get_metadata(self.get_private_key(), client_id)

    def get_subscriptions(self, client_id):
        if False:
            return 10
        '\n        Proxy to ``getSubscriptions`` SAMP Hub method.\n        '
        return self.hub.get_subscriptions(self.get_private_key(), client_id)

    def get_registered_clients(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Proxy to ``getRegisteredClients`` SAMP Hub method.\n\n        This returns all the registered clients, excluding the current client.\n        '
        return self.hub.get_registered_clients(self.get_private_key())

    def get_subscribed_clients(self, mtype):
        if False:
            print('Hello World!')
        '\n        Proxy to ``getSubscribedClients`` SAMP Hub method.\n        '
        return self.hub.get_subscribed_clients(self.get_private_key(), mtype)

    def _format_easy_msg(self, mtype, params):
        if False:
            for i in range(10):
                print('nop')
        msg = {}
        if 'extra_kws' in params:
            extra = params['extra_kws']
            del params['extra_kws']
            msg = {'samp.mtype': mtype, 'samp.params': params}
            msg.update(extra)
        else:
            msg = {'samp.mtype': mtype, 'samp.params': params}
        return msg

    def notify(self, recipient_id, message):
        if False:
            i = 10
            return i + 15
        '\n        Proxy to ``notify`` SAMP Hub method.\n        '
        return self.hub.notify(self.get_private_key(), recipient_id, message)

    def enotify(self, recipient_id, mtype, **params):
        if False:
            return 10
        '\n        Easy to use version of :meth:`~astropy.samp.integrated_client.SAMPIntegratedClient.notify`.\n\n        This is a proxy to ``notify`` method that allows to send the\n        notification message in a simplified way.\n\n        Note that reserved ``extra_kws`` keyword is a dictionary with the\n        special meaning of being used to add extra keywords, in addition to\n        the standard ``samp.mtype`` and ``samp.params``, to the message sent.\n\n        Parameters\n        ----------\n        recipient_id : str\n            Recipient ID\n\n        mtype : str\n            the MType to be notified\n\n        params : dict or set of str\n            Variable keyword set which contains the list of parameters for the\n            specified MType.\n\n        Examples\n        --------\n        >>> from astropy.samp import SAMPIntegratedClient\n        >>> cli = SAMPIntegratedClient()\n        >>> ...\n        >>> cli.enotify("samp.msg.progress", msgid = "xyz", txt = "initialization",\n        ...             percent = "10", extra_kws = {"my.extra.info": "just an example"})\n        '
        return self.notify(recipient_id, self._format_easy_msg(mtype, params))

    def notify_all(self, message):
        if False:
            i = 10
            return i + 15
        '\n        Proxy to ``notifyAll`` SAMP Hub method.\n        '
        return self.hub.notify_all(self.get_private_key(), message)

    def enotify_all(self, mtype, **params):
        if False:
            while True:
                i = 10
        '\n        Easy to use version of :meth:`~astropy.samp.integrated_client.SAMPIntegratedClient.notify_all`.\n\n        This is a proxy to ``notifyAll`` method that allows to send the\n        notification message in a simplified way.\n\n        Note that reserved ``extra_kws`` keyword is a dictionary with the\n        special meaning of being used to add extra keywords, in addition to\n        the standard ``samp.mtype`` and ``samp.params``, to the message sent.\n\n        Parameters\n        ----------\n        mtype : str\n            MType to be notified.\n\n        params : dict or set of str\n            Variable keyword set which contains the list of parameters for\n            the specified MType.\n\n        Examples\n        --------\n        >>> from astropy.samp import SAMPIntegratedClient\n        >>> cli = SAMPIntegratedClient()\n        >>> ...\n        >>> cli.enotify_all("samp.msg.progress", txt = "initialization",\n        ...                 percent = "10",\n        ...                 extra_kws = {"my.extra.info": "just an example"})\n        '
        return self.notify_all(self._format_easy_msg(mtype, params))

    def call(self, recipient_id, msg_tag, message):
        if False:
            while True:
                i = 10
        '\n        Proxy to ``call`` SAMP Hub method.\n        '
        return self.hub.call(self.get_private_key(), recipient_id, msg_tag, message)

    def ecall(self, recipient_id, msg_tag, mtype, **params):
        if False:
            return 10
        '\n        Easy to use version of :meth:`~astropy.samp.integrated_client.SAMPIntegratedClient.call`.\n\n        This is a proxy to ``call`` method that allows to send a call message\n        in a simplified way.\n\n        Note that reserved ``extra_kws`` keyword is a dictionary with the\n        special meaning of being used to add extra keywords, in addition to\n        the standard ``samp.mtype`` and ``samp.params``, to the message sent.\n\n        Parameters\n        ----------\n        recipient_id : str\n            Recipient ID\n\n        msg_tag : str\n            Message tag to use\n\n        mtype : str\n            MType to be sent\n\n        params : dict of set of str\n            Variable keyword set which contains the list of parameters for\n            the specified MType.\n\n        Examples\n        --------\n        >>> from astropy.samp import SAMPIntegratedClient\n        >>> cli = SAMPIntegratedClient()\n        >>> ...\n        >>> msgid = cli.ecall("abc", "xyz", "samp.msg.progress",\n        ...                   txt = "initialization", percent = "10",\n        ...                   extra_kws = {"my.extra.info": "just an example"})\n        '
        return self.call(recipient_id, msg_tag, self._format_easy_msg(mtype, params))

    def call_all(self, msg_tag, message):
        if False:
            return 10
        '\n        Proxy to ``callAll`` SAMP Hub method.\n        '
        return self.hub.call_all(self.get_private_key(), msg_tag, message)

    def ecall_all(self, msg_tag, mtype, **params):
        if False:
            print('Hello World!')
        '\n        Easy to use version of :meth:`~astropy.samp.integrated_client.SAMPIntegratedClient.call_all`.\n\n        This is a proxy to ``callAll`` method that allows to send the call\n        message in a simplified way.\n\n        Note that reserved ``extra_kws`` keyword is a dictionary with the\n        special meaning of being used to add extra keywords, in addition to\n        the standard ``samp.mtype`` and ``samp.params``, to the message sent.\n\n        Parameters\n        ----------\n        msg_tag : str\n            Message tag to use\n\n        mtype : str\n            MType to be sent\n\n        params : dict of set of str\n            Variable keyword set which contains the list of parameters for\n            the specified MType.\n\n        Examples\n        --------\n        >>> from astropy.samp import SAMPIntegratedClient\n        >>> cli = SAMPIntegratedClient()\n        >>> ...\n        >>> msgid = cli.ecall_all("xyz", "samp.msg.progress",\n        ...                       txt = "initialization", percent = "10",\n        ...                       extra_kws = {"my.extra.info": "just an example"})\n        '
        self.call_all(msg_tag, self._format_easy_msg(mtype, params))

    def call_and_wait(self, recipient_id, message, timeout):
        if False:
            for i in range(10):
                print('nop')
        '\n        Proxy to ``callAndWait`` SAMP Hub method.\n        '
        return self.hub.call_and_wait(self.get_private_key(), recipient_id, message, timeout)

    def ecall_and_wait(self, recipient_id, mtype, timeout, **params):
        if False:
            print('Hello World!')
        '\n        Easy to use version of :meth:`~astropy.samp.integrated_client.SAMPIntegratedClient.call_and_wait`.\n\n        This is a proxy to ``callAndWait`` method that allows to send the call\n        message in a simplified way.\n\n        Note that reserved ``extra_kws`` keyword is a dictionary with the\n        special meaning of being used to add extra keywords, in addition to\n        the standard ``samp.mtype`` and ``samp.params``, to the message sent.\n\n        Parameters\n        ----------\n        recipient_id : str\n            Recipient ID\n\n        mtype : str\n            MType to be sent\n\n        timeout : str\n            Call timeout in seconds\n\n        params : dict of set of str\n            Variable keyword set which contains the list of parameters for\n            the specified MType.\n\n        Examples\n        --------\n        >>> from astropy.samp import SAMPIntegratedClient\n        >>> cli = SAMPIntegratedClient()\n        >>> ...\n        >>> cli.ecall_and_wait("xyz", "samp.msg.progress", "5",\n        ...                    txt = "initialization", percent = "10",\n        ...                    extra_kws = {"my.extra.info": "just an example"})\n        '
        return self.call_and_wait(recipient_id, self._format_easy_msg(mtype, params), timeout)

    def reply(self, msg_id, response):
        if False:
            while True:
                i = 10
        '\n        Proxy to ``reply`` SAMP Hub method.\n        '
        return self.hub.reply(self.get_private_key(), msg_id, response)

    def _format_easy_response(self, status, result, error):
        if False:
            return 10
        msg = {'samp.status': status}
        if result is not None:
            msg.update({'samp.result': result})
        if error is not None:
            msg.update({'samp.error': error})
        return msg

    def ereply(self, msg_id, status, result=None, error=None):
        if False:
            for i in range(10):
                print('nop')
        '\n        Easy to use version of :meth:`~astropy.samp.integrated_client.SAMPIntegratedClient.reply`.\n\n        This is a proxy to ``reply`` method that allows to send a reply\n        message in a simplified way.\n\n        Parameters\n        ----------\n        msg_id : str\n            Message ID to which reply.\n\n        status : str\n            Content of the ``samp.status`` response keyword.\n\n        result : dict\n            Content of the ``samp.result`` response keyword.\n\n        error : dict\n            Content of the ``samp.error`` response keyword.\n\n        Examples\n        --------\n        >>> from astropy.samp import SAMPIntegratedClient, SAMP_STATUS_ERROR\n        >>> cli = SAMPIntegratedClient()\n        >>> ...\n        >>> cli.ereply("abd", SAMP_STATUS_ERROR, result={},\n        ...            error={"samp.errortxt": "Test error message"})\n        '
        return self.reply(msg_id, self._format_easy_response(status, result, error))

    def receive_notification(self, private_key, sender_id, message):
        if False:
            print('Hello World!')
        return self.client.receive_notification(private_key, sender_id, message)
    receive_notification.__doc__ = SAMPClient.receive_notification.__doc__

    def receive_call(self, private_key, sender_id, msg_id, message):
        if False:
            while True:
                i = 10
        return self.client.receive_call(private_key, sender_id, msg_id, message)
    receive_call.__doc__ = SAMPClient.receive_call.__doc__

    def receive_response(self, private_key, responder_id, msg_tag, response):
        if False:
            for i in range(10):
                print('nop')
        return self.client.receive_response(private_key, responder_id, msg_tag, response)
    receive_response.__doc__ = SAMPClient.receive_response.__doc__

    def bind_receive_message(self, mtype, function, declare=True, metadata=None):
        if False:
            i = 10
            return i + 15
        self.client.bind_receive_message(mtype, function, declare=True, metadata=None)
    bind_receive_message.__doc__ = SAMPClient.bind_receive_message.__doc__

    def bind_receive_notification(self, mtype, function, declare=True, metadata=None):
        if False:
            print('Hello World!')
        self.client.bind_receive_notification(mtype, function, declare, metadata)
    bind_receive_notification.__doc__ = SAMPClient.bind_receive_notification.__doc__

    def bind_receive_call(self, mtype, function, declare=True, metadata=None):
        if False:
            return 10
        self.client.bind_receive_call(mtype, function, declare, metadata)
    bind_receive_call.__doc__ = SAMPClient.bind_receive_call.__doc__

    def bind_receive_response(self, msg_tag, function):
        if False:
            while True:
                i = 10
        self.client.bind_receive_response(msg_tag, function)
    bind_receive_response.__doc__ = SAMPClient.bind_receive_response.__doc__

    def unbind_receive_notification(self, mtype, declare=True):
        if False:
            print('Hello World!')
        self.client.unbind_receive_notification(mtype, declare)
    unbind_receive_notification.__doc__ = SAMPClient.unbind_receive_notification.__doc__

    def unbind_receive_call(self, mtype, declare=True):
        if False:
            i = 10
            return i + 15
        self.client.unbind_receive_call(mtype, declare)
    unbind_receive_call.__doc__ = SAMPClient.unbind_receive_call.__doc__

    def unbind_receive_response(self, msg_tag):
        if False:
            return 10
        self.client.unbind_receive_response(msg_tag)
    unbind_receive_response.__doc__ = SAMPClient.unbind_receive_response.__doc__

    def declare_subscriptions(self, subscriptions=None):
        if False:
            i = 10
            return i + 15
        self.client.declare_subscriptions(subscriptions)
    declare_subscriptions.__doc__ = SAMPClient.declare_subscriptions.__doc__

    def get_private_key(self):
        if False:
            i = 10
            return i + 15
        return self.client.get_private_key()
    get_private_key.__doc__ = SAMPClient.get_private_key.__doc__

    def get_public_id(self):
        if False:
            return 10
        return self.client.get_public_id()
    get_public_id.__doc__ = SAMPClient.get_public_id.__doc__