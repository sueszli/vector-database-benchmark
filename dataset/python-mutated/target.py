from __future__ import annotations
import enum
import typing
from dataclasses import dataclass
import streamlink.webbrowser.cdp.devtools.browser as browser
import streamlink.webbrowser.cdp.devtools.page as page
from streamlink.webbrowser.cdp.devtools.util import T_JSON_DICT, event_class

class TargetID(str):

    def to_json(self) -> str:
        if False:
            while True:
                i = 10
        return self

    @classmethod
    def from_json(cls, json: str) -> TargetID:
        if False:
            i = 10
            return i + 15
        return cls(json)

    def __repr__(self):
        if False:
            return 10
        return f'TargetID({super().__repr__()})'

class SessionID(str):
    """
    Unique identifier of attached debugging session.
    """

    def to_json(self) -> str:
        if False:
            i = 10
            return i + 15
        return self

    @classmethod
    def from_json(cls, json: str) -> SessionID:
        if False:
            i = 10
            return i + 15
        return cls(json)

    def __repr__(self):
        if False:
            for i in range(10):
                print('nop')
        return f'SessionID({super().__repr__()})'

@dataclass
class TargetInfo:
    target_id: TargetID
    type_: str
    title: str
    url: str
    attached: bool
    can_access_opener: bool
    opener_id: typing.Optional[TargetID] = None
    opener_frame_id: typing.Optional[page.FrameId] = None
    browser_context_id: typing.Optional[browser.BrowserContextID] = None
    subtype: typing.Optional[str] = None

    def to_json(self) -> T_JSON_DICT:
        if False:
            print('Hello World!')
        json: T_JSON_DICT = {}
        json['targetId'] = self.target_id.to_json()
        json['type'] = self.type_
        json['title'] = self.title
        json['url'] = self.url
        json['attached'] = self.attached
        json['canAccessOpener'] = self.can_access_opener
        if self.opener_id is not None:
            json['openerId'] = self.opener_id.to_json()
        if self.opener_frame_id is not None:
            json['openerFrameId'] = self.opener_frame_id.to_json()
        if self.browser_context_id is not None:
            json['browserContextId'] = self.browser_context_id.to_json()
        if self.subtype is not None:
            json['subtype'] = self.subtype
        return json

    @classmethod
    def from_json(cls, json: T_JSON_DICT) -> TargetInfo:
        if False:
            return 10
        return cls(target_id=TargetID.from_json(json['targetId']), type_=str(json['type']), title=str(json['title']), url=str(json['url']), attached=bool(json['attached']), can_access_opener=bool(json['canAccessOpener']), opener_id=TargetID.from_json(json['openerId']) if 'openerId' in json else None, opener_frame_id=page.FrameId.from_json(json['openerFrameId']) if 'openerFrameId' in json else None, browser_context_id=browser.BrowserContextID.from_json(json['browserContextId']) if 'browserContextId' in json else None, subtype=str(json['subtype']) if 'subtype' in json else None)

@dataclass
class FilterEntry:
    """
    A filter used by target query/discovery/auto-attach operations.
    """
    exclude: typing.Optional[bool] = None
    type_: typing.Optional[str] = None

    def to_json(self) -> T_JSON_DICT:
        if False:
            for i in range(10):
                print('nop')
        json: T_JSON_DICT = {}
        if self.exclude is not None:
            json['exclude'] = self.exclude
        if self.type_ is not None:
            json['type'] = self.type_
        return json

    @classmethod
    def from_json(cls, json: T_JSON_DICT) -> FilterEntry:
        if False:
            print('Hello World!')
        return cls(exclude=bool(json['exclude']) if 'exclude' in json else None, type_=str(json['type']) if 'type' in json else None)

class TargetFilter(list):
    """
    The entries in TargetFilter are matched sequentially against targets and
    the first entry that matches determines if the target is included or not,
    depending on the value of ``exclude`` field in the entry.
    If filter is not specified, the one assumed is
    [{type: "browser", exclude: true}, {type: "tab", exclude: true}, {}]
    (i.e. include everything but ``browser`` and ``tab``).
    """

    def to_json(self) -> typing.List[FilterEntry]:
        if False:
            return 10
        return self

    @classmethod
    def from_json(cls, json: typing.List[FilterEntry]) -> TargetFilter:
        if False:
            i = 10
            return i + 15
        return cls(json)

    def __repr__(self):
        if False:
            i = 10
            return i + 15
        return f'TargetFilter({super().__repr__()})'

@dataclass
class RemoteLocation:
    host: str
    port: int

    def to_json(self) -> T_JSON_DICT:
        if False:
            return 10
        json: T_JSON_DICT = {}
        json['host'] = self.host
        json['port'] = self.port
        return json

    @classmethod
    def from_json(cls, json: T_JSON_DICT) -> RemoteLocation:
        if False:
            i = 10
            return i + 15
        return cls(host=str(json['host']), port=int(json['port']))

def activate_target(target_id: TargetID) -> typing.Generator[T_JSON_DICT, T_JSON_DICT, None]:
    if False:
        i = 10
        return i + 15
    '\n    Activates (focuses) the target.\n\n    :param target_id:\n    '
    params: T_JSON_DICT = {}
    params['targetId'] = target_id.to_json()
    cmd_dict: T_JSON_DICT = {'method': 'Target.activateTarget', 'params': params}
    yield cmd_dict

def attach_to_target(target_id: TargetID, flatten: typing.Optional[bool]=None) -> typing.Generator[T_JSON_DICT, T_JSON_DICT, SessionID]:
    if False:
        while True:
            i = 10
    '\n    Attaches to the target with given id.\n\n    :param target_id:\n    :param flatten: *(Optional)* Enables "flat" access to the session via specifying sessionId attribute in the commands. We plan to make this the default, deprecate non-flattened mode, and eventually retire it. See crbug.com/991325.\n    :returns: Id assigned to the session.\n    '
    params: T_JSON_DICT = {}
    params['targetId'] = target_id.to_json()
    if flatten is not None:
        params['flatten'] = flatten
    cmd_dict: T_JSON_DICT = {'method': 'Target.attachToTarget', 'params': params}
    json = (yield cmd_dict)
    return SessionID.from_json(json['sessionId'])

def attach_to_browser_target() -> typing.Generator[T_JSON_DICT, T_JSON_DICT, SessionID]:
    if False:
        return 10
    '\n    Attaches to the browser target, only uses flat sessionId mode.\n\n    **EXPERIMENTAL**\n\n    :returns: Id assigned to the session.\n    '
    cmd_dict: T_JSON_DICT = {'method': 'Target.attachToBrowserTarget'}
    json = (yield cmd_dict)
    return SessionID.from_json(json['sessionId'])

def close_target(target_id: TargetID) -> typing.Generator[T_JSON_DICT, T_JSON_DICT, bool]:
    if False:
        print('Hello World!')
    '\n    Closes the target. If the target is a page that gets closed too.\n\n    :param target_id:\n    :returns: Always set to true. If an error occurs, the response indicates protocol error.\n    '
    params: T_JSON_DICT = {}
    params['targetId'] = target_id.to_json()
    cmd_dict: T_JSON_DICT = {'method': 'Target.closeTarget', 'params': params}
    json = (yield cmd_dict)
    return bool(json['success'])

def expose_dev_tools_protocol(target_id: TargetID, binding_name: typing.Optional[str]=None) -> typing.Generator[T_JSON_DICT, T_JSON_DICT, None]:
    if False:
        while True:
            i = 10
    "\n    Inject object to the target's main frame that provides a communication\n    channel with browser target.\n\n    Injected object will be available as ``window[bindingName]``.\n\n    The object has the follwing API:\n    - ``binding.send(json)`` - a method to send messages over the remote debugging protocol\n    - ``binding.onmessage = json => handleMessage(json)`` - a callback that will be called for the protocol notifications and command responses.\n\n    **EXPERIMENTAL**\n\n    :param target_id:\n    :param binding_name: *(Optional)* Binding name, 'cdp' if not specified.\n    "
    params: T_JSON_DICT = {}
    params['targetId'] = target_id.to_json()
    if binding_name is not None:
        params['bindingName'] = binding_name
    cmd_dict: T_JSON_DICT = {'method': 'Target.exposeDevToolsProtocol', 'params': params}
    yield cmd_dict

def create_browser_context(dispose_on_detach: typing.Optional[bool]=None, proxy_server: typing.Optional[str]=None, proxy_bypass_list: typing.Optional[str]=None, origins_with_universal_network_access: typing.Optional[typing.List[str]]=None) -> typing.Generator[T_JSON_DICT, T_JSON_DICT, browser.BrowserContextID]:
    if False:
        for i in range(10):
            print('nop')
    '\n    Creates a new empty BrowserContext. Similar to an incognito profile but you can have more than\n    one.\n\n    **EXPERIMENTAL**\n\n    :param dispose_on_detach: *(Optional)* If specified, disposes this context when debugging session disconnects.\n    :param proxy_server: *(Optional)* Proxy server, similar to the one passed to --proxy-server\n    :param proxy_bypass_list: *(Optional)* Proxy bypass list, similar to the one passed to --proxy-bypass-list\n    :param origins_with_universal_network_access: *(Optional)* An optional list of origins to grant unlimited cross-origin access to. Parts of the URL other than those constituting origin are ignored.\n    :returns: The id of the context created.\n    '
    params: T_JSON_DICT = {}
    if dispose_on_detach is not None:
        params['disposeOnDetach'] = dispose_on_detach
    if proxy_server is not None:
        params['proxyServer'] = proxy_server
    if proxy_bypass_list is not None:
        params['proxyBypassList'] = proxy_bypass_list
    if origins_with_universal_network_access is not None:
        params['originsWithUniversalNetworkAccess'] = list(origins_with_universal_network_access)
    cmd_dict: T_JSON_DICT = {'method': 'Target.createBrowserContext', 'params': params}
    json = (yield cmd_dict)
    return browser.BrowserContextID.from_json(json['browserContextId'])

def get_browser_contexts() -> typing.Generator[T_JSON_DICT, T_JSON_DICT, typing.List[browser.BrowserContextID]]:
    if False:
        i = 10
        return i + 15
    '\n    Returns all browser contexts created with ``Target.createBrowserContext`` method.\n\n    **EXPERIMENTAL**\n\n    :returns: An array of browser context ids.\n    '
    cmd_dict: T_JSON_DICT = {'method': 'Target.getBrowserContexts'}
    json = (yield cmd_dict)
    return [browser.BrowserContextID.from_json(i) for i in json['browserContextIds']]

def create_target(url: str, width: typing.Optional[int]=None, height: typing.Optional[int]=None, browser_context_id: typing.Optional[browser.BrowserContextID]=None, enable_begin_frame_control: typing.Optional[bool]=None, new_window: typing.Optional[bool]=None, background: typing.Optional[bool]=None, for_tab: typing.Optional[bool]=None) -> typing.Generator[T_JSON_DICT, T_JSON_DICT, TargetID]:
    if False:
        i = 10
        return i + 15
    '\n    Creates a new page.\n\n    :param url: The initial URL the page will be navigated to. An empty string indicates about:blank.\n    :param width: *(Optional)* Frame width in DIP (headless chrome only).\n    :param height: *(Optional)* Frame height in DIP (headless chrome only).\n    :param browser_context_id: **(EXPERIMENTAL)** *(Optional)* The browser context to create the page in.\n    :param enable_begin_frame_control: **(EXPERIMENTAL)** *(Optional)* Whether BeginFrames for this target will be controlled via DevTools (headless chrome only, not supported on MacOS yet, false by default).\n    :param new_window: *(Optional)* Whether to create a new Window or Tab (chrome-only, false by default).\n    :param background: *(Optional)* Whether to create the target in background or foreground (chrome-only, false by default).\n    :param for_tab: **(EXPERIMENTAL)** *(Optional)* Whether to create the target of type "tab".\n    :returns: The id of the page opened.\n    '
    params: T_JSON_DICT = {}
    params['url'] = url
    if width is not None:
        params['width'] = width
    if height is not None:
        params['height'] = height
    if browser_context_id is not None:
        params['browserContextId'] = browser_context_id.to_json()
    if enable_begin_frame_control is not None:
        params['enableBeginFrameControl'] = enable_begin_frame_control
    if new_window is not None:
        params['newWindow'] = new_window
    if background is not None:
        params['background'] = background
    if for_tab is not None:
        params['forTab'] = for_tab
    cmd_dict: T_JSON_DICT = {'method': 'Target.createTarget', 'params': params}
    json = (yield cmd_dict)
    return TargetID.from_json(json['targetId'])

def detach_from_target(session_id: typing.Optional[SessionID]=None, target_id: typing.Optional[TargetID]=None) -> typing.Generator[T_JSON_DICT, T_JSON_DICT, None]:
    if False:
        for i in range(10):
            print('nop')
    '\n    Detaches session with given id.\n\n    :param session_id: *(Optional)* Session to detach.\n    :param target_id: *(Optional)* Deprecated.\n    '
    params: T_JSON_DICT = {}
    if session_id is not None:
        params['sessionId'] = session_id.to_json()
    if target_id is not None:
        params['targetId'] = target_id.to_json()
    cmd_dict: T_JSON_DICT = {'method': 'Target.detachFromTarget', 'params': params}
    yield cmd_dict

def dispose_browser_context(browser_context_id: browser.BrowserContextID) -> typing.Generator[T_JSON_DICT, T_JSON_DICT, None]:
    if False:
        i = 10
        return i + 15
    '\n    Deletes a BrowserContext. All the belonging pages will be closed without calling their\n    beforeunload hooks.\n\n    **EXPERIMENTAL**\n\n    :param browser_context_id:\n    '
    params: T_JSON_DICT = {}
    params['browserContextId'] = browser_context_id.to_json()
    cmd_dict: T_JSON_DICT = {'method': 'Target.disposeBrowserContext', 'params': params}
    yield cmd_dict

def get_target_info(target_id: typing.Optional[TargetID]=None) -> typing.Generator[T_JSON_DICT, T_JSON_DICT, TargetInfo]:
    if False:
        print('Hello World!')
    '\n    Returns information about a target.\n\n    **EXPERIMENTAL**\n\n    :param target_id: *(Optional)*\n    :returns:\n    '
    params: T_JSON_DICT = {}
    if target_id is not None:
        params['targetId'] = target_id.to_json()
    cmd_dict: T_JSON_DICT = {'method': 'Target.getTargetInfo', 'params': params}
    json = (yield cmd_dict)
    return TargetInfo.from_json(json['targetInfo'])

def get_targets(filter_: typing.Optional[TargetFilter]=None) -> typing.Generator[T_JSON_DICT, T_JSON_DICT, typing.List[TargetInfo]]:
    if False:
        return 10
    '\n    Retrieves a list of available targets.\n\n    :param filter_: **(EXPERIMENTAL)** *(Optional)* Only targets matching filter will be reported. If filter is not specified and target discovery is currently enabled, a filter used for target discovery is used for consistency.\n    :returns: The list of targets.\n    '
    params: T_JSON_DICT = {}
    if filter_ is not None:
        params['filter'] = filter_.to_json()
    cmd_dict: T_JSON_DICT = {'method': 'Target.getTargets', 'params': params}
    json = (yield cmd_dict)
    return [TargetInfo.from_json(i) for i in json['targetInfos']]

def send_message_to_target(message: str, session_id: typing.Optional[SessionID]=None, target_id: typing.Optional[TargetID]=None) -> typing.Generator[T_JSON_DICT, T_JSON_DICT, None]:
    if False:
        for i in range(10):
            print('nop')
    '\n    Sends protocol message over session with given id.\n    Consider using flat mode instead; see commands attachToTarget, setAutoAttach,\n    and crbug.com/991325.\n\n    :param message:\n    :param session_id: *(Optional)* Identifier of the session.\n    :param target_id: *(Optional)* Deprecated.\n    '
    params: T_JSON_DICT = {}
    params['message'] = message
    if session_id is not None:
        params['sessionId'] = session_id.to_json()
    if target_id is not None:
        params['targetId'] = target_id.to_json()
    cmd_dict: T_JSON_DICT = {'method': 'Target.sendMessageToTarget', 'params': params}
    yield cmd_dict

def set_auto_attach(auto_attach: bool, wait_for_debugger_on_start: bool, flatten: typing.Optional[bool]=None, filter_: typing.Optional[TargetFilter]=None) -> typing.Generator[T_JSON_DICT, T_JSON_DICT, None]:
    if False:
        return 10
    '\n    Controls whether to automatically attach to new targets which are considered to be related to\n    this one. When turned on, attaches to all existing related targets as well. When turned off,\n    automatically detaches from all currently attached targets.\n    This also clears all targets added by ``autoAttachRelated`` from the list of targets to watch\n    for creation of related targets.\n\n    **EXPERIMENTAL**\n\n    :param auto_attach: Whether to auto-attach to related targets.\n    :param wait_for_debugger_on_start: Whether to pause new targets when attaching to them. Use ```Runtime.runIfWaitingForDebugger``` to run paused targets.\n    :param flatten: *(Optional)* Enables "flat" access to the session via specifying sessionId attribute in the commands. We plan to make this the default, deprecate non-flattened mode, and eventually retire it. See crbug.com/991325.\n    :param filter_: **(EXPERIMENTAL)** *(Optional)* Only targets matching filter will be attached.\n    '
    params: T_JSON_DICT = {}
    params['autoAttach'] = auto_attach
    params['waitForDebuggerOnStart'] = wait_for_debugger_on_start
    if flatten is not None:
        params['flatten'] = flatten
    if filter_ is not None:
        params['filter'] = filter_.to_json()
    cmd_dict: T_JSON_DICT = {'method': 'Target.setAutoAttach', 'params': params}
    yield cmd_dict

def auto_attach_related(target_id: TargetID, wait_for_debugger_on_start: bool, filter_: typing.Optional[TargetFilter]=None) -> typing.Generator[T_JSON_DICT, T_JSON_DICT, None]:
    if False:
        print('Hello World!')
    '\n    Adds the specified target to the list of targets that will be monitored for any related target\n    creation (such as child frames, child workers and new versions of service worker) and reported\n    through ``attachedToTarget``. The specified target is also auto-attached.\n    This cancels the effect of any previous ``setAutoAttach`` and is also cancelled by subsequent\n    ``setAutoAttach``. Only available at the Browser target.\n\n    **EXPERIMENTAL**\n\n    :param target_id:\n    :param wait_for_debugger_on_start: Whether to pause new targets when attaching to them. Use ```Runtime.runIfWaitingForDebugger``` to run paused targets.\n    :param filter_: **(EXPERIMENTAL)** *(Optional)* Only targets matching filter will be attached.\n    '
    params: T_JSON_DICT = {}
    params['targetId'] = target_id.to_json()
    params['waitForDebuggerOnStart'] = wait_for_debugger_on_start
    if filter_ is not None:
        params['filter'] = filter_.to_json()
    cmd_dict: T_JSON_DICT = {'method': 'Target.autoAttachRelated', 'params': params}
    yield cmd_dict

def set_discover_targets(discover: bool, filter_: typing.Optional[TargetFilter]=None) -> typing.Generator[T_JSON_DICT, T_JSON_DICT, None]:
    if False:
        print('Hello World!')
    '\n    Controls whether to discover available targets and notify via\n    ``targetCreated/targetInfoChanged/targetDestroyed`` events.\n\n    :param discover: Whether to discover available targets.\n    :param filter_: **(EXPERIMENTAL)** *(Optional)* Only targets matching filter will be attached. If ```discover```` is false, ````filter``` must be omitted or empty.\n    '
    params: T_JSON_DICT = {}
    params['discover'] = discover
    if filter_ is not None:
        params['filter'] = filter_.to_json()
    cmd_dict: T_JSON_DICT = {'method': 'Target.setDiscoverTargets', 'params': params}
    yield cmd_dict

def set_remote_locations(locations: typing.List[RemoteLocation]) -> typing.Generator[T_JSON_DICT, T_JSON_DICT, None]:
    if False:
        for i in range(10):
            print('nop')
    '\n    Enables target discovery for the specified locations, when ``setDiscoverTargets`` was set to\n    ``true``.\n\n    **EXPERIMENTAL**\n\n    :param locations: List of remote locations.\n    '
    params: T_JSON_DICT = {}
    params['locations'] = [i.to_json() for i in locations]
    cmd_dict: T_JSON_DICT = {'method': 'Target.setRemoteLocations', 'params': params}
    yield cmd_dict

@event_class('Target.attachedToTarget')
@dataclass
class AttachedToTarget:
    """
    **EXPERIMENTAL**

    Issued when attached to target because of auto-attach or ``attachToTarget`` command.
    """
    session_id: SessionID
    target_info: TargetInfo
    waiting_for_debugger: bool

    @classmethod
    def from_json(cls, json: T_JSON_DICT) -> AttachedToTarget:
        if False:
            return 10
        return cls(session_id=SessionID.from_json(json['sessionId']), target_info=TargetInfo.from_json(json['targetInfo']), waiting_for_debugger=bool(json['waitingForDebugger']))

@event_class('Target.detachedFromTarget')
@dataclass
class DetachedFromTarget:
    """
    **EXPERIMENTAL**

    Issued when detached from target for any reason (including ``detachFromTarget`` command). Can be
    issued multiple times per target if multiple sessions have been attached to it.
    """
    session_id: SessionID
    target_id: typing.Optional[TargetID]

    @classmethod
    def from_json(cls, json: T_JSON_DICT) -> DetachedFromTarget:
        if False:
            while True:
                i = 10
        return cls(session_id=SessionID.from_json(json['sessionId']), target_id=TargetID.from_json(json['targetId']) if 'targetId' in json else None)

@event_class('Target.receivedMessageFromTarget')
@dataclass
class ReceivedMessageFromTarget:
    """
    Notifies about a new protocol message received from the session (as reported in
    ``attachedToTarget`` event).
    """
    session_id: SessionID
    message: str
    target_id: typing.Optional[TargetID]

    @classmethod
    def from_json(cls, json: T_JSON_DICT) -> ReceivedMessageFromTarget:
        if False:
            return 10
        return cls(session_id=SessionID.from_json(json['sessionId']), message=str(json['message']), target_id=TargetID.from_json(json['targetId']) if 'targetId' in json else None)

@event_class('Target.targetCreated')
@dataclass
class TargetCreated:
    """
    Issued when a possible inspection target is created.
    """
    target_info: TargetInfo

    @classmethod
    def from_json(cls, json: T_JSON_DICT) -> TargetCreated:
        if False:
            i = 10
            return i + 15
        return cls(target_info=TargetInfo.from_json(json['targetInfo']))

@event_class('Target.targetDestroyed')
@dataclass
class TargetDestroyed:
    """
    Issued when a target is destroyed.
    """
    target_id: TargetID

    @classmethod
    def from_json(cls, json: T_JSON_DICT) -> TargetDestroyed:
        if False:
            return 10
        return cls(target_id=TargetID.from_json(json['targetId']))

@event_class('Target.targetCrashed')
@dataclass
class TargetCrashed:
    """
    Issued when a target has crashed.
    """
    target_id: TargetID
    status: str
    error_code: int

    @classmethod
    def from_json(cls, json: T_JSON_DICT) -> TargetCrashed:
        if False:
            for i in range(10):
                print('nop')
        return cls(target_id=TargetID.from_json(json['targetId']), status=str(json['status']), error_code=int(json['errorCode']))

@event_class('Target.targetInfoChanged')
@dataclass
class TargetInfoChanged:
    """
    Issued when some information about a target has changed. This only happens between
    ``targetCreated`` and ``targetDestroyed``.
    """
    target_info: TargetInfo

    @classmethod
    def from_json(cls, json: T_JSON_DICT) -> TargetInfoChanged:
        if False:
            for i in range(10):
                print('nop')
        return cls(target_info=TargetInfo.from_json(json['targetInfo']))