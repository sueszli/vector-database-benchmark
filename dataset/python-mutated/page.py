from __future__ import annotations
import enum
import typing
from dataclasses import dataclass
import streamlink.webbrowser.cdp.devtools.debugger as debugger
import streamlink.webbrowser.cdp.devtools.dom as dom
import streamlink.webbrowser.cdp.devtools.emulation as emulation
import streamlink.webbrowser.cdp.devtools.io as io
import streamlink.webbrowser.cdp.devtools.network as network
import streamlink.webbrowser.cdp.devtools.runtime as runtime
from streamlink.webbrowser.cdp.devtools.util import T_JSON_DICT, event_class

class FrameId(str):
    """
    Unique frame identifier.
    """

    def to_json(self) -> str:
        if False:
            while True:
                i = 10
        return self

    @classmethod
    def from_json(cls, json: str) -> FrameId:
        if False:
            return 10
        return cls(json)

    def __repr__(self):
        if False:
            print('Hello World!')
        return f'FrameId({super().__repr__()})'

class AdFrameType(enum.Enum):
    """
    Indicates whether a frame has been identified as an ad.
    """
    NONE = 'none'
    CHILD = 'child'
    ROOT = 'root'

    def to_json(self) -> str:
        if False:
            i = 10
            return i + 15
        return self.value

    @classmethod
    def from_json(cls, json: str) -> AdFrameType:
        if False:
            while True:
                i = 10
        return cls(json)

class AdFrameExplanation(enum.Enum):
    PARENT_IS_AD = 'ParentIsAd'
    CREATED_BY_AD_SCRIPT = 'CreatedByAdScript'
    MATCHED_BLOCKING_RULE = 'MatchedBlockingRule'

    def to_json(self) -> str:
        if False:
            while True:
                i = 10
        return self.value

    @classmethod
    def from_json(cls, json: str) -> AdFrameExplanation:
        if False:
            while True:
                i = 10
        return cls(json)

@dataclass
class AdFrameStatus:
    """
    Indicates whether a frame has been identified as an ad and why.
    """
    ad_frame_type: AdFrameType
    explanations: typing.Optional[typing.List[AdFrameExplanation]] = None

    def to_json(self) -> T_JSON_DICT:
        if False:
            while True:
                i = 10
        json: T_JSON_DICT = {}
        json['adFrameType'] = self.ad_frame_type.to_json()
        if self.explanations is not None:
            json['explanations'] = [i.to_json() for i in self.explanations]
        return json

    @classmethod
    def from_json(cls, json: T_JSON_DICT) -> AdFrameStatus:
        if False:
            return 10
        return cls(ad_frame_type=AdFrameType.from_json(json['adFrameType']), explanations=[AdFrameExplanation.from_json(i) for i in json['explanations']] if 'explanations' in json else None)

@dataclass
class AdScriptId:
    """
    Identifies the bottom-most script which caused the frame to be labelled
    as an ad.
    """
    script_id: runtime.ScriptId
    debugger_id: runtime.UniqueDebuggerId

    def to_json(self) -> T_JSON_DICT:
        if False:
            return 10
        json: T_JSON_DICT = {}
        json['scriptId'] = self.script_id.to_json()
        json['debuggerId'] = self.debugger_id.to_json()
        return json

    @classmethod
    def from_json(cls, json: T_JSON_DICT) -> AdScriptId:
        if False:
            i = 10
            return i + 15
        return cls(script_id=runtime.ScriptId.from_json(json['scriptId']), debugger_id=runtime.UniqueDebuggerId.from_json(json['debuggerId']))

class SecureContextType(enum.Enum):
    """
    Indicates whether the frame is a secure context and why it is the case.
    """
    SECURE = 'Secure'
    SECURE_LOCALHOST = 'SecureLocalhost'
    INSECURE_SCHEME = 'InsecureScheme'
    INSECURE_ANCESTOR = 'InsecureAncestor'

    def to_json(self) -> str:
        if False:
            print('Hello World!')
        return self.value

    @classmethod
    def from_json(cls, json: str) -> SecureContextType:
        if False:
            while True:
                i = 10
        return cls(json)

class CrossOriginIsolatedContextType(enum.Enum):
    """
    Indicates whether the frame is cross-origin isolated and why it is the case.
    """
    ISOLATED = 'Isolated'
    NOT_ISOLATED = 'NotIsolated'
    NOT_ISOLATED_FEATURE_DISABLED = 'NotIsolatedFeatureDisabled'

    def to_json(self) -> str:
        if False:
            return 10
        return self.value

    @classmethod
    def from_json(cls, json: str) -> CrossOriginIsolatedContextType:
        if False:
            for i in range(10):
                print('nop')
        return cls(json)

class GatedAPIFeatures(enum.Enum):
    SHARED_ARRAY_BUFFERS = 'SharedArrayBuffers'
    SHARED_ARRAY_BUFFERS_TRANSFER_ALLOWED = 'SharedArrayBuffersTransferAllowed'
    PERFORMANCE_MEASURE_MEMORY = 'PerformanceMeasureMemory'
    PERFORMANCE_PROFILE = 'PerformanceProfile'

    def to_json(self) -> str:
        if False:
            while True:
                i = 10
        return self.value

    @classmethod
    def from_json(cls, json: str) -> GatedAPIFeatures:
        if False:
            for i in range(10):
                print('nop')
        return cls(json)

class PermissionsPolicyFeature(enum.Enum):
    """
    All Permissions Policy features. This enum should match the one defined
    in third_party/blink/renderer/core/permissions_policy/permissions_policy_features.json5.
    """
    ACCELEROMETER = 'accelerometer'
    AMBIENT_LIGHT_SENSOR = 'ambient-light-sensor'
    ATTRIBUTION_REPORTING = 'attribution-reporting'
    AUTOPLAY = 'autoplay'
    BLUETOOTH = 'bluetooth'
    BROWSING_TOPICS = 'browsing-topics'
    CAMERA = 'camera'
    CH_DPR = 'ch-dpr'
    CH_DEVICE_MEMORY = 'ch-device-memory'
    CH_DOWNLINK = 'ch-downlink'
    CH_ECT = 'ch-ect'
    CH_PREFERS_COLOR_SCHEME = 'ch-prefers-color-scheme'
    CH_PREFERS_REDUCED_MOTION = 'ch-prefers-reduced-motion'
    CH_RTT = 'ch-rtt'
    CH_SAVE_DATA = 'ch-save-data'
    CH_UA = 'ch-ua'
    CH_UA_ARCH = 'ch-ua-arch'
    CH_UA_BITNESS = 'ch-ua-bitness'
    CH_UA_PLATFORM = 'ch-ua-platform'
    CH_UA_MODEL = 'ch-ua-model'
    CH_UA_MOBILE = 'ch-ua-mobile'
    CH_UA_FULL_VERSION = 'ch-ua-full-version'
    CH_UA_FULL_VERSION_LIST = 'ch-ua-full-version-list'
    CH_UA_PLATFORM_VERSION = 'ch-ua-platform-version'
    CH_UA_WOW64 = 'ch-ua-wow64'
    CH_VIEWPORT_HEIGHT = 'ch-viewport-height'
    CH_VIEWPORT_WIDTH = 'ch-viewport-width'
    CH_WIDTH = 'ch-width'
    CLIPBOARD_READ = 'clipboard-read'
    CLIPBOARD_WRITE = 'clipboard-write'
    COMPUTE_PRESSURE = 'compute-pressure'
    CROSS_ORIGIN_ISOLATED = 'cross-origin-isolated'
    DIRECT_SOCKETS = 'direct-sockets'
    DISPLAY_CAPTURE = 'display-capture'
    DOCUMENT_DOMAIN = 'document-domain'
    ENCRYPTED_MEDIA = 'encrypted-media'
    EXECUTION_WHILE_OUT_OF_VIEWPORT = 'execution-while-out-of-viewport'
    EXECUTION_WHILE_NOT_RENDERED = 'execution-while-not-rendered'
    FOCUS_WITHOUT_USER_ACTIVATION = 'focus-without-user-activation'
    FULLSCREEN = 'fullscreen'
    FROBULATE = 'frobulate'
    GAMEPAD = 'gamepad'
    GEOLOCATION = 'geolocation'
    GYROSCOPE = 'gyroscope'
    HID = 'hid'
    IDENTITY_CREDENTIALS_GET = 'identity-credentials-get'
    IDLE_DETECTION = 'idle-detection'
    INTEREST_COHORT = 'interest-cohort'
    JOIN_AD_INTEREST_GROUP = 'join-ad-interest-group'
    KEYBOARD_MAP = 'keyboard-map'
    LOCAL_FONTS = 'local-fonts'
    MAGNETOMETER = 'magnetometer'
    MICROPHONE = 'microphone'
    MIDI = 'midi'
    OTP_CREDENTIALS = 'otp-credentials'
    PAYMENT = 'payment'
    PICTURE_IN_PICTURE = 'picture-in-picture'
    PRIVATE_AGGREGATION = 'private-aggregation'
    PRIVATE_STATE_TOKEN_ISSUANCE = 'private-state-token-issuance'
    PRIVATE_STATE_TOKEN_REDEMPTION = 'private-state-token-redemption'
    PUBLICKEY_CREDENTIALS_GET = 'publickey-credentials-get'
    RUN_AD_AUCTION = 'run-ad-auction'
    SCREEN_WAKE_LOCK = 'screen-wake-lock'
    SERIAL = 'serial'
    SHARED_AUTOFILL = 'shared-autofill'
    SHARED_STORAGE = 'shared-storage'
    SHARED_STORAGE_SELECT_URL = 'shared-storage-select-url'
    SMART_CARD = 'smart-card'
    STORAGE_ACCESS = 'storage-access'
    SYNC_XHR = 'sync-xhr'
    UNLOAD = 'unload'
    USB = 'usb'
    VERTICAL_SCROLL = 'vertical-scroll'
    WEB_SHARE = 'web-share'
    WINDOW_MANAGEMENT = 'window-management'
    WINDOW_PLACEMENT = 'window-placement'
    XR_SPATIAL_TRACKING = 'xr-spatial-tracking'

    def to_json(self) -> str:
        if False:
            i = 10
            return i + 15
        return self.value

    @classmethod
    def from_json(cls, json: str) -> PermissionsPolicyFeature:
        if False:
            return 10
        return cls(json)

class PermissionsPolicyBlockReason(enum.Enum):
    """
    Reason for a permissions policy feature to be disabled.
    """
    HEADER = 'Header'
    IFRAME_ATTRIBUTE = 'IframeAttribute'
    IN_FENCED_FRAME_TREE = 'InFencedFrameTree'
    IN_ISOLATED_APP = 'InIsolatedApp'

    def to_json(self) -> str:
        if False:
            i = 10
            return i + 15
        return self.value

    @classmethod
    def from_json(cls, json: str) -> PermissionsPolicyBlockReason:
        if False:
            i = 10
            return i + 15
        return cls(json)

@dataclass
class PermissionsPolicyBlockLocator:
    frame_id: FrameId
    block_reason: PermissionsPolicyBlockReason

    def to_json(self) -> T_JSON_DICT:
        if False:
            for i in range(10):
                print('nop')
        json: T_JSON_DICT = {}
        json['frameId'] = self.frame_id.to_json()
        json['blockReason'] = self.block_reason.to_json()
        return json

    @classmethod
    def from_json(cls, json: T_JSON_DICT) -> PermissionsPolicyBlockLocator:
        if False:
            while True:
                i = 10
        return cls(frame_id=FrameId.from_json(json['frameId']), block_reason=PermissionsPolicyBlockReason.from_json(json['blockReason']))

@dataclass
class PermissionsPolicyFeatureState:
    feature: PermissionsPolicyFeature
    allowed: bool
    locator: typing.Optional[PermissionsPolicyBlockLocator] = None

    def to_json(self) -> T_JSON_DICT:
        if False:
            for i in range(10):
                print('nop')
        json: T_JSON_DICT = {}
        json['feature'] = self.feature.to_json()
        json['allowed'] = self.allowed
        if self.locator is not None:
            json['locator'] = self.locator.to_json()
        return json

    @classmethod
    def from_json(cls, json: T_JSON_DICT) -> PermissionsPolicyFeatureState:
        if False:
            for i in range(10):
                print('nop')
        return cls(feature=PermissionsPolicyFeature.from_json(json['feature']), allowed=bool(json['allowed']), locator=PermissionsPolicyBlockLocator.from_json(json['locator']) if 'locator' in json else None)

class OriginTrialTokenStatus(enum.Enum):
    """
    Origin Trial(https://www.chromium.org/blink/origin-trials) support.
    Status for an Origin Trial token.
    """
    SUCCESS = 'Success'
    NOT_SUPPORTED = 'NotSupported'
    INSECURE = 'Insecure'
    EXPIRED = 'Expired'
    WRONG_ORIGIN = 'WrongOrigin'
    INVALID_SIGNATURE = 'InvalidSignature'
    MALFORMED = 'Malformed'
    WRONG_VERSION = 'WrongVersion'
    FEATURE_DISABLED = 'FeatureDisabled'
    TOKEN_DISABLED = 'TokenDisabled'
    FEATURE_DISABLED_FOR_USER = 'FeatureDisabledForUser'
    UNKNOWN_TRIAL = 'UnknownTrial'

    def to_json(self) -> str:
        if False:
            while True:
                i = 10
        return self.value

    @classmethod
    def from_json(cls, json: str) -> OriginTrialTokenStatus:
        if False:
            while True:
                i = 10
        return cls(json)

class OriginTrialStatus(enum.Enum):
    """
    Status for an Origin Trial.
    """
    ENABLED = 'Enabled'
    VALID_TOKEN_NOT_PROVIDED = 'ValidTokenNotProvided'
    OS_NOT_SUPPORTED = 'OSNotSupported'
    TRIAL_NOT_ALLOWED = 'TrialNotAllowed'

    def to_json(self) -> str:
        if False:
            while True:
                i = 10
        return self.value

    @classmethod
    def from_json(cls, json: str) -> OriginTrialStatus:
        if False:
            print('Hello World!')
        return cls(json)

class OriginTrialUsageRestriction(enum.Enum):
    NONE = 'None'
    SUBSET = 'Subset'

    def to_json(self) -> str:
        if False:
            print('Hello World!')
        return self.value

    @classmethod
    def from_json(cls, json: str) -> OriginTrialUsageRestriction:
        if False:
            for i in range(10):
                print('nop')
        return cls(json)

@dataclass
class OriginTrialToken:
    origin: str
    match_sub_domains: bool
    trial_name: str
    expiry_time: network.TimeSinceEpoch
    is_third_party: bool
    usage_restriction: OriginTrialUsageRestriction

    def to_json(self) -> T_JSON_DICT:
        if False:
            print('Hello World!')
        json: T_JSON_DICT = {}
        json['origin'] = self.origin
        json['matchSubDomains'] = self.match_sub_domains
        json['trialName'] = self.trial_name
        json['expiryTime'] = self.expiry_time.to_json()
        json['isThirdParty'] = self.is_third_party
        json['usageRestriction'] = self.usage_restriction.to_json()
        return json

    @classmethod
    def from_json(cls, json: T_JSON_DICT) -> OriginTrialToken:
        if False:
            print('Hello World!')
        return cls(origin=str(json['origin']), match_sub_domains=bool(json['matchSubDomains']), trial_name=str(json['trialName']), expiry_time=network.TimeSinceEpoch.from_json(json['expiryTime']), is_third_party=bool(json['isThirdParty']), usage_restriction=OriginTrialUsageRestriction.from_json(json['usageRestriction']))

@dataclass
class OriginTrialTokenWithStatus:
    raw_token_text: str
    status: OriginTrialTokenStatus
    parsed_token: typing.Optional[OriginTrialToken] = None

    def to_json(self) -> T_JSON_DICT:
        if False:
            return 10
        json: T_JSON_DICT = {}
        json['rawTokenText'] = self.raw_token_text
        json['status'] = self.status.to_json()
        if self.parsed_token is not None:
            json['parsedToken'] = self.parsed_token.to_json()
        return json

    @classmethod
    def from_json(cls, json: T_JSON_DICT) -> OriginTrialTokenWithStatus:
        if False:
            while True:
                i = 10
        return cls(raw_token_text=str(json['rawTokenText']), status=OriginTrialTokenStatus.from_json(json['status']), parsed_token=OriginTrialToken.from_json(json['parsedToken']) if 'parsedToken' in json else None)

@dataclass
class OriginTrial:
    trial_name: str
    status: OriginTrialStatus
    tokens_with_status: typing.List[OriginTrialTokenWithStatus]

    def to_json(self) -> T_JSON_DICT:
        if False:
            print('Hello World!')
        json: T_JSON_DICT = {}
        json['trialName'] = self.trial_name
        json['status'] = self.status.to_json()
        json['tokensWithStatus'] = [i.to_json() for i in self.tokens_with_status]
        return json

    @classmethod
    def from_json(cls, json: T_JSON_DICT) -> OriginTrial:
        if False:
            for i in range(10):
                print('nop')
        return cls(trial_name=str(json['trialName']), status=OriginTrialStatus.from_json(json['status']), tokens_with_status=[OriginTrialTokenWithStatus.from_json(i) for i in json['tokensWithStatus']])

@dataclass
class Frame:
    """
    Information about the Frame on the page.
    """
    id_: FrameId
    loader_id: network.LoaderId
    url: str
    domain_and_registry: str
    security_origin: str
    mime_type: str
    secure_context_type: SecureContextType
    cross_origin_isolated_context_type: CrossOriginIsolatedContextType
    gated_api_features: typing.List[GatedAPIFeatures]
    parent_id: typing.Optional[FrameId] = None
    name: typing.Optional[str] = None
    url_fragment: typing.Optional[str] = None
    unreachable_url: typing.Optional[str] = None
    ad_frame_status: typing.Optional[AdFrameStatus] = None

    def to_json(self) -> T_JSON_DICT:
        if False:
            return 10
        json: T_JSON_DICT = {}
        json['id'] = self.id_.to_json()
        json['loaderId'] = self.loader_id.to_json()
        json['url'] = self.url
        json['domainAndRegistry'] = self.domain_and_registry
        json['securityOrigin'] = self.security_origin
        json['mimeType'] = self.mime_type
        json['secureContextType'] = self.secure_context_type.to_json()
        json['crossOriginIsolatedContextType'] = self.cross_origin_isolated_context_type.to_json()
        json['gatedAPIFeatures'] = [i.to_json() for i in self.gated_api_features]
        if self.parent_id is not None:
            json['parentId'] = self.parent_id.to_json()
        if self.name is not None:
            json['name'] = self.name
        if self.url_fragment is not None:
            json['urlFragment'] = self.url_fragment
        if self.unreachable_url is not None:
            json['unreachableUrl'] = self.unreachable_url
        if self.ad_frame_status is not None:
            json['adFrameStatus'] = self.ad_frame_status.to_json()
        return json

    @classmethod
    def from_json(cls, json: T_JSON_DICT) -> Frame:
        if False:
            return 10
        return cls(id_=FrameId.from_json(json['id']), loader_id=network.LoaderId.from_json(json['loaderId']), url=str(json['url']), domain_and_registry=str(json['domainAndRegistry']), security_origin=str(json['securityOrigin']), mime_type=str(json['mimeType']), secure_context_type=SecureContextType.from_json(json['secureContextType']), cross_origin_isolated_context_type=CrossOriginIsolatedContextType.from_json(json['crossOriginIsolatedContextType']), gated_api_features=[GatedAPIFeatures.from_json(i) for i in json['gatedAPIFeatures']], parent_id=FrameId.from_json(json['parentId']) if 'parentId' in json else None, name=str(json['name']) if 'name' in json else None, url_fragment=str(json['urlFragment']) if 'urlFragment' in json else None, unreachable_url=str(json['unreachableUrl']) if 'unreachableUrl' in json else None, ad_frame_status=AdFrameStatus.from_json(json['adFrameStatus']) if 'adFrameStatus' in json else None)

@dataclass
class FrameResource:
    """
    Information about the Resource on the page.
    """
    url: str
    type_: network.ResourceType
    mime_type: str
    last_modified: typing.Optional[network.TimeSinceEpoch] = None
    content_size: typing.Optional[float] = None
    failed: typing.Optional[bool] = None
    canceled: typing.Optional[bool] = None

    def to_json(self) -> T_JSON_DICT:
        if False:
            return 10
        json: T_JSON_DICT = {}
        json['url'] = self.url
        json['type'] = self.type_.to_json()
        json['mimeType'] = self.mime_type
        if self.last_modified is not None:
            json['lastModified'] = self.last_modified.to_json()
        if self.content_size is not None:
            json['contentSize'] = self.content_size
        if self.failed is not None:
            json['failed'] = self.failed
        if self.canceled is not None:
            json['canceled'] = self.canceled
        return json

    @classmethod
    def from_json(cls, json: T_JSON_DICT) -> FrameResource:
        if False:
            return 10
        return cls(url=str(json['url']), type_=network.ResourceType.from_json(json['type']), mime_type=str(json['mimeType']), last_modified=network.TimeSinceEpoch.from_json(json['lastModified']) if 'lastModified' in json else None, content_size=float(json['contentSize']) if 'contentSize' in json else None, failed=bool(json['failed']) if 'failed' in json else None, canceled=bool(json['canceled']) if 'canceled' in json else None)

@dataclass
class FrameResourceTree:
    """
    Information about the Frame hierarchy along with their cached resources.
    """
    frame: Frame
    resources: typing.List[FrameResource]
    child_frames: typing.Optional[typing.List[FrameResourceTree]] = None

    def to_json(self) -> T_JSON_DICT:
        if False:
            while True:
                i = 10
        json: T_JSON_DICT = {}
        json['frame'] = self.frame.to_json()
        json['resources'] = [i.to_json() for i in self.resources]
        if self.child_frames is not None:
            json['childFrames'] = [i.to_json() for i in self.child_frames]
        return json

    @classmethod
    def from_json(cls, json: T_JSON_DICT) -> FrameResourceTree:
        if False:
            i = 10
            return i + 15
        return cls(frame=Frame.from_json(json['frame']), resources=[FrameResource.from_json(i) for i in json['resources']], child_frames=[FrameResourceTree.from_json(i) for i in json['childFrames']] if 'childFrames' in json else None)

@dataclass
class FrameTree:
    """
    Information about the Frame hierarchy.
    """
    frame: Frame
    child_frames: typing.Optional[typing.List[FrameTree]] = None

    def to_json(self) -> T_JSON_DICT:
        if False:
            return 10
        json: T_JSON_DICT = {}
        json['frame'] = self.frame.to_json()
        if self.child_frames is not None:
            json['childFrames'] = [i.to_json() for i in self.child_frames]
        return json

    @classmethod
    def from_json(cls, json: T_JSON_DICT) -> FrameTree:
        if False:
            print('Hello World!')
        return cls(frame=Frame.from_json(json['frame']), child_frames=[FrameTree.from_json(i) for i in json['childFrames']] if 'childFrames' in json else None)

class ScriptIdentifier(str):
    """
    Unique script identifier.
    """

    def to_json(self) -> str:
        if False:
            while True:
                i = 10
        return self

    @classmethod
    def from_json(cls, json: str) -> ScriptIdentifier:
        if False:
            for i in range(10):
                print('nop')
        return cls(json)

    def __repr__(self):
        if False:
            while True:
                i = 10
        return f'ScriptIdentifier({super().__repr__()})'

class TransitionType(enum.Enum):
    """
    Transition type.
    """
    LINK = 'link'
    TYPED = 'typed'
    ADDRESS_BAR = 'address_bar'
    AUTO_BOOKMARK = 'auto_bookmark'
    AUTO_SUBFRAME = 'auto_subframe'
    MANUAL_SUBFRAME = 'manual_subframe'
    GENERATED = 'generated'
    AUTO_TOPLEVEL = 'auto_toplevel'
    FORM_SUBMIT = 'form_submit'
    RELOAD = 'reload'
    KEYWORD = 'keyword'
    KEYWORD_GENERATED = 'keyword_generated'
    OTHER = 'other'

    def to_json(self) -> str:
        if False:
            for i in range(10):
                print('nop')
        return self.value

    @classmethod
    def from_json(cls, json: str) -> TransitionType:
        if False:
            print('Hello World!')
        return cls(json)

@dataclass
class NavigationEntry:
    """
    Navigation history entry.
    """
    id_: int
    url: str
    user_typed_url: str
    title: str
    transition_type: TransitionType

    def to_json(self) -> T_JSON_DICT:
        if False:
            return 10
        json: T_JSON_DICT = {}
        json['id'] = self.id_
        json['url'] = self.url
        json['userTypedURL'] = self.user_typed_url
        json['title'] = self.title
        json['transitionType'] = self.transition_type.to_json()
        return json

    @classmethod
    def from_json(cls, json: T_JSON_DICT) -> NavigationEntry:
        if False:
            for i in range(10):
                print('nop')
        return cls(id_=int(json['id']), url=str(json['url']), user_typed_url=str(json['userTypedURL']), title=str(json['title']), transition_type=TransitionType.from_json(json['transitionType']))

@dataclass
class ScreencastFrameMetadata:
    """
    Screencast frame metadata.
    """
    offset_top: float
    page_scale_factor: float
    device_width: float
    device_height: float
    scroll_offset_x: float
    scroll_offset_y: float
    timestamp: typing.Optional[network.TimeSinceEpoch] = None

    def to_json(self) -> T_JSON_DICT:
        if False:
            while True:
                i = 10
        json: T_JSON_DICT = {}
        json['offsetTop'] = self.offset_top
        json['pageScaleFactor'] = self.page_scale_factor
        json['deviceWidth'] = self.device_width
        json['deviceHeight'] = self.device_height
        json['scrollOffsetX'] = self.scroll_offset_x
        json['scrollOffsetY'] = self.scroll_offset_y
        if self.timestamp is not None:
            json['timestamp'] = self.timestamp.to_json()
        return json

    @classmethod
    def from_json(cls, json: T_JSON_DICT) -> ScreencastFrameMetadata:
        if False:
            i = 10
            return i + 15
        return cls(offset_top=float(json['offsetTop']), page_scale_factor=float(json['pageScaleFactor']), device_width=float(json['deviceWidth']), device_height=float(json['deviceHeight']), scroll_offset_x=float(json['scrollOffsetX']), scroll_offset_y=float(json['scrollOffsetY']), timestamp=network.TimeSinceEpoch.from_json(json['timestamp']) if 'timestamp' in json else None)

class DialogType(enum.Enum):
    """
    Javascript dialog type.
    """
    ALERT = 'alert'
    CONFIRM = 'confirm'
    PROMPT = 'prompt'
    BEFOREUNLOAD = 'beforeunload'

    def to_json(self) -> str:
        if False:
            while True:
                i = 10
        return self.value

    @classmethod
    def from_json(cls, json: str) -> DialogType:
        if False:
            while True:
                i = 10
        return cls(json)

@dataclass
class AppManifestError:
    """
    Error while paring app manifest.
    """
    message: str
    critical: int
    line: int
    column: int

    def to_json(self) -> T_JSON_DICT:
        if False:
            for i in range(10):
                print('nop')
        json: T_JSON_DICT = {}
        json['message'] = self.message
        json['critical'] = self.critical
        json['line'] = self.line
        json['column'] = self.column
        return json

    @classmethod
    def from_json(cls, json: T_JSON_DICT) -> AppManifestError:
        if False:
            return 10
        return cls(message=str(json['message']), critical=int(json['critical']), line=int(json['line']), column=int(json['column']))

@dataclass
class AppManifestParsedProperties:
    """
    Parsed app manifest properties.
    """
    scope: str

    def to_json(self) -> T_JSON_DICT:
        if False:
            for i in range(10):
                print('nop')
        json: T_JSON_DICT = {}
        json['scope'] = self.scope
        return json

    @classmethod
    def from_json(cls, json: T_JSON_DICT) -> AppManifestParsedProperties:
        if False:
            return 10
        return cls(scope=str(json['scope']))

@dataclass
class LayoutViewport:
    """
    Layout viewport position and dimensions.
    """
    page_x: int
    page_y: int
    client_width: int
    client_height: int

    def to_json(self) -> T_JSON_DICT:
        if False:
            i = 10
            return i + 15
        json: T_JSON_DICT = {}
        json['pageX'] = self.page_x
        json['pageY'] = self.page_y
        json['clientWidth'] = self.client_width
        json['clientHeight'] = self.client_height
        return json

    @classmethod
    def from_json(cls, json: T_JSON_DICT) -> LayoutViewport:
        if False:
            while True:
                i = 10
        return cls(page_x=int(json['pageX']), page_y=int(json['pageY']), client_width=int(json['clientWidth']), client_height=int(json['clientHeight']))

@dataclass
class VisualViewport:
    """
    Visual viewport position, dimensions, and scale.
    """
    offset_x: float
    offset_y: float
    page_x: float
    page_y: float
    client_width: float
    client_height: float
    scale: float
    zoom: typing.Optional[float] = None

    def to_json(self) -> T_JSON_DICT:
        if False:
            while True:
                i = 10
        json: T_JSON_DICT = {}
        json['offsetX'] = self.offset_x
        json['offsetY'] = self.offset_y
        json['pageX'] = self.page_x
        json['pageY'] = self.page_y
        json['clientWidth'] = self.client_width
        json['clientHeight'] = self.client_height
        json['scale'] = self.scale
        if self.zoom is not None:
            json['zoom'] = self.zoom
        return json

    @classmethod
    def from_json(cls, json: T_JSON_DICT) -> VisualViewport:
        if False:
            for i in range(10):
                print('nop')
        return cls(offset_x=float(json['offsetX']), offset_y=float(json['offsetY']), page_x=float(json['pageX']), page_y=float(json['pageY']), client_width=float(json['clientWidth']), client_height=float(json['clientHeight']), scale=float(json['scale']), zoom=float(json['zoom']) if 'zoom' in json else None)

@dataclass
class Viewport:
    """
    Viewport for capturing screenshot.
    """
    x: float
    y: float
    width: float
    height: float
    scale: float

    def to_json(self) -> T_JSON_DICT:
        if False:
            print('Hello World!')
        json: T_JSON_DICT = {}
        json['x'] = self.x
        json['y'] = self.y
        json['width'] = self.width
        json['height'] = self.height
        json['scale'] = self.scale
        return json

    @classmethod
    def from_json(cls, json: T_JSON_DICT) -> Viewport:
        if False:
            print('Hello World!')
        return cls(x=float(json['x']), y=float(json['y']), width=float(json['width']), height=float(json['height']), scale=float(json['scale']))

@dataclass
class FontFamilies:
    """
    Generic font families collection.
    """
    standard: typing.Optional[str] = None
    fixed: typing.Optional[str] = None
    serif: typing.Optional[str] = None
    sans_serif: typing.Optional[str] = None
    cursive: typing.Optional[str] = None
    fantasy: typing.Optional[str] = None
    math: typing.Optional[str] = None

    def to_json(self) -> T_JSON_DICT:
        if False:
            i = 10
            return i + 15
        json: T_JSON_DICT = {}
        if self.standard is not None:
            json['standard'] = self.standard
        if self.fixed is not None:
            json['fixed'] = self.fixed
        if self.serif is not None:
            json['serif'] = self.serif
        if self.sans_serif is not None:
            json['sansSerif'] = self.sans_serif
        if self.cursive is not None:
            json['cursive'] = self.cursive
        if self.fantasy is not None:
            json['fantasy'] = self.fantasy
        if self.math is not None:
            json['math'] = self.math
        return json

    @classmethod
    def from_json(cls, json: T_JSON_DICT) -> FontFamilies:
        if False:
            i = 10
            return i + 15
        return cls(standard=str(json['standard']) if 'standard' in json else None, fixed=str(json['fixed']) if 'fixed' in json else None, serif=str(json['serif']) if 'serif' in json else None, sans_serif=str(json['sansSerif']) if 'sansSerif' in json else None, cursive=str(json['cursive']) if 'cursive' in json else None, fantasy=str(json['fantasy']) if 'fantasy' in json else None, math=str(json['math']) if 'math' in json else None)

@dataclass
class ScriptFontFamilies:
    """
    Font families collection for a script.
    """
    script: str
    font_families: FontFamilies

    def to_json(self) -> T_JSON_DICT:
        if False:
            return 10
        json: T_JSON_DICT = {}
        json['script'] = self.script
        json['fontFamilies'] = self.font_families.to_json()
        return json

    @classmethod
    def from_json(cls, json: T_JSON_DICT) -> ScriptFontFamilies:
        if False:
            while True:
                i = 10
        return cls(script=str(json['script']), font_families=FontFamilies.from_json(json['fontFamilies']))

@dataclass
class FontSizes:
    """
    Default font sizes.
    """
    standard: typing.Optional[int] = None
    fixed: typing.Optional[int] = None

    def to_json(self) -> T_JSON_DICT:
        if False:
            for i in range(10):
                print('nop')
        json: T_JSON_DICT = {}
        if self.standard is not None:
            json['standard'] = self.standard
        if self.fixed is not None:
            json['fixed'] = self.fixed
        return json

    @classmethod
    def from_json(cls, json: T_JSON_DICT) -> FontSizes:
        if False:
            print('Hello World!')
        return cls(standard=int(json['standard']) if 'standard' in json else None, fixed=int(json['fixed']) if 'fixed' in json else None)

class ClientNavigationReason(enum.Enum):
    FORM_SUBMISSION_GET = 'formSubmissionGet'
    FORM_SUBMISSION_POST = 'formSubmissionPost'
    HTTP_HEADER_REFRESH = 'httpHeaderRefresh'
    SCRIPT_INITIATED = 'scriptInitiated'
    META_TAG_REFRESH = 'metaTagRefresh'
    PAGE_BLOCK_INTERSTITIAL = 'pageBlockInterstitial'
    RELOAD = 'reload'
    ANCHOR_CLICK = 'anchorClick'

    def to_json(self) -> str:
        if False:
            return 10
        return self.value

    @classmethod
    def from_json(cls, json: str) -> ClientNavigationReason:
        if False:
            for i in range(10):
                print('nop')
        return cls(json)

class ClientNavigationDisposition(enum.Enum):
    CURRENT_TAB = 'currentTab'
    NEW_TAB = 'newTab'
    NEW_WINDOW = 'newWindow'
    DOWNLOAD = 'download'

    def to_json(self) -> str:
        if False:
            return 10
        return self.value

    @classmethod
    def from_json(cls, json: str) -> ClientNavigationDisposition:
        if False:
            i = 10
            return i + 15
        return cls(json)

@dataclass
class InstallabilityErrorArgument:
    name: str
    value: str

    def to_json(self) -> T_JSON_DICT:
        if False:
            while True:
                i = 10
        json: T_JSON_DICT = {}
        json['name'] = self.name
        json['value'] = self.value
        return json

    @classmethod
    def from_json(cls, json: T_JSON_DICT) -> InstallabilityErrorArgument:
        if False:
            return 10
        return cls(name=str(json['name']), value=str(json['value']))

@dataclass
class InstallabilityError:
    """
    The installability error
    """
    error_id: str
    error_arguments: typing.List[InstallabilityErrorArgument]

    def to_json(self) -> T_JSON_DICT:
        if False:
            for i in range(10):
                print('nop')
        json: T_JSON_DICT = {}
        json['errorId'] = self.error_id
        json['errorArguments'] = [i.to_json() for i in self.error_arguments]
        return json

    @classmethod
    def from_json(cls, json: T_JSON_DICT) -> InstallabilityError:
        if False:
            i = 10
            return i + 15
        return cls(error_id=str(json['errorId']), error_arguments=[InstallabilityErrorArgument.from_json(i) for i in json['errorArguments']])

class ReferrerPolicy(enum.Enum):
    """
    The referring-policy used for the navigation.
    """
    NO_REFERRER = 'noReferrer'
    NO_REFERRER_WHEN_DOWNGRADE = 'noReferrerWhenDowngrade'
    ORIGIN = 'origin'
    ORIGIN_WHEN_CROSS_ORIGIN = 'originWhenCrossOrigin'
    SAME_ORIGIN = 'sameOrigin'
    STRICT_ORIGIN = 'strictOrigin'
    STRICT_ORIGIN_WHEN_CROSS_ORIGIN = 'strictOriginWhenCrossOrigin'
    UNSAFE_URL = 'unsafeUrl'

    def to_json(self) -> str:
        if False:
            i = 10
            return i + 15
        return self.value

    @classmethod
    def from_json(cls, json: str) -> ReferrerPolicy:
        if False:
            while True:
                i = 10
        return cls(json)

@dataclass
class CompilationCacheParams:
    """
    Per-script compilation cache parameters for ``Page.produceCompilationCache``
    """
    url: str
    eager: typing.Optional[bool] = None

    def to_json(self) -> T_JSON_DICT:
        if False:
            print('Hello World!')
        json: T_JSON_DICT = {}
        json['url'] = self.url
        if self.eager is not None:
            json['eager'] = self.eager
        return json

    @classmethod
    def from_json(cls, json: T_JSON_DICT) -> CompilationCacheParams:
        if False:
            i = 10
            return i + 15
        return cls(url=str(json['url']), eager=bool(json['eager']) if 'eager' in json else None)

class AutoResponseMode(enum.Enum):
    """
    Enum of possible auto-reponse for permisison / prompt dialogs.
    """
    NONE = 'none'
    AUTO_ACCEPT = 'autoAccept'
    AUTO_REJECT = 'autoReject'
    AUTO_OPT_OUT = 'autoOptOut'

    def to_json(self) -> str:
        if False:
            while True:
                i = 10
        return self.value

    @classmethod
    def from_json(cls, json: str) -> AutoResponseMode:
        if False:
            print('Hello World!')
        return cls(json)

class NavigationType(enum.Enum):
    """
    The type of a frameNavigated event.
    """
    NAVIGATION = 'Navigation'
    BACK_FORWARD_CACHE_RESTORE = 'BackForwardCacheRestore'

    def to_json(self) -> str:
        if False:
            print('Hello World!')
        return self.value

    @classmethod
    def from_json(cls, json: str) -> NavigationType:
        if False:
            i = 10
            return i + 15
        return cls(json)

class BackForwardCacheNotRestoredReason(enum.Enum):
    """
    List of not restored reasons for back-forward cache.
    """
    NOT_PRIMARY_MAIN_FRAME = 'NotPrimaryMainFrame'
    BACK_FORWARD_CACHE_DISABLED = 'BackForwardCacheDisabled'
    RELATED_ACTIVE_CONTENTS_EXIST = 'RelatedActiveContentsExist'
    HTTP_STATUS_NOT_OK = 'HTTPStatusNotOK'
    SCHEME_NOT_HTTP_OR_HTTPS = 'SchemeNotHTTPOrHTTPS'
    LOADING = 'Loading'
    WAS_GRANTED_MEDIA_ACCESS = 'WasGrantedMediaAccess'
    DISABLE_FOR_RENDER_FRAME_HOST_CALLED = 'DisableForRenderFrameHostCalled'
    DOMAIN_NOT_ALLOWED = 'DomainNotAllowed'
    HTTP_METHOD_NOT_GET = 'HTTPMethodNotGET'
    SUBFRAME_IS_NAVIGATING = 'SubframeIsNavigating'
    TIMEOUT = 'Timeout'
    CACHE_LIMIT = 'CacheLimit'
    JAVA_SCRIPT_EXECUTION = 'JavaScriptExecution'
    RENDERER_PROCESS_KILLED = 'RendererProcessKilled'
    RENDERER_PROCESS_CRASHED = 'RendererProcessCrashed'
    SCHEDULER_TRACKED_FEATURE_USED = 'SchedulerTrackedFeatureUsed'
    CONFLICTING_BROWSING_INSTANCE = 'ConflictingBrowsingInstance'
    CACHE_FLUSHED = 'CacheFlushed'
    SERVICE_WORKER_VERSION_ACTIVATION = 'ServiceWorkerVersionActivation'
    SESSION_RESTORED = 'SessionRestored'
    SERVICE_WORKER_POST_MESSAGE = 'ServiceWorkerPostMessage'
    ENTERED_BACK_FORWARD_CACHE_BEFORE_SERVICE_WORKER_HOST_ADDED = 'EnteredBackForwardCacheBeforeServiceWorkerHostAdded'
    RENDER_FRAME_HOST_REUSED_SAME_SITE = 'RenderFrameHostReused_SameSite'
    RENDER_FRAME_HOST_REUSED_CROSS_SITE = 'RenderFrameHostReused_CrossSite'
    SERVICE_WORKER_CLAIM = 'ServiceWorkerClaim'
    IGNORE_EVENT_AND_EVICT = 'IgnoreEventAndEvict'
    HAVE_INNER_CONTENTS = 'HaveInnerContents'
    TIMEOUT_PUTTING_IN_CACHE = 'TimeoutPuttingInCache'
    BACK_FORWARD_CACHE_DISABLED_BY_LOW_MEMORY = 'BackForwardCacheDisabledByLowMemory'
    BACK_FORWARD_CACHE_DISABLED_BY_COMMAND_LINE = 'BackForwardCacheDisabledByCommandLine'
    NETWORK_REQUEST_DATAPIPE_DRAINED_AS_BYTES_CONSUMER = 'NetworkRequestDatapipeDrainedAsBytesConsumer'
    NETWORK_REQUEST_REDIRECTED = 'NetworkRequestRedirected'
    NETWORK_REQUEST_TIMEOUT = 'NetworkRequestTimeout'
    NETWORK_EXCEEDS_BUFFER_LIMIT = 'NetworkExceedsBufferLimit'
    NAVIGATION_CANCELLED_WHILE_RESTORING = 'NavigationCancelledWhileRestoring'
    NOT_MOST_RECENT_NAVIGATION_ENTRY = 'NotMostRecentNavigationEntry'
    BACK_FORWARD_CACHE_DISABLED_FOR_PRERENDER = 'BackForwardCacheDisabledForPrerender'
    USER_AGENT_OVERRIDE_DIFFERS = 'UserAgentOverrideDiffers'
    FOREGROUND_CACHE_LIMIT = 'ForegroundCacheLimit'
    BROWSING_INSTANCE_NOT_SWAPPED = 'BrowsingInstanceNotSwapped'
    BACK_FORWARD_CACHE_DISABLED_FOR_DELEGATE = 'BackForwardCacheDisabledForDelegate'
    UNLOAD_HANDLER_EXISTS_IN_MAIN_FRAME = 'UnloadHandlerExistsInMainFrame'
    UNLOAD_HANDLER_EXISTS_IN_SUB_FRAME = 'UnloadHandlerExistsInSubFrame'
    SERVICE_WORKER_UNREGISTRATION = 'ServiceWorkerUnregistration'
    CACHE_CONTROL_NO_STORE = 'CacheControlNoStore'
    CACHE_CONTROL_NO_STORE_COOKIE_MODIFIED = 'CacheControlNoStoreCookieModified'
    CACHE_CONTROL_NO_STORE_HTTP_ONLY_COOKIE_MODIFIED = 'CacheControlNoStoreHTTPOnlyCookieModified'
    NO_RESPONSE_HEAD = 'NoResponseHead'
    UNKNOWN = 'Unknown'
    ACTIVATION_NAVIGATIONS_DISALLOWED_FOR_BUG1234857 = 'ActivationNavigationsDisallowedForBug1234857'
    ERROR_DOCUMENT = 'ErrorDocument'
    FENCED_FRAMES_EMBEDDER = 'FencedFramesEmbedder'
    COOKIE_DISABLED = 'CookieDisabled'
    WEB_SOCKET = 'WebSocket'
    WEB_TRANSPORT = 'WebTransport'
    WEB_RTC = 'WebRTC'
    MAIN_RESOURCE_HAS_CACHE_CONTROL_NO_STORE = 'MainResourceHasCacheControlNoStore'
    MAIN_RESOURCE_HAS_CACHE_CONTROL_NO_CACHE = 'MainResourceHasCacheControlNoCache'
    SUBRESOURCE_HAS_CACHE_CONTROL_NO_STORE = 'SubresourceHasCacheControlNoStore'
    SUBRESOURCE_HAS_CACHE_CONTROL_NO_CACHE = 'SubresourceHasCacheControlNoCache'
    CONTAINS_PLUGINS = 'ContainsPlugins'
    DOCUMENT_LOADED = 'DocumentLoaded'
    DEDICATED_WORKER_OR_WORKLET = 'DedicatedWorkerOrWorklet'
    OUTSTANDING_NETWORK_REQUEST_OTHERS = 'OutstandingNetworkRequestOthers'
    OUTSTANDING_INDEXED_DB_TRANSACTION = 'OutstandingIndexedDBTransaction'
    REQUESTED_MIDI_PERMISSION = 'RequestedMIDIPermission'
    REQUESTED_AUDIO_CAPTURE_PERMISSION = 'RequestedAudioCapturePermission'
    REQUESTED_VIDEO_CAPTURE_PERMISSION = 'RequestedVideoCapturePermission'
    REQUESTED_BACK_FORWARD_CACHE_BLOCKED_SENSORS = 'RequestedBackForwardCacheBlockedSensors'
    REQUESTED_BACKGROUND_WORK_PERMISSION = 'RequestedBackgroundWorkPermission'
    BROADCAST_CHANNEL = 'BroadcastChannel'
    INDEXED_DB_CONNECTION = 'IndexedDBConnection'
    WEB_XR = 'WebXR'
    SHARED_WORKER = 'SharedWorker'
    WEB_LOCKS = 'WebLocks'
    WEB_HID = 'WebHID'
    WEB_SHARE = 'WebShare'
    REQUESTED_STORAGE_ACCESS_GRANT = 'RequestedStorageAccessGrant'
    WEB_NFC = 'WebNfc'
    OUTSTANDING_NETWORK_REQUEST_FETCH = 'OutstandingNetworkRequestFetch'
    OUTSTANDING_NETWORK_REQUEST_XHR = 'OutstandingNetworkRequestXHR'
    APP_BANNER = 'AppBanner'
    PRINTING = 'Printing'
    WEB_DATABASE = 'WebDatabase'
    PICTURE_IN_PICTURE = 'PictureInPicture'
    PORTAL = 'Portal'
    SPEECH_RECOGNIZER = 'SpeechRecognizer'
    IDLE_MANAGER = 'IdleManager'
    PAYMENT_MANAGER = 'PaymentManager'
    SPEECH_SYNTHESIS = 'SpeechSynthesis'
    KEYBOARD_LOCK = 'KeyboardLock'
    WEB_OTP_SERVICE = 'WebOTPService'
    OUTSTANDING_NETWORK_REQUEST_DIRECT_SOCKET = 'OutstandingNetworkRequestDirectSocket'
    INJECTED_JAVASCRIPT = 'InjectedJavascript'
    INJECTED_STYLE_SHEET = 'InjectedStyleSheet'
    KEEPALIVE_REQUEST = 'KeepaliveRequest'
    INDEXED_DB_EVENT = 'IndexedDBEvent'
    DUMMY = 'Dummy'
    JS_NETWORK_REQUEST_RECEIVED_CACHE_CONTROL_NO_STORE_RESOURCE = 'JsNetworkRequestReceivedCacheControlNoStoreResource'
    WEB_SERIAL = 'WebSerial'
    CONTENT_SECURITY_HANDLER = 'ContentSecurityHandler'
    CONTENT_WEB_AUTHENTICATION_API = 'ContentWebAuthenticationAPI'
    CONTENT_FILE_CHOOSER = 'ContentFileChooser'
    CONTENT_SERIAL = 'ContentSerial'
    CONTENT_FILE_SYSTEM_ACCESS = 'ContentFileSystemAccess'
    CONTENT_MEDIA_DEVICES_DISPATCHER_HOST = 'ContentMediaDevicesDispatcherHost'
    CONTENT_WEB_BLUETOOTH = 'ContentWebBluetooth'
    CONTENT_WEB_USB = 'ContentWebUSB'
    CONTENT_MEDIA_SESSION_SERVICE = 'ContentMediaSessionService'
    CONTENT_SCREEN_READER = 'ContentScreenReader'
    EMBEDDER_POPUP_BLOCKER_TAB_HELPER = 'EmbedderPopupBlockerTabHelper'
    EMBEDDER_SAFE_BROWSING_TRIGGERED_POPUP_BLOCKER = 'EmbedderSafeBrowsingTriggeredPopupBlocker'
    EMBEDDER_SAFE_BROWSING_THREAT_DETAILS = 'EmbedderSafeBrowsingThreatDetails'
    EMBEDDER_APP_BANNER_MANAGER = 'EmbedderAppBannerManager'
    EMBEDDER_DOM_DISTILLER_VIEWER_SOURCE = 'EmbedderDomDistillerViewerSource'
    EMBEDDER_DOM_DISTILLER_SELF_DELETING_REQUEST_DELEGATE = 'EmbedderDomDistillerSelfDeletingRequestDelegate'
    EMBEDDER_OOM_INTERVENTION_TAB_HELPER = 'EmbedderOomInterventionTabHelper'
    EMBEDDER_OFFLINE_PAGE = 'EmbedderOfflinePage'
    EMBEDDER_CHROME_PASSWORD_MANAGER_CLIENT_BIND_CREDENTIAL_MANAGER = 'EmbedderChromePasswordManagerClientBindCredentialManager'
    EMBEDDER_PERMISSION_REQUEST_MANAGER = 'EmbedderPermissionRequestManager'
    EMBEDDER_MODAL_DIALOG = 'EmbedderModalDialog'
    EMBEDDER_EXTENSIONS = 'EmbedderExtensions'
    EMBEDDER_EXTENSION_MESSAGING = 'EmbedderExtensionMessaging'
    EMBEDDER_EXTENSION_MESSAGING_FOR_OPEN_PORT = 'EmbedderExtensionMessagingForOpenPort'
    EMBEDDER_EXTENSION_SENT_MESSAGE_TO_CACHED_FRAME = 'EmbedderExtensionSentMessageToCachedFrame'

    def to_json(self) -> str:
        if False:
            while True:
                i = 10
        return self.value

    @classmethod
    def from_json(cls, json: str) -> BackForwardCacheNotRestoredReason:
        if False:
            return 10
        return cls(json)

class BackForwardCacheNotRestoredReasonType(enum.Enum):
    """
    Types of not restored reasons for back-forward cache.
    """
    SUPPORT_PENDING = 'SupportPending'
    PAGE_SUPPORT_NEEDED = 'PageSupportNeeded'
    CIRCUMSTANTIAL = 'Circumstantial'

    def to_json(self) -> str:
        if False:
            for i in range(10):
                print('nop')
        return self.value

    @classmethod
    def from_json(cls, json: str) -> BackForwardCacheNotRestoredReasonType:
        if False:
            while True:
                i = 10
        return cls(json)

@dataclass
class BackForwardCacheNotRestoredExplanation:
    type_: BackForwardCacheNotRestoredReasonType
    reason: BackForwardCacheNotRestoredReason
    context: typing.Optional[str] = None

    def to_json(self) -> T_JSON_DICT:
        if False:
            print('Hello World!')
        json: T_JSON_DICT = {}
        json['type'] = self.type_.to_json()
        json['reason'] = self.reason.to_json()
        if self.context is not None:
            json['context'] = self.context
        return json

    @classmethod
    def from_json(cls, json: T_JSON_DICT) -> BackForwardCacheNotRestoredExplanation:
        if False:
            return 10
        return cls(type_=BackForwardCacheNotRestoredReasonType.from_json(json['type']), reason=BackForwardCacheNotRestoredReason.from_json(json['reason']), context=str(json['context']) if 'context' in json else None)

@dataclass
class BackForwardCacheNotRestoredExplanationTree:
    url: str
    explanations: typing.List[BackForwardCacheNotRestoredExplanation]
    children: typing.List[BackForwardCacheNotRestoredExplanationTree]

    def to_json(self) -> T_JSON_DICT:
        if False:
            i = 10
            return i + 15
        json: T_JSON_DICT = {}
        json['url'] = self.url
        json['explanations'] = [i.to_json() for i in self.explanations]
        json['children'] = [i.to_json() for i in self.children]
        return json

    @classmethod
    def from_json(cls, json: T_JSON_DICT) -> BackForwardCacheNotRestoredExplanationTree:
        if False:
            return 10
        return cls(url=str(json['url']), explanations=[BackForwardCacheNotRestoredExplanation.from_json(i) for i in json['explanations']], children=[BackForwardCacheNotRestoredExplanationTree.from_json(i) for i in json['children']])

def add_script_to_evaluate_on_load(script_source: str) -> typing.Generator[T_JSON_DICT, T_JSON_DICT, ScriptIdentifier]:
    if False:
        for i in range(10):
            print('nop')
    '\n    Deprecated, please use addScriptToEvaluateOnNewDocument instead.\n\n    **EXPERIMENTAL**\n\n    :param script_source:\n    :returns: Identifier of the added script.\n    '
    params: T_JSON_DICT = {}
    params['scriptSource'] = script_source
    cmd_dict: T_JSON_DICT = {'method': 'Page.addScriptToEvaluateOnLoad', 'params': params}
    json = (yield cmd_dict)
    return ScriptIdentifier.from_json(json['identifier'])

def add_script_to_evaluate_on_new_document(source: str, world_name: typing.Optional[str]=None, include_command_line_api: typing.Optional[bool]=None) -> typing.Generator[T_JSON_DICT, T_JSON_DICT, ScriptIdentifier]:
    if False:
        return 10
    "\n    Evaluates given script in every frame upon creation (before loading frame's scripts).\n\n    :param source:\n    :param world_name: **(EXPERIMENTAL)** *(Optional)* If specified, creates an isolated world with the given name and evaluates given script in it. This world name will be used as the ExecutionContextDescription::name when the corresponding event is emitted.\n    :param include_command_line_api: **(EXPERIMENTAL)** *(Optional)* Specifies whether command line API should be available to the script, defaults to false.\n    :returns: Identifier of the added script.\n    "
    params: T_JSON_DICT = {}
    params['source'] = source
    if world_name is not None:
        params['worldName'] = world_name
    if include_command_line_api is not None:
        params['includeCommandLineAPI'] = include_command_line_api
    cmd_dict: T_JSON_DICT = {'method': 'Page.addScriptToEvaluateOnNewDocument', 'params': params}
    json = (yield cmd_dict)
    return ScriptIdentifier.from_json(json['identifier'])

def bring_to_front() -> typing.Generator[T_JSON_DICT, T_JSON_DICT, None]:
    if False:
        while True:
            i = 10
    '\n    Brings page to front (activates tab).\n    '
    cmd_dict: T_JSON_DICT = {'method': 'Page.bringToFront'}
    yield cmd_dict

def capture_screenshot(format_: typing.Optional[str]=None, quality: typing.Optional[int]=None, clip: typing.Optional[Viewport]=None, from_surface: typing.Optional[bool]=None, capture_beyond_viewport: typing.Optional[bool]=None, optimize_for_speed: typing.Optional[bool]=None) -> typing.Generator[T_JSON_DICT, T_JSON_DICT, str]:
    if False:
        print('Hello World!')
    '\n    Capture page screenshot.\n\n    :param format_: *(Optional)* Image compression format (defaults to png).\n    :param quality: *(Optional)* Compression quality from range [0..100] (jpeg only).\n    :param clip: *(Optional)* Capture the screenshot of a given region only.\n    :param from_surface: **(EXPERIMENTAL)** *(Optional)* Capture the screenshot from the surface, rather than the view. Defaults to true.\n    :param capture_beyond_viewport: **(EXPERIMENTAL)** *(Optional)* Capture the screenshot beyond the viewport. Defaults to false.\n    :param optimize_for_speed: **(EXPERIMENTAL)** *(Optional)* Optimize image encoding for speed, not for resulting size (defaults to false)\n    :returns: Base64-encoded image data. (Encoded as a base64 string when passed over JSON)\n    '
    params: T_JSON_DICT = {}
    if format_ is not None:
        params['format'] = format_
    if quality is not None:
        params['quality'] = quality
    if clip is not None:
        params['clip'] = clip.to_json()
    if from_surface is not None:
        params['fromSurface'] = from_surface
    if capture_beyond_viewport is not None:
        params['captureBeyondViewport'] = capture_beyond_viewport
    if optimize_for_speed is not None:
        params['optimizeForSpeed'] = optimize_for_speed
    cmd_dict: T_JSON_DICT = {'method': 'Page.captureScreenshot', 'params': params}
    json = (yield cmd_dict)
    return str(json['data'])

def capture_snapshot(format_: typing.Optional[str]=None) -> typing.Generator[T_JSON_DICT, T_JSON_DICT, str]:
    if False:
        while True:
            i = 10
    '\n    Returns a snapshot of the page as a string. For MHTML format, the serialization includes\n    iframes, shadow DOM, external resources, and element-inline styles.\n\n    **EXPERIMENTAL**\n\n    :param format_: *(Optional)* Format (defaults to mhtml).\n    :returns: Serialized page data.\n    '
    params: T_JSON_DICT = {}
    if format_ is not None:
        params['format'] = format_
    cmd_dict: T_JSON_DICT = {'method': 'Page.captureSnapshot', 'params': params}
    json = (yield cmd_dict)
    return str(json['data'])

def clear_device_metrics_override() -> typing.Generator[T_JSON_DICT, T_JSON_DICT, None]:
    if False:
        i = 10
        return i + 15
    '\n    Clears the overridden device metrics.\n\n    **EXPERIMENTAL**\n    '
    cmd_dict: T_JSON_DICT = {'method': 'Page.clearDeviceMetricsOverride'}
    yield cmd_dict

def clear_device_orientation_override() -> typing.Generator[T_JSON_DICT, T_JSON_DICT, None]:
    if False:
        print('Hello World!')
    '\n    Clears the overridden Device Orientation.\n\n    **EXPERIMENTAL**\n    '
    cmd_dict: T_JSON_DICT = {'method': 'Page.clearDeviceOrientationOverride'}
    yield cmd_dict

def clear_geolocation_override() -> typing.Generator[T_JSON_DICT, T_JSON_DICT, None]:
    if False:
        print('Hello World!')
    '\n    Clears the overridden Geolocation Position and Error.\n    '
    cmd_dict: T_JSON_DICT = {'method': 'Page.clearGeolocationOverride'}
    yield cmd_dict

def create_isolated_world(frame_id: FrameId, world_name: typing.Optional[str]=None, grant_univeral_access: typing.Optional[bool]=None) -> typing.Generator[T_JSON_DICT, T_JSON_DICT, runtime.ExecutionContextId]:
    if False:
        for i in range(10):
            print('nop')
    '\n    Creates an isolated world for the given frame.\n\n    :param frame_id: Id of the frame in which the isolated world should be created.\n    :param world_name: *(Optional)* An optional name which is reported in the Execution Context.\n    :param grant_univeral_access: *(Optional)* Whether or not universal access should be granted to the isolated world. This is a powerful option, use with caution.\n    :returns: Execution context of the isolated world.\n    '
    params: T_JSON_DICT = {}
    params['frameId'] = frame_id.to_json()
    if world_name is not None:
        params['worldName'] = world_name
    if grant_univeral_access is not None:
        params['grantUniveralAccess'] = grant_univeral_access
    cmd_dict: T_JSON_DICT = {'method': 'Page.createIsolatedWorld', 'params': params}
    json = (yield cmd_dict)
    return runtime.ExecutionContextId.from_json(json['executionContextId'])

def delete_cookie(cookie_name: str, url: str) -> typing.Generator[T_JSON_DICT, T_JSON_DICT, None]:
    if False:
        print('Hello World!')
    '\n    Deletes browser cookie with given name, domain and path.\n\n    **EXPERIMENTAL**\n\n    :param cookie_name: Name of the cookie to remove.\n    :param url: URL to match cooke domain and path.\n    '
    params: T_JSON_DICT = {}
    params['cookieName'] = cookie_name
    params['url'] = url
    cmd_dict: T_JSON_DICT = {'method': 'Page.deleteCookie', 'params': params}
    yield cmd_dict

def disable() -> typing.Generator[T_JSON_DICT, T_JSON_DICT, None]:
    if False:
        for i in range(10):
            print('nop')
    '\n    Disables page domain notifications.\n    '
    cmd_dict: T_JSON_DICT = {'method': 'Page.disable'}
    yield cmd_dict

def enable() -> typing.Generator[T_JSON_DICT, T_JSON_DICT, None]:
    if False:
        return 10
    '\n    Enables page domain notifications.\n    '
    cmd_dict: T_JSON_DICT = {'method': 'Page.enable'}
    yield cmd_dict

def get_app_manifest() -> typing.Generator[T_JSON_DICT, T_JSON_DICT, typing.Tuple[str, typing.List[AppManifestError], typing.Optional[str], typing.Optional[AppManifestParsedProperties]]]:
    if False:
        print('Hello World!')
    '\n\n\n    :returns: A tuple with the following items:\n\n        0. **url** - Manifest location.\n        1. **errors** -\n        2. **data** - *(Optional)* Manifest content.\n        3. **parsed** - *(Optional)* Parsed manifest properties\n    '
    cmd_dict: T_JSON_DICT = {'method': 'Page.getAppManifest'}
    json = (yield cmd_dict)
    return (str(json['url']), [AppManifestError.from_json(i) for i in json['errors']], str(json['data']) if 'data' in json else None, AppManifestParsedProperties.from_json(json['parsed']) if 'parsed' in json else None)

def get_installability_errors() -> typing.Generator[T_JSON_DICT, T_JSON_DICT, typing.List[InstallabilityError]]:
    if False:
        while True:
            i = 10
    '\n\n\n    **EXPERIMENTAL**\n\n    :returns:\n    '
    cmd_dict: T_JSON_DICT = {'method': 'Page.getInstallabilityErrors'}
    json = (yield cmd_dict)
    return [InstallabilityError.from_json(i) for i in json['installabilityErrors']]

def get_manifest_icons() -> typing.Generator[T_JSON_DICT, T_JSON_DICT, typing.Optional[str]]:
    if False:
        print('Hello World!')
    "\n    Deprecated because it's not guaranteed that the returned icon is in fact the one used for PWA installation.\n\n    **EXPERIMENTAL**\n\n    :returns:\n    "
    cmd_dict: T_JSON_DICT = {'method': 'Page.getManifestIcons'}
    json = (yield cmd_dict)
    return str(json['primaryIcon']) if 'primaryIcon' in json else None

def get_app_id() -> typing.Generator[T_JSON_DICT, T_JSON_DICT, typing.Tuple[typing.Optional[str], typing.Optional[str]]]:
    if False:
        for i in range(10):
            print('nop')
    "\n    Returns the unique (PWA) app id.\n    Only returns values if the feature flag 'WebAppEnableManifestId' is enabled\n\n    **EXPERIMENTAL**\n\n    :returns: A tuple with the following items:\n\n        0. **appId** - *(Optional)* App id, either from manifest's id attribute or computed from start_url\n        1. **recommendedId** - *(Optional)* Recommendation for manifest's id attribute to match current id computed from start_url\n    "
    cmd_dict: T_JSON_DICT = {'method': 'Page.getAppId'}
    json = (yield cmd_dict)
    return (str(json['appId']) if 'appId' in json else None, str(json['recommendedId']) if 'recommendedId' in json else None)

def get_ad_script_id(frame_id: FrameId) -> typing.Generator[T_JSON_DICT, T_JSON_DICT, typing.Optional[AdScriptId]]:
    if False:
        print('Hello World!')
    '\n\n\n    **EXPERIMENTAL**\n\n    :param frame_id:\n    :returns: *(Optional)* Identifies the bottom-most script which caused the frame to be labelled as an ad. Only sent if frame is labelled as an ad and id is available.\n    '
    params: T_JSON_DICT = {}
    params['frameId'] = frame_id.to_json()
    cmd_dict: T_JSON_DICT = {'method': 'Page.getAdScriptId', 'params': params}
    json = (yield cmd_dict)
    return AdScriptId.from_json(json['adScriptId']) if 'adScriptId' in json else None

def get_cookies() -> typing.Generator[T_JSON_DICT, T_JSON_DICT, typing.List[network.Cookie]]:
    if False:
        print('Hello World!')
    '\n    Returns all browser cookies for the page and all of its subframes. Depending\n    on the backend support, will return detailed cookie information in the\n    ``cookies`` field.\n\n    **EXPERIMENTAL**\n\n    :returns: Array of cookie objects.\n    '
    cmd_dict: T_JSON_DICT = {'method': 'Page.getCookies'}
    json = (yield cmd_dict)
    return [network.Cookie.from_json(i) for i in json['cookies']]

def get_frame_tree() -> typing.Generator[T_JSON_DICT, T_JSON_DICT, FrameTree]:
    if False:
        while True:
            i = 10
    '\n    Returns present frame tree structure.\n\n    :returns: Present frame tree structure.\n    '
    cmd_dict: T_JSON_DICT = {'method': 'Page.getFrameTree'}
    json = (yield cmd_dict)
    return FrameTree.from_json(json['frameTree'])

def get_layout_metrics() -> typing.Generator[T_JSON_DICT, T_JSON_DICT, typing.Tuple[LayoutViewport, VisualViewport, dom.Rect, LayoutViewport, VisualViewport, dom.Rect]]:
    if False:
        print('Hello World!')
    '\n    Returns metrics relating to the layouting of the page, such as viewport bounds/scale.\n\n    :returns: A tuple with the following items:\n\n        0. **layoutViewport** - Deprecated metrics relating to the layout viewport. Is in device pixels. Use ``cssLayoutViewport`` instead.\n        1. **visualViewport** - Deprecated metrics relating to the visual viewport. Is in device pixels. Use ``cssVisualViewport`` instead.\n        2. **contentSize** - Deprecated size of scrollable area. Is in DP. Use ``cssContentSize`` instead.\n        3. **cssLayoutViewport** - Metrics relating to the layout viewport in CSS pixels.\n        4. **cssVisualViewport** - Metrics relating to the visual viewport in CSS pixels.\n        5. **cssContentSize** - Size of scrollable area in CSS pixels.\n    '
    cmd_dict: T_JSON_DICT = {'method': 'Page.getLayoutMetrics'}
    json = (yield cmd_dict)
    return (LayoutViewport.from_json(json['layoutViewport']), VisualViewport.from_json(json['visualViewport']), dom.Rect.from_json(json['contentSize']), LayoutViewport.from_json(json['cssLayoutViewport']), VisualViewport.from_json(json['cssVisualViewport']), dom.Rect.from_json(json['cssContentSize']))

def get_navigation_history() -> typing.Generator[T_JSON_DICT, T_JSON_DICT, typing.Tuple[int, typing.List[NavigationEntry]]]:
    if False:
        while True:
            i = 10
    '\n    Returns navigation history for the current page.\n\n    :returns: A tuple with the following items:\n\n        0. **currentIndex** - Index of the current navigation history entry.\n        1. **entries** - Array of navigation history entries.\n    '
    cmd_dict: T_JSON_DICT = {'method': 'Page.getNavigationHistory'}
    json = (yield cmd_dict)
    return (int(json['currentIndex']), [NavigationEntry.from_json(i) for i in json['entries']])

def reset_navigation_history() -> typing.Generator[T_JSON_DICT, T_JSON_DICT, None]:
    if False:
        print('Hello World!')
    '\n    Resets navigation history for the current page.\n    '
    cmd_dict: T_JSON_DICT = {'method': 'Page.resetNavigationHistory'}
    yield cmd_dict

def get_resource_content(frame_id: FrameId, url: str) -> typing.Generator[T_JSON_DICT, T_JSON_DICT, typing.Tuple[str, bool]]:
    if False:
        while True:
            i = 10
    '\n    Returns content of the given resource.\n\n    **EXPERIMENTAL**\n\n    :param frame_id: Frame id to get resource for.\n    :param url: URL of the resource to get content for.\n    :returns: A tuple with the following items:\n\n        0. **content** - Resource content.\n        1. **base64Encoded** - True, if content was served as base64.\n    '
    params: T_JSON_DICT = {}
    params['frameId'] = frame_id.to_json()
    params['url'] = url
    cmd_dict: T_JSON_DICT = {'method': 'Page.getResourceContent', 'params': params}
    json = (yield cmd_dict)
    return (str(json['content']), bool(json['base64Encoded']))

def get_resource_tree() -> typing.Generator[T_JSON_DICT, T_JSON_DICT, FrameResourceTree]:
    if False:
        while True:
            i = 10
    '\n    Returns present frame / resource tree structure.\n\n    **EXPERIMENTAL**\n\n    :returns: Present frame / resource tree structure.\n    '
    cmd_dict: T_JSON_DICT = {'method': 'Page.getResourceTree'}
    json = (yield cmd_dict)
    return FrameResourceTree.from_json(json['frameTree'])

def handle_java_script_dialog(accept: bool, prompt_text: typing.Optional[str]=None) -> typing.Generator[T_JSON_DICT, T_JSON_DICT, None]:
    if False:
        return 10
    '\n    Accepts or dismisses a JavaScript initiated dialog (alert, confirm, prompt, or onbeforeunload).\n\n    :param accept: Whether to accept or dismiss the dialog.\n    :param prompt_text: *(Optional)* The text to enter into the dialog prompt before accepting. Used only if this is a prompt dialog.\n    '
    params: T_JSON_DICT = {}
    params['accept'] = accept
    if prompt_text is not None:
        params['promptText'] = prompt_text
    cmd_dict: T_JSON_DICT = {'method': 'Page.handleJavaScriptDialog', 'params': params}
    yield cmd_dict

def navigate(url: str, referrer: typing.Optional[str]=None, transition_type: typing.Optional[TransitionType]=None, frame_id: typing.Optional[FrameId]=None, referrer_policy: typing.Optional[ReferrerPolicy]=None) -> typing.Generator[T_JSON_DICT, T_JSON_DICT, typing.Tuple[FrameId, typing.Optional[network.LoaderId], typing.Optional[str]]]:
    if False:
        i = 10
        return i + 15
    '\n    Navigates current page to the given URL.\n\n    :param url: URL to navigate the page to.\n    :param referrer: *(Optional)* Referrer URL.\n    :param transition_type: *(Optional)* Intended transition type.\n    :param frame_id: *(Optional)* Frame id to navigate, if not specified navigates the top frame.\n    :param referrer_policy: **(EXPERIMENTAL)** *(Optional)* Referrer-policy used for the navigation.\n    :returns: A tuple with the following items:\n\n        0. **frameId** - Frame id that has navigated (or failed to navigate)\n        1. **loaderId** - *(Optional)* Loader identifier. This is omitted in case of same-document navigation, as the previously committed loaderId would not change.\n        2. **errorText** - *(Optional)* User friendly error message, present if and only if navigation has failed.\n    '
    params: T_JSON_DICT = {}
    params['url'] = url
    if referrer is not None:
        params['referrer'] = referrer
    if transition_type is not None:
        params['transitionType'] = transition_type.to_json()
    if frame_id is not None:
        params['frameId'] = frame_id.to_json()
    if referrer_policy is not None:
        params['referrerPolicy'] = referrer_policy.to_json()
    cmd_dict: T_JSON_DICT = {'method': 'Page.navigate', 'params': params}
    json = (yield cmd_dict)
    return (FrameId.from_json(json['frameId']), network.LoaderId.from_json(json['loaderId']) if 'loaderId' in json else None, str(json['errorText']) if 'errorText' in json else None)

def navigate_to_history_entry(entry_id: int) -> typing.Generator[T_JSON_DICT, T_JSON_DICT, None]:
    if False:
        for i in range(10):
            print('nop')
    '\n    Navigates current page to the given history entry.\n\n    :param entry_id: Unique id of the entry to navigate to.\n    '
    params: T_JSON_DICT = {}
    params['entryId'] = entry_id
    cmd_dict: T_JSON_DICT = {'method': 'Page.navigateToHistoryEntry', 'params': params}
    yield cmd_dict

def print_to_pdf(landscape: typing.Optional[bool]=None, display_header_footer: typing.Optional[bool]=None, print_background: typing.Optional[bool]=None, scale: typing.Optional[float]=None, paper_width: typing.Optional[float]=None, paper_height: typing.Optional[float]=None, margin_top: typing.Optional[float]=None, margin_bottom: typing.Optional[float]=None, margin_left: typing.Optional[float]=None, margin_right: typing.Optional[float]=None, page_ranges: typing.Optional[str]=None, header_template: typing.Optional[str]=None, footer_template: typing.Optional[str]=None, prefer_css_page_size: typing.Optional[bool]=None, transfer_mode: typing.Optional[str]=None) -> typing.Generator[T_JSON_DICT, T_JSON_DICT, typing.Tuple[str, typing.Optional[io.StreamHandle]]]:
    if False:
        while True:
            i = 10
    "\n    Print page as PDF.\n\n    :param landscape: *(Optional)* Paper orientation. Defaults to false.\n    :param display_header_footer: *(Optional)* Display header and footer. Defaults to false.\n    :param print_background: *(Optional)* Print background graphics. Defaults to false.\n    :param scale: *(Optional)* Scale of the webpage rendering. Defaults to 1.\n    :param paper_width: *(Optional)* Paper width in inches. Defaults to 8.5 inches.\n    :param paper_height: *(Optional)* Paper height in inches. Defaults to 11 inches.\n    :param margin_top: *(Optional)* Top margin in inches. Defaults to 1cm (~0.4 inches).\n    :param margin_bottom: *(Optional)* Bottom margin in inches. Defaults to 1cm (~0.4 inches).\n    :param margin_left: *(Optional)* Left margin in inches. Defaults to 1cm (~0.4 inches).\n    :param margin_right: *(Optional)* Right margin in inches. Defaults to 1cm (~0.4 inches).\n    :param page_ranges: *(Optional)* Paper ranges to print, one based, e.g., '1-5, 8, 11-13'. Pages are printed in the document order, not in the order specified, and no more than once. Defaults to empty string, which implies the entire document is printed. The page numbers are quietly capped to actual page count of the document, and ranges beyond the end of the document are ignored. If this results in no pages to print, an error is reported. It is an error to specify a range with start greater than end.\n    :param header_template: *(Optional)* HTML template for the print header. Should be valid HTML markup with following classes used to inject printing values into them: - ```date````: formatted print date - ````title````: document title - ````url````: document location - ````pageNumber````: current page number - ````totalPages````: total pages in the document  For example, ````<span class=title></span>```` would generate span containing the title.\n    :param footer_template: *(Optional)* HTML template for the print footer. Should use the same format as the ````headerTemplate````.\n    :param prefer_css_page_size: *(Optional)* Whether or not to prefer page size as defined by css. Defaults to false, in which case the content will be scaled to fit the paper size.\n    :param transfer_mode: **(EXPERIMENTAL)** *(Optional)* return as stream\n    :returns: A tuple with the following items:\n\n        0. **data** - Base64-encoded pdf data. Empty if `` returnAsStream` is specified. (Encoded as a base64 string when passed over JSON)\n        1. **stream** - *(Optional)* A handle of the stream that holds resulting PDF data.\n    "
    params: T_JSON_DICT = {}
    if landscape is not None:
        params['landscape'] = landscape
    if display_header_footer is not None:
        params['displayHeaderFooter'] = display_header_footer
    if print_background is not None:
        params['printBackground'] = print_background
    if scale is not None:
        params['scale'] = scale
    if paper_width is not None:
        params['paperWidth'] = paper_width
    if paper_height is not None:
        params['paperHeight'] = paper_height
    if margin_top is not None:
        params['marginTop'] = margin_top
    if margin_bottom is not None:
        params['marginBottom'] = margin_bottom
    if margin_left is not None:
        params['marginLeft'] = margin_left
    if margin_right is not None:
        params['marginRight'] = margin_right
    if page_ranges is not None:
        params['pageRanges'] = page_ranges
    if header_template is not None:
        params['headerTemplate'] = header_template
    if footer_template is not None:
        params['footerTemplate'] = footer_template
    if prefer_css_page_size is not None:
        params['preferCSSPageSize'] = prefer_css_page_size
    if transfer_mode is not None:
        params['transferMode'] = transfer_mode
    cmd_dict: T_JSON_DICT = {'method': 'Page.printToPDF', 'params': params}
    json = (yield cmd_dict)
    return (str(json['data']), io.StreamHandle.from_json(json['stream']) if 'stream' in json else None)

def reload(ignore_cache: typing.Optional[bool]=None, script_to_evaluate_on_load: typing.Optional[str]=None) -> typing.Generator[T_JSON_DICT, T_JSON_DICT, None]:
    if False:
        for i in range(10):
            print('nop')
    '\n    Reloads given page optionally ignoring the cache.\n\n    :param ignore_cache: *(Optional)* If true, browser cache is ignored (as if the user pressed Shift+refresh).\n    :param script_to_evaluate_on_load: *(Optional)* If set, the script will be injected into all frames of the inspected page after reload. Argument will be ignored if reloading dataURL origin.\n    '
    params: T_JSON_DICT = {}
    if ignore_cache is not None:
        params['ignoreCache'] = ignore_cache
    if script_to_evaluate_on_load is not None:
        params['scriptToEvaluateOnLoad'] = script_to_evaluate_on_load
    cmd_dict: T_JSON_DICT = {'method': 'Page.reload', 'params': params}
    yield cmd_dict

def remove_script_to_evaluate_on_load(identifier: ScriptIdentifier) -> typing.Generator[T_JSON_DICT, T_JSON_DICT, None]:
    if False:
        i = 10
        return i + 15
    '\n    Deprecated, please use removeScriptToEvaluateOnNewDocument instead.\n\n    **EXPERIMENTAL**\n\n    :param identifier:\n    '
    params: T_JSON_DICT = {}
    params['identifier'] = identifier.to_json()
    cmd_dict: T_JSON_DICT = {'method': 'Page.removeScriptToEvaluateOnLoad', 'params': params}
    yield cmd_dict

def remove_script_to_evaluate_on_new_document(identifier: ScriptIdentifier) -> typing.Generator[T_JSON_DICT, T_JSON_DICT, None]:
    if False:
        while True:
            i = 10
    '\n    Removes given script from the list.\n\n    :param identifier:\n    '
    params: T_JSON_DICT = {}
    params['identifier'] = identifier.to_json()
    cmd_dict: T_JSON_DICT = {'method': 'Page.removeScriptToEvaluateOnNewDocument', 'params': params}
    yield cmd_dict

def screencast_frame_ack(session_id: int) -> typing.Generator[T_JSON_DICT, T_JSON_DICT, None]:
    if False:
        print('Hello World!')
    '\n    Acknowledges that a screencast frame has been received by the frontend.\n\n    **EXPERIMENTAL**\n\n    :param session_id: Frame number.\n    '
    params: T_JSON_DICT = {}
    params['sessionId'] = session_id
    cmd_dict: T_JSON_DICT = {'method': 'Page.screencastFrameAck', 'params': params}
    yield cmd_dict

def search_in_resource(frame_id: FrameId, url: str, query: str, case_sensitive: typing.Optional[bool]=None, is_regex: typing.Optional[bool]=None) -> typing.Generator[T_JSON_DICT, T_JSON_DICT, typing.List[debugger.SearchMatch]]:
    if False:
        return 10
    '\n    Searches for given string in resource content.\n\n    **EXPERIMENTAL**\n\n    :param frame_id: Frame id for resource to search in.\n    :param url: URL of the resource to search in.\n    :param query: String to search for.\n    :param case_sensitive: *(Optional)* If true, search is case sensitive.\n    :param is_regex: *(Optional)* If true, treats string parameter as regex.\n    :returns: List of search matches.\n    '
    params: T_JSON_DICT = {}
    params['frameId'] = frame_id.to_json()
    params['url'] = url
    params['query'] = query
    if case_sensitive is not None:
        params['caseSensitive'] = case_sensitive
    if is_regex is not None:
        params['isRegex'] = is_regex
    cmd_dict: T_JSON_DICT = {'method': 'Page.searchInResource', 'params': params}
    json = (yield cmd_dict)
    return [debugger.SearchMatch.from_json(i) for i in json['result']]

def set_ad_blocking_enabled(enabled: bool) -> typing.Generator[T_JSON_DICT, T_JSON_DICT, None]:
    if False:
        return 10
    "\n    Enable Chrome's experimental ad filter on all sites.\n\n    **EXPERIMENTAL**\n\n    :param enabled: Whether to block ads.\n    "
    params: T_JSON_DICT = {}
    params['enabled'] = enabled
    cmd_dict: T_JSON_DICT = {'method': 'Page.setAdBlockingEnabled', 'params': params}
    yield cmd_dict

def set_bypass_csp(enabled: bool) -> typing.Generator[T_JSON_DICT, T_JSON_DICT, None]:
    if False:
        print('Hello World!')
    '\n    Enable page Content Security Policy by-passing.\n\n    **EXPERIMENTAL**\n\n    :param enabled: Whether to bypass page CSP.\n    '
    params: T_JSON_DICT = {}
    params['enabled'] = enabled
    cmd_dict: T_JSON_DICT = {'method': 'Page.setBypassCSP', 'params': params}
    yield cmd_dict

def get_permissions_policy_state(frame_id: FrameId) -> typing.Generator[T_JSON_DICT, T_JSON_DICT, typing.List[PermissionsPolicyFeatureState]]:
    if False:
        for i in range(10):
            print('nop')
    '\n    Get Permissions Policy state on given frame.\n\n    **EXPERIMENTAL**\n\n    :param frame_id:\n    :returns:\n    '
    params: T_JSON_DICT = {}
    params['frameId'] = frame_id.to_json()
    cmd_dict: T_JSON_DICT = {'method': 'Page.getPermissionsPolicyState', 'params': params}
    json = (yield cmd_dict)
    return [PermissionsPolicyFeatureState.from_json(i) for i in json['states']]

def get_origin_trials(frame_id: FrameId) -> typing.Generator[T_JSON_DICT, T_JSON_DICT, typing.List[OriginTrial]]:
    if False:
        while True:
            i = 10
    '\n    Get Origin Trials on given frame.\n\n    **EXPERIMENTAL**\n\n    :param frame_id:\n    :returns:\n    '
    params: T_JSON_DICT = {}
    params['frameId'] = frame_id.to_json()
    cmd_dict: T_JSON_DICT = {'method': 'Page.getOriginTrials', 'params': params}
    json = (yield cmd_dict)
    return [OriginTrial.from_json(i) for i in json['originTrials']]

def set_device_metrics_override(width: int, height: int, device_scale_factor: float, mobile: bool, scale: typing.Optional[float]=None, screen_width: typing.Optional[int]=None, screen_height: typing.Optional[int]=None, position_x: typing.Optional[int]=None, position_y: typing.Optional[int]=None, dont_set_visible_size: typing.Optional[bool]=None, screen_orientation: typing.Optional[emulation.ScreenOrientation]=None, viewport: typing.Optional[Viewport]=None) -> typing.Generator[T_JSON_DICT, T_JSON_DICT, None]:
    if False:
        return 10
    '\n    Overrides the values of device screen dimensions (window.screen.width, window.screen.height,\n    window.innerWidth, window.innerHeight, and "device-width"/"device-height"-related CSS media\n    query results).\n\n    **EXPERIMENTAL**\n\n    :param width: Overriding width value in pixels (minimum 0, maximum 10000000). 0 disables the override.\n    :param height: Overriding height value in pixels (minimum 0, maximum 10000000). 0 disables the override.\n    :param device_scale_factor: Overriding device scale factor value. 0 disables the override.\n    :param mobile: Whether to emulate mobile device. This includes viewport meta tag, overlay scrollbars, text autosizing and more.\n    :param scale: *(Optional)* Scale to apply to resulting view image.\n    :param screen_width: *(Optional)* Overriding screen width value in pixels (minimum 0, maximum 10000000).\n    :param screen_height: *(Optional)* Overriding screen height value in pixels (minimum 0, maximum 10000000).\n    :param position_x: *(Optional)* Overriding view X position on screen in pixels (minimum 0, maximum 10000000).\n    :param position_y: *(Optional)* Overriding view Y position on screen in pixels (minimum 0, maximum 10000000).\n    :param dont_set_visible_size: *(Optional)* Do not set visible view size, rely upon explicit setVisibleSize call.\n    :param screen_orientation: *(Optional)* Screen orientation override.\n    :param viewport: *(Optional)* The viewport dimensions and scale. If not set, the override is cleared.\n    '
    params: T_JSON_DICT = {}
    params['width'] = width
    params['height'] = height
    params['deviceScaleFactor'] = device_scale_factor
    params['mobile'] = mobile
    if scale is not None:
        params['scale'] = scale
    if screen_width is not None:
        params['screenWidth'] = screen_width
    if screen_height is not None:
        params['screenHeight'] = screen_height
    if position_x is not None:
        params['positionX'] = position_x
    if position_y is not None:
        params['positionY'] = position_y
    if dont_set_visible_size is not None:
        params['dontSetVisibleSize'] = dont_set_visible_size
    if screen_orientation is not None:
        params['screenOrientation'] = screen_orientation.to_json()
    if viewport is not None:
        params['viewport'] = viewport.to_json()
    cmd_dict: T_JSON_DICT = {'method': 'Page.setDeviceMetricsOverride', 'params': params}
    yield cmd_dict

def set_device_orientation_override(alpha: float, beta: float, gamma: float) -> typing.Generator[T_JSON_DICT, T_JSON_DICT, None]:
    if False:
        print('Hello World!')
    '\n    Overrides the Device Orientation.\n\n    **EXPERIMENTAL**\n\n    :param alpha: Mock alpha\n    :param beta: Mock beta\n    :param gamma: Mock gamma\n    '
    params: T_JSON_DICT = {}
    params['alpha'] = alpha
    params['beta'] = beta
    params['gamma'] = gamma
    cmd_dict: T_JSON_DICT = {'method': 'Page.setDeviceOrientationOverride', 'params': params}
    yield cmd_dict

def set_font_families(font_families: FontFamilies, for_scripts: typing.Optional[typing.List[ScriptFontFamilies]]=None) -> typing.Generator[T_JSON_DICT, T_JSON_DICT, None]:
    if False:
        while True:
            i = 10
    "\n    Set generic font families.\n\n    **EXPERIMENTAL**\n\n    :param font_families: Specifies font families to set. If a font family is not specified, it won't be changed.\n    :param for_scripts: *(Optional)* Specifies font families to set for individual scripts.\n    "
    params: T_JSON_DICT = {}
    params['fontFamilies'] = font_families.to_json()
    if for_scripts is not None:
        params['forScripts'] = [i.to_json() for i in for_scripts]
    cmd_dict: T_JSON_DICT = {'method': 'Page.setFontFamilies', 'params': params}
    yield cmd_dict

def set_font_sizes(font_sizes: FontSizes) -> typing.Generator[T_JSON_DICT, T_JSON_DICT, None]:
    if False:
        for i in range(10):
            print('nop')
    "\n    Set default font sizes.\n\n    **EXPERIMENTAL**\n\n    :param font_sizes: Specifies font sizes to set. If a font size is not specified, it won't be changed.\n    "
    params: T_JSON_DICT = {}
    params['fontSizes'] = font_sizes.to_json()
    cmd_dict: T_JSON_DICT = {'method': 'Page.setFontSizes', 'params': params}
    yield cmd_dict

def set_document_content(frame_id: FrameId, html: str) -> typing.Generator[T_JSON_DICT, T_JSON_DICT, None]:
    if False:
        i = 10
        return i + 15
    "\n    Sets given markup as the document's HTML.\n\n    :param frame_id: Frame id to set HTML for.\n    :param html: HTML content to set.\n    "
    params: T_JSON_DICT = {}
    params['frameId'] = frame_id.to_json()
    params['html'] = html
    cmd_dict: T_JSON_DICT = {'method': 'Page.setDocumentContent', 'params': params}
    yield cmd_dict

def set_download_behavior(behavior: str, download_path: typing.Optional[str]=None) -> typing.Generator[T_JSON_DICT, T_JSON_DICT, None]:
    if False:
        while True:
            i = 10
    "\n    Set the behavior when downloading a file.\n\n    **EXPERIMENTAL**\n\n    :param behavior: Whether to allow all or deny all download requests, or use default Chrome behavior if available (otherwise deny).\n    :param download_path: *(Optional)* The default path to save downloaded files to. This is required if behavior is set to 'allow'\n    "
    params: T_JSON_DICT = {}
    params['behavior'] = behavior
    if download_path is not None:
        params['downloadPath'] = download_path
    cmd_dict: T_JSON_DICT = {'method': 'Page.setDownloadBehavior', 'params': params}
    yield cmd_dict

def set_geolocation_override(latitude: typing.Optional[float]=None, longitude: typing.Optional[float]=None, accuracy: typing.Optional[float]=None) -> typing.Generator[T_JSON_DICT, T_JSON_DICT, None]:
    if False:
        i = 10
        return i + 15
    '\n    Overrides the Geolocation Position or Error. Omitting any of the parameters emulates position\n    unavailable.\n\n    :param latitude: *(Optional)* Mock latitude\n    :param longitude: *(Optional)* Mock longitude\n    :param accuracy: *(Optional)* Mock accuracy\n    '
    params: T_JSON_DICT = {}
    if latitude is not None:
        params['latitude'] = latitude
    if longitude is not None:
        params['longitude'] = longitude
    if accuracy is not None:
        params['accuracy'] = accuracy
    cmd_dict: T_JSON_DICT = {'method': 'Page.setGeolocationOverride', 'params': params}
    yield cmd_dict

def set_lifecycle_events_enabled(enabled: bool) -> typing.Generator[T_JSON_DICT, T_JSON_DICT, None]:
    if False:
        i = 10
        return i + 15
    '\n    Controls whether page will emit lifecycle events.\n\n    **EXPERIMENTAL**\n\n    :param enabled: If true, starts emitting lifecycle events.\n    '
    params: T_JSON_DICT = {}
    params['enabled'] = enabled
    cmd_dict: T_JSON_DICT = {'method': 'Page.setLifecycleEventsEnabled', 'params': params}
    yield cmd_dict

def set_touch_emulation_enabled(enabled: bool, configuration: typing.Optional[str]=None) -> typing.Generator[T_JSON_DICT, T_JSON_DICT, None]:
    if False:
        i = 10
        return i + 15
    '\n    Toggles mouse event-based touch event emulation.\n\n    **EXPERIMENTAL**\n\n    :param enabled: Whether the touch event emulation should be enabled.\n    :param configuration: *(Optional)* Touch/gesture events configuration. Default: current platform.\n    '
    params: T_JSON_DICT = {}
    params['enabled'] = enabled
    if configuration is not None:
        params['configuration'] = configuration
    cmd_dict: T_JSON_DICT = {'method': 'Page.setTouchEmulationEnabled', 'params': params}
    yield cmd_dict

def start_screencast(format_: typing.Optional[str]=None, quality: typing.Optional[int]=None, max_width: typing.Optional[int]=None, max_height: typing.Optional[int]=None, every_nth_frame: typing.Optional[int]=None) -> typing.Generator[T_JSON_DICT, T_JSON_DICT, None]:
    if False:
        i = 10
        return i + 15
    '\n    Starts sending each frame using the ``screencastFrame`` event.\n\n    **EXPERIMENTAL**\n\n    :param format_: *(Optional)* Image compression format.\n    :param quality: *(Optional)* Compression quality from range [0..100].\n    :param max_width: *(Optional)* Maximum screenshot width.\n    :param max_height: *(Optional)* Maximum screenshot height.\n    :param every_nth_frame: *(Optional)* Send every n-th frame.\n    '
    params: T_JSON_DICT = {}
    if format_ is not None:
        params['format'] = format_
    if quality is not None:
        params['quality'] = quality
    if max_width is not None:
        params['maxWidth'] = max_width
    if max_height is not None:
        params['maxHeight'] = max_height
    if every_nth_frame is not None:
        params['everyNthFrame'] = every_nth_frame
    cmd_dict: T_JSON_DICT = {'method': 'Page.startScreencast', 'params': params}
    yield cmd_dict

def stop_loading() -> typing.Generator[T_JSON_DICT, T_JSON_DICT, None]:
    if False:
        print('Hello World!')
    '\n    Force the page stop all navigations and pending resource fetches.\n    '
    cmd_dict: T_JSON_DICT = {'method': 'Page.stopLoading'}
    yield cmd_dict

def crash() -> typing.Generator[T_JSON_DICT, T_JSON_DICT, None]:
    if False:
        return 10
    '\n    Crashes renderer on the IO thread, generates minidumps.\n\n    **EXPERIMENTAL**\n    '
    cmd_dict: T_JSON_DICT = {'method': 'Page.crash'}
    yield cmd_dict

def close() -> typing.Generator[T_JSON_DICT, T_JSON_DICT, None]:
    if False:
        for i in range(10):
            print('nop')
    '\n    Tries to close page, running its beforeunload hooks, if any.\n\n    **EXPERIMENTAL**\n    '
    cmd_dict: T_JSON_DICT = {'method': 'Page.close'}
    yield cmd_dict

def set_web_lifecycle_state(state: str) -> typing.Generator[T_JSON_DICT, T_JSON_DICT, None]:
    if False:
        print('Hello World!')
    '\n    Tries to update the web lifecycle state of the page.\n    It will transition the page to the given state according to:\n    https://github.com/WICG/web-lifecycle/\n\n    **EXPERIMENTAL**\n\n    :param state: Target lifecycle state\n    '
    params: T_JSON_DICT = {}
    params['state'] = state
    cmd_dict: T_JSON_DICT = {'method': 'Page.setWebLifecycleState', 'params': params}
    yield cmd_dict

def stop_screencast() -> typing.Generator[T_JSON_DICT, T_JSON_DICT, None]:
    if False:
        return 10
    '\n    Stops sending each frame in the ``screencastFrame``.\n\n    **EXPERIMENTAL**\n    '
    cmd_dict: T_JSON_DICT = {'method': 'Page.stopScreencast'}
    yield cmd_dict

def produce_compilation_cache(scripts: typing.List[CompilationCacheParams]) -> typing.Generator[T_JSON_DICT, T_JSON_DICT, None]:
    if False:
        for i in range(10):
            print('nop')
    '\n    Requests backend to produce compilation cache for the specified scripts.\n    ``scripts`` are appeneded to the list of scripts for which the cache\n    would be produced. The list may be reset during page navigation.\n    When script with a matching URL is encountered, the cache is optionally\n    produced upon backend discretion, based on internal heuristics.\n    See also: ``Page.compilationCacheProduced``.\n\n    **EXPERIMENTAL**\n\n    :param scripts:\n    '
    params: T_JSON_DICT = {}
    params['scripts'] = [i.to_json() for i in scripts]
    cmd_dict: T_JSON_DICT = {'method': 'Page.produceCompilationCache', 'params': params}
    yield cmd_dict

def add_compilation_cache(url: str, data: str) -> typing.Generator[T_JSON_DICT, T_JSON_DICT, None]:
    if False:
        print('Hello World!')
    '\n    Seeds compilation cache for given url. Compilation cache does not survive\n    cross-process navigation.\n\n    **EXPERIMENTAL**\n\n    :param url:\n    :param data: Base64-encoded data (Encoded as a base64 string when passed over JSON)\n    '
    params: T_JSON_DICT = {}
    params['url'] = url
    params['data'] = data
    cmd_dict: T_JSON_DICT = {'method': 'Page.addCompilationCache', 'params': params}
    yield cmd_dict

def clear_compilation_cache() -> typing.Generator[T_JSON_DICT, T_JSON_DICT, None]:
    if False:
        while True:
            i = 10
    '\n    Clears seeded compilation cache.\n\n    **EXPERIMENTAL**\n    '
    cmd_dict: T_JSON_DICT = {'method': 'Page.clearCompilationCache'}
    yield cmd_dict

def set_spc_transaction_mode(mode: AutoResponseMode) -> typing.Generator[T_JSON_DICT, T_JSON_DICT, None]:
    if False:
        for i in range(10):
            print('nop')
    '\n    Sets the Secure Payment Confirmation transaction mode.\n    https://w3c.github.io/secure-payment-confirmation/#sctn-automation-set-spc-transaction-mode\n\n    **EXPERIMENTAL**\n\n    :param mode:\n    '
    params: T_JSON_DICT = {}
    params['mode'] = mode.to_json()
    cmd_dict: T_JSON_DICT = {'method': 'Page.setSPCTransactionMode', 'params': params}
    yield cmd_dict

def set_rph_registration_mode(mode: AutoResponseMode) -> typing.Generator[T_JSON_DICT, T_JSON_DICT, None]:
    if False:
        while True:
            i = 10
    '\n    Extensions for Custom Handlers API:\n    https://html.spec.whatwg.org/multipage/system-state.html#rph-automation\n\n    **EXPERIMENTAL**\n\n    :param mode:\n    '
    params: T_JSON_DICT = {}
    params['mode'] = mode.to_json()
    cmd_dict: T_JSON_DICT = {'method': 'Page.setRPHRegistrationMode', 'params': params}
    yield cmd_dict

def generate_test_report(message: str, group: typing.Optional[str]=None) -> typing.Generator[T_JSON_DICT, T_JSON_DICT, None]:
    if False:
        print('Hello World!')
    '\n    Generates a report for testing.\n\n    **EXPERIMENTAL**\n\n    :param message: Message to be displayed in the report.\n    :param group: *(Optional)* Specifies the endpoint group to deliver the report to.\n    '
    params: T_JSON_DICT = {}
    params['message'] = message
    if group is not None:
        params['group'] = group
    cmd_dict: T_JSON_DICT = {'method': 'Page.generateTestReport', 'params': params}
    yield cmd_dict

def wait_for_debugger() -> typing.Generator[T_JSON_DICT, T_JSON_DICT, None]:
    if False:
        print('Hello World!')
    '\n    Pauses page execution. Can be resumed using generic Runtime.runIfWaitingForDebugger.\n\n    **EXPERIMENTAL**\n    '
    cmd_dict: T_JSON_DICT = {'method': 'Page.waitForDebugger'}
    yield cmd_dict

def set_intercept_file_chooser_dialog(enabled: bool) -> typing.Generator[T_JSON_DICT, T_JSON_DICT, None]:
    if False:
        i = 10
        return i + 15
    '\n    Intercept file chooser requests and transfer control to protocol clients.\n    When file chooser interception is enabled, native file chooser dialog is not shown.\n    Instead, a protocol event ``Page.fileChooserOpened`` is emitted.\n\n    **EXPERIMENTAL**\n\n    :param enabled:\n    '
    params: T_JSON_DICT = {}
    params['enabled'] = enabled
    cmd_dict: T_JSON_DICT = {'method': 'Page.setInterceptFileChooserDialog', 'params': params}
    yield cmd_dict

def set_prerendering_allowed(is_allowed: bool) -> typing.Generator[T_JSON_DICT, T_JSON_DICT, None]:
    if False:
        print('Hello World!')
    '\n    Enable/disable prerendering manually.\n\n    This command is a short-term solution for https://crbug.com/1440085.\n    See https://docs.google.com/document/d/12HVmFxYj5Jc-eJr5OmWsa2bqTJsbgGLKI6ZIyx0_wpA\n    for more details.\n\n    TODO(https://crbug.com/1440085): Remove this once Puppeteer supports tab targets.\n\n    **EXPERIMENTAL**\n\n    :param is_allowed:\n    '
    params: T_JSON_DICT = {}
    params['isAllowed'] = is_allowed
    cmd_dict: T_JSON_DICT = {'method': 'Page.setPrerenderingAllowed', 'params': params}
    yield cmd_dict

@event_class('Page.domContentEventFired')
@dataclass
class DomContentEventFired:
    timestamp: network.MonotonicTime

    @classmethod
    def from_json(cls, json: T_JSON_DICT) -> DomContentEventFired:
        if False:
            return 10
        return cls(timestamp=network.MonotonicTime.from_json(json['timestamp']))

@event_class('Page.fileChooserOpened')
@dataclass
class FileChooserOpened:
    """
    Emitted only when ``page.interceptFileChooser`` is enabled.
    """
    frame_id: FrameId
    mode: str
    backend_node_id: typing.Optional[dom.BackendNodeId]

    @classmethod
    def from_json(cls, json: T_JSON_DICT) -> FileChooserOpened:
        if False:
            print('Hello World!')
        return cls(frame_id=FrameId.from_json(json['frameId']), mode=str(json['mode']), backend_node_id=dom.BackendNodeId.from_json(json['backendNodeId']) if 'backendNodeId' in json else None)

@event_class('Page.frameAttached')
@dataclass
class FrameAttached:
    """
    Fired when frame has been attached to its parent.
    """
    frame_id: FrameId
    parent_frame_id: FrameId
    stack: typing.Optional[runtime.StackTrace]

    @classmethod
    def from_json(cls, json: T_JSON_DICT) -> FrameAttached:
        if False:
            return 10
        return cls(frame_id=FrameId.from_json(json['frameId']), parent_frame_id=FrameId.from_json(json['parentFrameId']), stack=runtime.StackTrace.from_json(json['stack']) if 'stack' in json else None)

@event_class('Page.frameClearedScheduledNavigation')
@dataclass
class FrameClearedScheduledNavigation:
    """
    Fired when frame no longer has a scheduled navigation.
    """
    frame_id: FrameId

    @classmethod
    def from_json(cls, json: T_JSON_DICT) -> FrameClearedScheduledNavigation:
        if False:
            print('Hello World!')
        return cls(frame_id=FrameId.from_json(json['frameId']))

@event_class('Page.frameDetached')
@dataclass
class FrameDetached:
    """
    Fired when frame has been detached from its parent.
    """
    frame_id: FrameId
    reason: str

    @classmethod
    def from_json(cls, json: T_JSON_DICT) -> FrameDetached:
        if False:
            return 10
        return cls(frame_id=FrameId.from_json(json['frameId']), reason=str(json['reason']))

@event_class('Page.frameNavigated')
@dataclass
class FrameNavigated:
    """
    Fired once navigation of the frame has completed. Frame is now associated with the new loader.
    """
    frame: Frame
    type_: NavigationType

    @classmethod
    def from_json(cls, json: T_JSON_DICT) -> FrameNavigated:
        if False:
            for i in range(10):
                print('nop')
        return cls(frame=Frame.from_json(json['frame']), type_=NavigationType.from_json(json['type']))

@event_class('Page.documentOpened')
@dataclass
class DocumentOpened:
    """
    **EXPERIMENTAL**

    Fired when opening document to write to.
    """
    frame: Frame

    @classmethod
    def from_json(cls, json: T_JSON_DICT) -> DocumentOpened:
        if False:
            for i in range(10):
                print('nop')
        return cls(frame=Frame.from_json(json['frame']))

@event_class('Page.frameResized')
@dataclass
class FrameResized:
    """
    **EXPERIMENTAL**


    """

    @classmethod
    def from_json(cls, json: T_JSON_DICT) -> FrameResized:
        if False:
            i = 10
            return i + 15
        return cls()

@event_class('Page.frameRequestedNavigation')
@dataclass
class FrameRequestedNavigation:
    """
    **EXPERIMENTAL**

    Fired when a renderer-initiated navigation is requested.
    Navigation may still be cancelled after the event is issued.
    """
    frame_id: FrameId
    reason: ClientNavigationReason
    url: str
    disposition: ClientNavigationDisposition

    @classmethod
    def from_json(cls, json: T_JSON_DICT) -> FrameRequestedNavigation:
        if False:
            while True:
                i = 10
        return cls(frame_id=FrameId.from_json(json['frameId']), reason=ClientNavigationReason.from_json(json['reason']), url=str(json['url']), disposition=ClientNavigationDisposition.from_json(json['disposition']))

@event_class('Page.frameScheduledNavigation')
@dataclass
class FrameScheduledNavigation:
    """
    Fired when frame schedules a potential navigation.
    """
    frame_id: FrameId
    delay: float
    reason: ClientNavigationReason
    url: str

    @classmethod
    def from_json(cls, json: T_JSON_DICT) -> FrameScheduledNavigation:
        if False:
            i = 10
            return i + 15
        return cls(frame_id=FrameId.from_json(json['frameId']), delay=float(json['delay']), reason=ClientNavigationReason.from_json(json['reason']), url=str(json['url']))

@event_class('Page.frameStartedLoading')
@dataclass
class FrameStartedLoading:
    """
    **EXPERIMENTAL**

    Fired when frame has started loading.
    """
    frame_id: FrameId

    @classmethod
    def from_json(cls, json: T_JSON_DICT) -> FrameStartedLoading:
        if False:
            print('Hello World!')
        return cls(frame_id=FrameId.from_json(json['frameId']))

@event_class('Page.frameStoppedLoading')
@dataclass
class FrameStoppedLoading:
    """
    **EXPERIMENTAL**

    Fired when frame has stopped loading.
    """
    frame_id: FrameId

    @classmethod
    def from_json(cls, json: T_JSON_DICT) -> FrameStoppedLoading:
        if False:
            i = 10
            return i + 15
        return cls(frame_id=FrameId.from_json(json['frameId']))

@event_class('Page.downloadWillBegin')
@dataclass
class DownloadWillBegin:
    """
    **EXPERIMENTAL**

    Fired when page is about to start a download.
    Deprecated. Use Browser.downloadWillBegin instead.
    """
    frame_id: FrameId
    guid: str
    url: str
    suggested_filename: str

    @classmethod
    def from_json(cls, json: T_JSON_DICT) -> DownloadWillBegin:
        if False:
            return 10
        return cls(frame_id=FrameId.from_json(json['frameId']), guid=str(json['guid']), url=str(json['url']), suggested_filename=str(json['suggestedFilename']))

@event_class('Page.downloadProgress')
@dataclass
class DownloadProgress:
    """
    **EXPERIMENTAL**

    Fired when download makes progress. Last call has ``done`` == true.
    Deprecated. Use Browser.downloadProgress instead.
    """
    guid: str
    total_bytes: float
    received_bytes: float
    state: str

    @classmethod
    def from_json(cls, json: T_JSON_DICT) -> DownloadProgress:
        if False:
            for i in range(10):
                print('nop')
        return cls(guid=str(json['guid']), total_bytes=float(json['totalBytes']), received_bytes=float(json['receivedBytes']), state=str(json['state']))

@event_class('Page.interstitialHidden')
@dataclass
class InterstitialHidden:
    """
    Fired when interstitial page was hidden
    """

    @classmethod
    def from_json(cls, json: T_JSON_DICT) -> InterstitialHidden:
        if False:
            i = 10
            return i + 15
        return cls()

@event_class('Page.interstitialShown')
@dataclass
class InterstitialShown:
    """
    Fired when interstitial page was shown
    """

    @classmethod
    def from_json(cls, json: T_JSON_DICT) -> InterstitialShown:
        if False:
            i = 10
            return i + 15
        return cls()

@event_class('Page.javascriptDialogClosed')
@dataclass
class JavascriptDialogClosed:
    """
    Fired when a JavaScript initiated dialog (alert, confirm, prompt, or onbeforeunload) has been
    closed.
    """
    result: bool
    user_input: str

    @classmethod
    def from_json(cls, json: T_JSON_DICT) -> JavascriptDialogClosed:
        if False:
            print('Hello World!')
        return cls(result=bool(json['result']), user_input=str(json['userInput']))

@event_class('Page.javascriptDialogOpening')
@dataclass
class JavascriptDialogOpening:
    """
    Fired when a JavaScript initiated dialog (alert, confirm, prompt, or onbeforeunload) is about to
    open.
    """
    url: str
    message: str
    type_: DialogType
    has_browser_handler: bool
    default_prompt: typing.Optional[str]

    @classmethod
    def from_json(cls, json: T_JSON_DICT) -> JavascriptDialogOpening:
        if False:
            for i in range(10):
                print('nop')
        return cls(url=str(json['url']), message=str(json['message']), type_=DialogType.from_json(json['type']), has_browser_handler=bool(json['hasBrowserHandler']), default_prompt=str(json['defaultPrompt']) if 'defaultPrompt' in json else None)

@event_class('Page.lifecycleEvent')
@dataclass
class LifecycleEvent:
    """
    Fired for top level page lifecycle events such as navigation, load, paint, etc.
    """
    frame_id: FrameId
    loader_id: network.LoaderId
    name: str
    timestamp: network.MonotonicTime

    @classmethod
    def from_json(cls, json: T_JSON_DICT) -> LifecycleEvent:
        if False:
            print('Hello World!')
        return cls(frame_id=FrameId.from_json(json['frameId']), loader_id=network.LoaderId.from_json(json['loaderId']), name=str(json['name']), timestamp=network.MonotonicTime.from_json(json['timestamp']))

@event_class('Page.backForwardCacheNotUsed')
@dataclass
class BackForwardCacheNotUsed:
    """
    **EXPERIMENTAL**

    Fired for failed bfcache history navigations if BackForwardCache feature is enabled. Do
    not assume any ordering with the Page.frameNavigated event. This event is fired only for
    main-frame history navigation where the document changes (non-same-document navigations),
    when bfcache navigation fails.
    """
    loader_id: network.LoaderId
    frame_id: FrameId
    not_restored_explanations: typing.List[BackForwardCacheNotRestoredExplanation]
    not_restored_explanations_tree: typing.Optional[BackForwardCacheNotRestoredExplanationTree]

    @classmethod
    def from_json(cls, json: T_JSON_DICT) -> BackForwardCacheNotUsed:
        if False:
            return 10
        return cls(loader_id=network.LoaderId.from_json(json['loaderId']), frame_id=FrameId.from_json(json['frameId']), not_restored_explanations=[BackForwardCacheNotRestoredExplanation.from_json(i) for i in json['notRestoredExplanations']], not_restored_explanations_tree=BackForwardCacheNotRestoredExplanationTree.from_json(json['notRestoredExplanationsTree']) if 'notRestoredExplanationsTree' in json else None)

@event_class('Page.loadEventFired')
@dataclass
class LoadEventFired:
    timestamp: network.MonotonicTime

    @classmethod
    def from_json(cls, json: T_JSON_DICT) -> LoadEventFired:
        if False:
            print('Hello World!')
        return cls(timestamp=network.MonotonicTime.from_json(json['timestamp']))

@event_class('Page.navigatedWithinDocument')
@dataclass
class NavigatedWithinDocument:
    """
    **EXPERIMENTAL**

    Fired when same-document navigation happens, e.g. due to history API usage or anchor navigation.
    """
    frame_id: FrameId
    url: str

    @classmethod
    def from_json(cls, json: T_JSON_DICT) -> NavigatedWithinDocument:
        if False:
            print('Hello World!')
        return cls(frame_id=FrameId.from_json(json['frameId']), url=str(json['url']))

@event_class('Page.screencastFrame')
@dataclass
class ScreencastFrame:
    """
    **EXPERIMENTAL**

    Compressed image data requested by the ``startScreencast``.
    """
    data: str
    metadata: ScreencastFrameMetadata
    session_id: int

    @classmethod
    def from_json(cls, json: T_JSON_DICT) -> ScreencastFrame:
        if False:
            print('Hello World!')
        return cls(data=str(json['data']), metadata=ScreencastFrameMetadata.from_json(json['metadata']), session_id=int(json['sessionId']))

@event_class('Page.screencastVisibilityChanged')
@dataclass
class ScreencastVisibilityChanged:
    """
    **EXPERIMENTAL**

    Fired when the page with currently enabled screencast was shown or hidden .
    """
    visible: bool

    @classmethod
    def from_json(cls, json: T_JSON_DICT) -> ScreencastVisibilityChanged:
        if False:
            i = 10
            return i + 15
        return cls(visible=bool(json['visible']))

@event_class('Page.windowOpen')
@dataclass
class WindowOpen:
    """
    Fired when a new window is going to be opened, via window.open(), link click, form submission,
    etc.
    """
    url: str
    window_name: str
    window_features: typing.List[str]
    user_gesture: bool

    @classmethod
    def from_json(cls, json: T_JSON_DICT) -> WindowOpen:
        if False:
            for i in range(10):
                print('nop')
        return cls(url=str(json['url']), window_name=str(json['windowName']), window_features=[str(i) for i in json['windowFeatures']], user_gesture=bool(json['userGesture']))

@event_class('Page.compilationCacheProduced')
@dataclass
class CompilationCacheProduced:
    """
    **EXPERIMENTAL**

    Issued for every compilation cache generated. Is only available
    if Page.setGenerateCompilationCache is enabled.
    """
    url: str
    data: str

    @classmethod
    def from_json(cls, json: T_JSON_DICT) -> CompilationCacheProduced:
        if False:
            i = 10
            return i + 15
        return cls(url=str(json['url']), data=str(json['data']))