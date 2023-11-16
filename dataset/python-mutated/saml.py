import logging
import re
from typing import TYPE_CHECKING, Callable, Dict, Optional, Set, Tuple
import attr
import saml2
import saml2.response
from saml2.client import Saml2Client
from synapse.api.errors import SynapseError
from synapse.config import ConfigError
from synapse.handlers.sso import MappingException, UserAttributes
from synapse.http.servlet import parse_string
from synapse.http.site import SynapseRequest
from synapse.module_api import ModuleApi
from synapse.types import MXID_LOCALPART_ALLOWED_CHARACTERS, UserID, map_username_to_mxid_localpart
from synapse.util.iterutils import chunk_seq
if TYPE_CHECKING:
    from synapse.server import HomeServer
logger = logging.getLogger(__name__)

@attr.s(slots=True, auto_attribs=True)
class Saml2SessionData:
    """Data we track about SAML2 sessions"""
    creation_time: int
    ui_auth_session_id: Optional[str] = None

class SamlHandler:

    def __init__(self, hs: 'HomeServer'):
        if False:
            print('Hello World!')
        self.store = hs.get_datastores().main
        self.clock = hs.get_clock()
        self.server_name = hs.hostname
        self._saml_client = Saml2Client(hs.config.saml2.saml2_sp_config)
        self._saml_idp_entityid = hs.config.saml2.saml2_idp_entityid
        self._saml2_session_lifetime = hs.config.saml2.saml2_session_lifetime
        self._grandfathered_mxid_source_attribute = hs.config.saml2.saml2_grandfathered_mxid_source_attribute
        self._saml2_attribute_requirements = hs.config.saml2.attribute_requirements
        self._user_mapping_provider = hs.config.saml2.saml2_user_mapping_provider_class(hs.config.saml2.saml2_user_mapping_provider_config, ModuleApi(hs, hs.get_auth_handler()))
        self.idp_id = 'saml'
        self.idp_name = hs.config.saml2.idp_name
        self.idp_icon = hs.config.saml2.idp_icon
        self.idp_brand = hs.config.saml2.idp_brand
        self._outstanding_requests_dict: Dict[str, Saml2SessionData] = {}
        self._sso_handler = hs.get_sso_handler()
        self._sso_handler.register_identity_provider(self)

    async def handle_redirect_request(self, request: SynapseRequest, client_redirect_url: Optional[bytes], ui_auth_session_id: Optional[str]=None) -> str:
        """Handle an incoming request to /login/sso/redirect

        Args:
            request: the incoming HTTP request
            client_redirect_url: the URL that we should redirect the
                client to after login (or None for UI Auth).
            ui_auth_session_id: The session ID of the ongoing UI Auth (or
                None if this is a login).

        Returns:
            URL to redirect to
        """
        if not client_redirect_url:
            client_redirect_url = b'unused'
        (reqid, info) = self._saml_client.prepare_for_authenticate(entityid=self._saml_idp_entityid, relay_state=client_redirect_url)
        logger.info('Initiating a new SAML session: %s' % (reqid,))
        now = self.clock.time_msec()
        self._outstanding_requests_dict[reqid] = Saml2SessionData(creation_time=now, ui_auth_session_id=ui_auth_session_id)
        for (key, value) in info['headers']:
            if key == 'Location':
                return value
        raise Exception("prepare_for_authenticate didn't return a Location header")

    async def handle_saml_response(self, request: SynapseRequest) -> None:
        """Handle an incoming request to /_synapse/client/saml2/authn_response

        Args:
            request: the incoming request from the browser. We'll
                respond to it with a redirect.

        Returns:
            Completes once we have handled the request.
        """
        resp_bytes = parse_string(request, 'SAMLResponse', required=True)
        relay_state = parse_string(request, 'RelayState', required=True)
        self.expire_sessions()
        try:
            saml2_auth = self._saml_client.parse_authn_request_response(resp_bytes, saml2.BINDING_HTTP_POST, outstanding=self._outstanding_requests_dict)
        except saml2.response.UnsolicitedResponse as e:
            logger.warning(str(e))
            self._sso_handler.render_error(request, 'unsolicited_response', 'Unexpected SAML2 login.')
            return
        except Exception as e:
            self._sso_handler.render_error(request, 'invalid_response', 'Unable to parse SAML2 response: %s.' % (e,))
            return
        if saml2_auth.not_signed:
            self._sso_handler.render_error(request, 'unsigned_respond', 'SAML2 response was not signed.')
            return
        logger.debug('SAML2 response: %s', saml2_auth.origxml)
        await self._handle_authn_response(request, saml2_auth, relay_state)

    async def _handle_authn_response(self, request: SynapseRequest, saml2_auth: saml2.response.AuthnResponse, relay_state: str) -> None:
        """Handle an AuthnResponse, having parsed it from the request params

        Assumes that the signature on the response object has been checked. Maps
        the user onto an MXID, registering them if necessary, and returns a response
        to the browser.

        Args:
            request: the incoming request from the browser. We'll respond to it with an
                HTML page or a redirect
            saml2_auth: the parsed AuthnResponse object
            relay_state: the RelayState query param, which encodes the URI to rediret
               back to
        """
        for assertion in saml2_auth.assertions:
            count = 0
            for part in chunk_seq(str(assertion), 10000):
                logger.info('SAML2 assertion: %s%s', '(%i)...' % (count,) if count else '', part)
                count += 1
        logger.info('SAML2 mapped attributes: %s', saml2_auth.ava)
        current_session = self._outstanding_requests_dict.pop(saml2_auth.in_response_to, None)
        if current_session and current_session.ui_auth_session_id:
            try:
                remote_user_id = self._remote_id_from_saml_response(saml2_auth, None)
            except MappingException as e:
                logger.exception('Failed to extract remote user id from SAML response')
                self._sso_handler.render_error(request, 'mapping_error', str(e))
                return
            return await self._sso_handler.complete_sso_ui_auth_request(self.idp_id, remote_user_id, current_session.ui_auth_session_id, request)
        if not self._sso_handler.check_required_attributes(request, saml2_auth.ava, self._saml2_attribute_requirements):
            return
        try:
            await self._complete_saml_login(saml2_auth, request, relay_state)
        except MappingException as e:
            logger.exception('Could not map user')
            self._sso_handler.render_error(request, 'mapping_error', str(e))

    async def _complete_saml_login(self, saml2_auth: saml2.response.AuthnResponse, request: SynapseRequest, client_redirect_url: str) -> None:
        """
        Given a SAML response, complete the login flow

        Retrieves the remote user ID, registers the user if necessary, and serves
        a redirect back to the client with a login-token.

        Args:
            saml2_auth: The parsed SAML2 response.
            request: The request to respond to
            client_redirect_url: The redirect URL passed in by the client.

        Raises:
            MappingException if there was a problem mapping the response to a user.
            RedirectException: some mapping providers may raise this if they need
                to redirect to an interstitial page.
        """
        remote_user_id = self._remote_id_from_saml_response(saml2_auth, client_redirect_url)

        async def saml_response_to_remapped_user_attributes(failures: int) -> UserAttributes:
            """
            Call the mapping provider to map a SAML response to user attributes and coerce the result into the standard form.

            This is backwards compatibility for abstraction for the SSO handler.
            """
            result = self._user_mapping_provider.saml_response_to_user_attributes(saml2_auth, failures, client_redirect_url)
            return UserAttributes(localpart=result.get('mxid_localpart'), display_name=result.get('displayname'), emails=result.get('emails', []))

        async def grandfather_existing_users() -> Optional[str]:
            if self._grandfathered_mxid_source_attribute and self._grandfathered_mxid_source_attribute in saml2_auth.ava:
                attrval = saml2_auth.ava[self._grandfathered_mxid_source_attribute][0]
                user_id = UserID(map_username_to_mxid_localpart(attrval), self.server_name).to_string()
                logger.debug('Looking for existing account based on mapped %s %s', self._grandfathered_mxid_source_attribute, user_id)
                users = await self.store.get_users_by_id_case_insensitive(user_id)
                if users:
                    registered_user_id = list(users.keys())[0]
                    logger.info('Grandfathering mapping to %s', registered_user_id)
                    return registered_user_id
            return None
        await self._sso_handler.complete_sso_login_request(self.idp_id, remote_user_id, request, client_redirect_url, saml_response_to_remapped_user_attributes, grandfather_existing_users)

    def _remote_id_from_saml_response(self, saml2_auth: saml2.response.AuthnResponse, client_redirect_url: Optional[str]) -> str:
        if False:
            return 10
        'Extract the unique remote id from a SAML2 AuthnResponse\n\n        Args:\n            saml2_auth: The parsed SAML2 response.\n            client_redirect_url: The redirect URL passed in by the client.\n        Returns:\n            remote user id\n\n        Raises:\n            MappingException if there was an error extracting the user id\n        '
        remote_user_id = self._user_mapping_provider.get_remote_user_id(saml2_auth, client_redirect_url)
        if not remote_user_id:
            raise MappingException('Failed to extract remote user id from SAML response')
        return remote_user_id

    def expire_sessions(self) -> None:
        if False:
            return 10
        expire_before = self.clock.time_msec() - self._saml2_session_lifetime
        to_expire = set()
        for (reqid, data) in self._outstanding_requests_dict.items():
            if data.creation_time < expire_before:
                to_expire.add(reqid)
        for reqid in to_expire:
            logger.debug('Expiring session id %s', reqid)
            del self._outstanding_requests_dict[reqid]
DOT_REPLACE_PATTERN = re.compile('[^%s]' % (re.escape(''.join(MXID_LOCALPART_ALLOWED_CHARACTERS)),))

def dot_replace_for_mxid(username: str) -> str:
    if False:
        i = 10
        return i + 15
    'Replace any characters which are not allowed in Matrix IDs with a dot.'
    username = username.lower()
    username = DOT_REPLACE_PATTERN.sub('.', username)
    username = re.sub('^_', '', username)
    return username
MXID_MAPPER_MAP: Dict[str, Callable[[str], str]] = {'hexencode': map_username_to_mxid_localpart, 'dotreplace': dot_replace_for_mxid}

@attr.s(auto_attribs=True)
class SamlConfig:
    mxid_source_attribute: str
    mxid_mapper: Callable[[str], str]

class DefaultSamlMappingProvider:
    __version__ = '0.0.1'

    def __init__(self, parsed_config: SamlConfig, module_api: ModuleApi):
        if False:
            while True:
                i = 10
        'The default SAML user mapping provider\n\n        Args:\n            parsed_config: Module configuration\n            module_api: module api proxy\n        '
        self._mxid_source_attribute = parsed_config.mxid_source_attribute
        self._mxid_mapper = parsed_config.mxid_mapper
        self._grandfathered_mxid_source_attribute = module_api._hs.config.saml2.saml2_grandfathered_mxid_source_attribute

    def get_remote_user_id(self, saml_response: saml2.response.AuthnResponse, client_redirect_url: str) -> str:
        if False:
            for i in range(10):
                print('nop')
        'Extracts the remote user id from the SAML response'
        try:
            return saml_response.ava['uid'][0]
        except KeyError:
            logger.warning("SAML2 response lacks a 'uid' attestation")
            raise MappingException("'uid' not in SAML2 response")

    def saml_response_to_user_attributes(self, saml_response: saml2.response.AuthnResponse, failures: int, client_redirect_url: str) -> dict:
        if False:
            print('Hello World!')
        "Maps some text from a SAML response to attributes of a new user\n\n        Args:\n            saml_response: A SAML auth response object\n\n            failures: How many times a call to this function with this\n                saml_response has resulted in a failure\n\n            client_redirect_url: where the client wants to redirect to\n\n        Returns:\n            A dict containing new user attributes. Possible keys:\n                * mxid_localpart (str): Required. The localpart of the user's mxid\n                * displayname (str): The displayname of the user\n                * emails (list[str]): Any emails for the user\n        "
        try:
            mxid_source = saml_response.ava[self._mxid_source_attribute][0]
        except KeyError:
            logger.warning("SAML2 response lacks a '%s' attestation", self._mxid_source_attribute)
            raise SynapseError(400, '%s not in SAML2 response' % (self._mxid_source_attribute,))
        localpart = self._mxid_mapper(mxid_source)
        localpart += str(failures) if failures else ''
        displayname = saml_response.ava.get('displayName', [None])[0]
        emails = saml_response.ava.get('email', [])
        return {'mxid_localpart': localpart, 'displayname': displayname, 'emails': emails}

    @staticmethod
    def parse_config(config: dict) -> SamlConfig:
        if False:
            for i in range(10):
                print('nop')
        "Parse the dict provided by the homeserver's config\n        Args:\n            config: A dictionary containing configuration options for this provider\n        Returns:\n            A custom config object for this module\n        "
        mxid_source_attribute = config.get('mxid_source_attribute', 'uid')
        mapping_type = config.get('mxid_mapping', 'hexencode')
        try:
            mxid_mapper = MXID_MAPPER_MAP[mapping_type]
        except KeyError:
            raise ConfigError("saml2_config.user_mapping_provider.config: '%s' is not a valid mxid_mapping value" % (mapping_type,))
        return SamlConfig(mxid_source_attribute, mxid_mapper)

    @staticmethod
    def get_saml_attributes(config: SamlConfig) -> Tuple[Set[str], Set[str]]:
        if False:
            while True:
                i = 10
        'Returns the required attributes of a SAML\n\n        Args:\n            config: A SamlConfig object containing configuration params for this provider\n\n        Returns:\n            The first set equates to the saml auth response\n                attributes that are required for the module to function, whereas the\n                second set consists of those attributes which can be used if\n                available, but are not necessary\n        '
        return ({'uid', config.mxid_source_attribute}, {'displayName', 'email'})