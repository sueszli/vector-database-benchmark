import logging
from typing import TYPE_CHECKING, Any, Set
from synapse.api.errors import SynapseError
from synapse.api.urls import ConsentURIBuilder
from synapse.config import ConfigError
from synapse.types import get_localpart_from_id
if TYPE_CHECKING:
    from synapse.server import HomeServer
logger = logging.getLogger(__name__)

class ConsentServerNotices:
    """Keeps track of whether we need to send users server_notices about
    privacy policy consent, and sends one if we do.
    """

    def __init__(self, hs: 'HomeServer'):
        if False:
            i = 10
            return i + 15
        self._server_notices_manager = hs.get_server_notices_manager()
        self._store = hs.get_datastores().main
        self._users_in_progress: Set[str] = set()
        self._current_consent_version = hs.config.consent.user_consent_version
        self._server_notice_content = hs.config.consent.user_consent_server_notice_content
        self._send_to_guests = hs.config.consent.user_consent_server_notice_to_guests
        if self._server_notice_content is not None:
            if not self._server_notices_manager.is_enabled():
                raise ConfigError('user_consent configuration requires server notices, but server notices are not enabled.')
            if 'body' not in self._server_notice_content:
                raise ConfigError("user_consent server_notice_consent must contain a 'body' key.")
            self._consent_uri_builder = ConsentURIBuilder(hs.config)

    async def maybe_send_server_notice_to_user(self, user_id: str) -> None:
        """Check if we need to send a notice to this user, and does so if so

        Args:
            user_id: user to check
        """
        if self._server_notice_content is None:
            return
        assert self._current_consent_version is not None
        if user_id in self._users_in_progress:
            return
        self._users_in_progress.add(user_id)
        try:
            u = await self._store.get_user_by_id(user_id)
            if u is None:
                return
            if u.is_guest and (not self._send_to_guests):
                return
            if u.consent_version == self._current_consent_version:
                return
            if u.consent_server_notice_sent == self._current_consent_version:
                return
            try:
                consent_uri = self._consent_uri_builder.build_user_consent_uri(get_localpart_from_id(user_id))
                content = copy_with_str_subst(self._server_notice_content, {'consent_uri': consent_uri})
                await self._server_notices_manager.send_notice(user_id, content)
                await self._store.user_set_consent_server_notice_sent(user_id, self._current_consent_version)
            except SynapseError as e:
                logger.error('Error sending server notice about user consent: %s', e)
        finally:
            self._users_in_progress.remove(user_id)

def copy_with_str_subst(x: Any, substitutions: Any) -> Any:
    if False:
        print('Hello World!')
    "Deep-copy a structure, carrying out string substitutions on any strings\n\n    Args:\n        x: structure to be copied\n        substitutions: substitutions to be made - passed into the string '%' operator\n\n    Returns:\n        copy of x\n    "
    if isinstance(x, str):
        return x % substitutions
    if isinstance(x, dict):
        return {k: copy_with_str_subst(v, substitutions) for (k, v) in x.items()}
    if isinstance(x, (list, tuple)):
        return [copy_with_str_subst(y, substitutions) for y in x]
    return x