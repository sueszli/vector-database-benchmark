"""ACME AuthHandler."""
import datetime
import logging
import time
from typing import Dict
from typing import Iterable
from typing import List
from typing import Optional
from typing import Sequence
from typing import Tuple
from typing import Type
import josepy
from requests.models import Response
from acme import challenges
from acme import client
from acme import errors as acme_errors
from acme import messages
from certbot import achallenges
from certbot import configuration
from certbot import errors
from certbot import interfaces
from certbot._internal import error_handler
from certbot._internal.account import Account
from certbot.display import util as display_util
from certbot.plugins import common as plugin_common
logger = logging.getLogger(__name__)

class AuthHandler:
    """ACME Authorization Handler for a client.

    :ivar auth: Authenticator capable of solving
        :class:`~acme.challenges.Challenge` types
    :type auth: certbot.interfaces.Authenticator

    :ivar acme.client.ClientV2 acme_client: ACME client API.

    :ivar account: Client's Account
    :type account: :class:`certbot._internal.account.Account`

    :ivar list pref_challs: sorted user specified preferred challenges
        type strings with the most preferred challenge listed first

    """

    def __init__(self, auth: interfaces.Authenticator, acme_client: Optional[client.ClientV2], account: Optional[Account], pref_challs: List[str]) -> None:
        if False:
            i = 10
            return i + 15
        self.auth = auth
        self.acme = acme_client
        self.account = account
        self.pref_challs = pref_challs

    def handle_authorizations(self, orderr: messages.OrderResource, config: configuration.NamespaceConfig, best_effort: bool=False, max_retries: int=30, max_time_mins: float=30) -> List[messages.AuthorizationResource]:
        if False:
            i = 10
            return i + 15
        '\n        Retrieve all authorizations, perform all challenges required to validate\n        these authorizations, then poll and wait for the authorization to be checked.\n        :param acme.messages.OrderResource orderr: must have authorizations filled in\n        :param certbot.configuration.NamespaceConfig config: current Certbot configuration\n        :param bool best_effort: if True, not all authorizations need to be validated (eg. renew)\n        :param int max_retries: maximum number of retries to poll authorizations\n        :param float max_time_mins: maximum time (in minutes) to poll authorizations\n        :returns: list of all validated authorizations\n        :rtype: List\n\n        :raises .AuthorizationError: If unable to retrieve all authorizations\n        '
        authzrs = orderr.authorizations[:]
        if not authzrs:
            raise errors.AuthorizationError('No authorization to handle.')
        if not self.acme:
            raise errors.Error('No ACME client defined, authorizations cannot be handled.')
        achalls = self._choose_challenges(authzrs)
        if not achalls:
            return authzrs
        with error_handler.ExitHandler(self._cleanup_challenges, achalls):
            try:
                resps = self.auth.perform(achalls)
                if config.debug_challenges:
                    display_util.notification('Challenges loaded. Press continue to submit to CA.\n' + self._debug_challenges_msg(achalls, config), pause=True)
            except errors.AuthorizationError as error:
                logger.critical('Failure in setting up challenges.')
                logger.info('Attempting to clean up outstanding challenges...')
                raise error
            assert len(resps) == len(achalls), 'Some challenges have not been performed.'
            for (achall, resp) in zip(achalls, resps):
                self.acme.answer_challenge(achall.challb, resp)
            logger.info('Waiting for verification...')
            self._poll_authorizations(authzrs, max_retries, max_time_mins, best_effort)
            authzrs_validated = [authzr for authzr in authzrs if authzr.body.status == messages.STATUS_VALID]
            if not authzrs_validated:
                raise errors.AuthorizationError('All challenges have failed.')
            return authzrs_validated
        raise errors.Error('An unexpected error occurred while handling the authorizations.')

    def deactivate_valid_authorizations(self, orderr: messages.OrderResource) -> Tuple[List, List]:
        if False:
            return 10
        '\n        Deactivate all `valid` authorizations in the order, so that they cannot be re-used\n        in subsequent orders.\n        :param messages.OrderResource orderr: must have authorizations filled in\n        :returns: tuple of list of successfully deactivated authorizations, and\n                  list of unsuccessfully deactivated authorizations.\n        :rtype: tuple\n        '
        if not self.acme:
            raise errors.Error('No ACME client defined, cannot deactivate valid authorizations.')
        to_deactivate = [authzr for authzr in orderr.authorizations if authzr.body.status == messages.STATUS_VALID]
        deactivated = []
        failed = []
        for authzr in to_deactivate:
            try:
                authzr = self.acme.deactivate_authorization(authzr)
                deactivated.append(authzr)
            except acme_errors.Error as e:
                failed.append(authzr)
                logger.debug('Failed to deactivate authorization %s: %s', authzr.uri, e)
        return (deactivated, failed)

    def _poll_authorizations(self, authzrs: List[messages.AuthorizationResource], max_retries: int, deadline_minutes: float, best_effort: bool) -> None:
        if False:
            i = 10
            return i + 15
        '\n        Poll the ACME CA server, to wait for confirmation that authorizations have their challenges\n        all verified. The poll may occur several times, until all authorizations are checked\n        (valid or invalid), or a maximum of retries, or the polling deadline is reached.\n        '
        if not self.acme:
            raise errors.Error('No ACME client defined, cannot poll authorizations.')
        authzrs_to_check: Dict[int, Tuple[messages.AuthorizationResource, Optional[Response]]] = {index: (authzr, None) for (index, authzr) in enumerate(authzrs)}
        authzrs_failed_to_report = []
        deadline = datetime.datetime.now() + datetime.timedelta(minutes=deadline_minutes)
        sleep_seconds: float = 1
        for _ in range(max_retries):
            if sleep_seconds > 0:
                time.sleep(sleep_seconds)
            authzrs_to_check = {index: self.acme.poll(authzr) for (index, (authzr, _)) in authzrs_to_check.items()}
            for (index, (authzr, _)) in authzrs_to_check.items():
                authzrs[index] = authzr
            authzrs_failed = [authzr for (authzr, _) in authzrs_to_check.values() if authzr.body.status == messages.STATUS_INVALID]
            for authzr_failed in authzrs_failed:
                logger.info('Challenge failed for domain %s', authzr_failed.body.identifier.value)
            authzrs_failed_to_report.extend(authzrs_failed)
            authzrs_to_check = {index: (authzr, resp) for (index, (authzr, resp)) in authzrs_to_check.items() if authzr.body.status == messages.STATUS_PENDING}
            if not authzrs_to_check or datetime.datetime.now() > deadline:
                break
            retry_after = max((self.acme.retry_after(resp, 3) for (_, resp) in authzrs_to_check.values() if resp is not None))
            retry_after = min(retry_after, deadline)
            sleep_seconds = (retry_after - datetime.datetime.now()).total_seconds()
        if authzrs_failed_to_report:
            self._report_failed_authzrs(authzrs_failed_to_report)
            if not best_effort:
                raise errors.AuthorizationError('Some challenges have failed.')
        if authzrs_to_check:
            raise errors.AuthorizationError('All authorizations were not finalized by the CA.')

    def _choose_challenges(self, authzrs: Iterable[messages.AuthorizationResource]) -> List[achallenges.AnnotatedChallenge]:
        if False:
            while True:
                i = 10
        '\n        Retrieve necessary and pending challenges to satisfy server.\n        NB: Necessary and already validated challenges are not retrieved,\n        as they can be reused for a certificate issuance.\n        '
        if not self.acme:
            raise errors.Error('No ACME client defined, cannot choose the challenges.')
        pending_authzrs = [authzr for authzr in authzrs if authzr.body.status != messages.STATUS_VALID]
        achalls: List[achallenges.AnnotatedChallenge] = []
        if pending_authzrs:
            logger.info('Performing the following challenges:')
        for authzr in pending_authzrs:
            authzr_challenges = authzr.body.challenges
            path = gen_challenge_path(authzr_challenges, self._get_chall_pref(authzr.body.identifier.value))
            achalls.extend(self._challenge_factory(authzr, path))
        return achalls

    def _get_chall_pref(self, domain: str) -> List[Type[challenges.Challenge]]:
        if False:
            i = 10
            return i + 15
        'Return list of challenge preferences.\n\n        :param str domain: domain for which you are requesting preferences\n\n        '
        chall_prefs = []
        plugin_pref = self.auth.get_chall_pref(domain)
        if self.pref_challs:
            plugin_pref_types = {chall.typ for chall in plugin_pref}
            for typ in self.pref_challs:
                if typ in plugin_pref_types:
                    chall_prefs.append(challenges.Challenge.TYPES[typ])
            if chall_prefs:
                return chall_prefs
            raise errors.AuthorizationError('None of the preferred challenges are supported by the selected plugin')
        chall_prefs.extend(plugin_pref)
        return chall_prefs

    def _cleanup_challenges(self, achalls: List[achallenges.AnnotatedChallenge]) -> None:
        if False:
            while True:
                i = 10
        'Cleanup challenges.\n\n        :param achalls: annotated challenges to cleanup\n        :type achalls: `list` of :class:`certbot.achallenges.AnnotatedChallenge`\n\n        '
        logger.info('Cleaning up challenges')
        self.auth.cleanup(achalls)

    def _challenge_factory(self, authzr: messages.AuthorizationResource, path: Sequence[int]) -> List[achallenges.AnnotatedChallenge]:
        if False:
            return 10
        'Construct Namedtuple Challenges\n\n        :param messages.AuthorizationResource authzr: authorization\n\n        :param list path: List of indices from `challenges`.\n\n        :returns: achalls, list of challenge type\n            :class:`certbot.achallenges.AnnotatedChallenge`\n        :rtype: list\n\n        :raises .errors.Error: if challenge type is not recognized\n\n        '
        if not self.account:
            raise errors.Error('Account is not set.')
        achalls = []
        for index in path:
            challb = authzr.body.challenges[index]
            achalls.append(challb_to_achall(challb, self.account.key, authzr.body.identifier.value))
        return achalls

    def _report_failed_authzrs(self, failed_authzrs: List[messages.AuthorizationResource]) -> None:
        if False:
            i = 10
            return i + 15
        'Notifies the user about failed authorizations.'
        if not self.account:
            raise errors.Error('Account is not set.')
        problems: Dict[str, List[achallenges.AnnotatedChallenge]] = {}
        failed_achalls = [challb_to_achall(challb, self.account.key, authzr.body.identifier.value) for authzr in failed_authzrs for challb in authzr.body.challenges if challb.error]
        for achall in failed_achalls:
            problems.setdefault(achall.error.typ, []).append(achall)
        msg = [f'\nCertbot failed to authenticate some domains (authenticator: {self.auth.name}). The Certificate Authority reported these problems:']
        for (_, achalls) in sorted(problems.items(), key=lambda item: item[0]):
            msg.append(_generate_failed_chall_msg(achalls))
        if failed_achalls and isinstance(self.auth, plugin_common.Plugin):
            msg.append(f'\nHint: {self.auth.auth_hint(failed_achalls)}\n')
        display_util.notify(''.join(msg))

    def _debug_challenges_msg(self, achalls: List[achallenges.AnnotatedChallenge], config: configuration.NamespaceConfig) -> str:
        if False:
            for i in range(10):
                print('nop')
        'Construct message for debug challenges prompt\n\n        :param list achalls: A list of\n            :class:`certbot.achallenges.AnnotatedChallenge`.\n        :param certbot.configuration.NamespaceConfig config: current Certbot configuration\n        :returns: Message containing challenge debug info\n        :rtype: str\n\n        '
        if config.verbose_count > 0:
            msg = []
            http01_achalls = {}
            dns01_achalls = {}
            for achall in achalls:
                if isinstance(achall.chall, challenges.HTTP01):
                    http01_achalls[achall.chall.uri(achall.domain)] = achall.validation(achall.account_key) + '\n'
                if isinstance(achall.chall, challenges.DNS01):
                    dns01_achalls[achall.validation_domain_name(achall.domain)] = achall.validation(achall.account_key) + '\n'
            if http01_achalls:
                msg.append('The following URLs should be accessible from the internet and return the value mentioned:\n')
                for (uri, key_authz) in http01_achalls.items():
                    msg.append(f'URL: {uri}\nExpected value: {key_authz}')
            if dns01_achalls:
                msg.append('The following FQDNs should return a TXT resource record with the value mentioned:\n')
                for (fqdn, key_authz_hash) in dns01_achalls.items():
                    msg.append(f'FQDN: {fqdn}\nExpected value: {key_authz_hash}')
            return '\n' + '\n'.join(msg)
        else:
            return 'Pass "-v" for more info about challenges.'

def challb_to_achall(challb: messages.ChallengeBody, account_key: josepy.JWK, domain: str) -> achallenges.AnnotatedChallenge:
    if False:
        print('Hello World!')
    'Converts a ChallengeBody object to an AnnotatedChallenge.\n\n    :param .ChallengeBody challb: ChallengeBody\n    :param .JWK account_key: Authorized Account Key\n    :param str domain: Domain of the challb\n\n    :returns: Appropriate AnnotatedChallenge\n    :rtype: :class:`certbot.achallenges.AnnotatedChallenge`\n\n    '
    chall = challb.chall
    logger.info('%s challenge for %s', chall.typ, domain)
    if isinstance(chall, challenges.KeyAuthorizationChallenge):
        return achallenges.KeyAuthorizationAnnotatedChallenge(challb=challb, domain=domain, account_key=account_key)
    elif isinstance(chall, challenges.DNS):
        return achallenges.DNS(challb=challb, domain=domain)
    else:
        return achallenges.Other(challb=challb, domain=domain)

def gen_challenge_path(challbs: List[messages.ChallengeBody], preferences: List[Type[challenges.Challenge]]) -> Tuple[int, ...]:
    if False:
        while True:
            i = 10
    'Generate a plan to get authority over the identity.\n\n    :param tuple challbs: A tuple of challenges\n        (:class:`acme.messages.Challenge`) from\n        :class:`acme.messages.AuthorizationResource` to be\n        fulfilled by the client in order to prove possession of the\n        identifier.\n\n    :param list preferences: List of challenge preferences for domain\n        (:class:`acme.challenges.Challenge` subclasses)\n\n    :returns: list of indices from ``challenges``.\n    :rtype: list\n\n    :raises certbot.errors.AuthorizationError: If a\n        path cannot be created that satisfies the CA given the preferences and\n        combinations.\n\n    '
    chall_cost = {}
    max_cost = 1
    for (i, chall_cls) in enumerate(preferences):
        chall_cost[chall_cls] = i
        max_cost += i
    best_combo: Optional[Tuple[int, ...]] = None
    best_combo_cost = max_cost
    combinations = tuple(((i,) for i in range(len(challbs))))
    combo_total = 0
    for combo in combinations:
        for challenge_index in combo:
            combo_total += chall_cost.get(challbs[challenge_index].chall.__class__, max_cost)
        if combo_total < best_combo_cost:
            best_combo = combo
            best_combo_cost = combo_total
        combo_total = 0
    if not best_combo:
        raise _report_no_chall_path(challbs)
    return best_combo

def _report_no_chall_path(challbs: List[messages.ChallengeBody]) -> errors.AuthorizationError:
    if False:
        return 10
    "Logs and return a raisable error reporting that no satisfiable chall path exists.\n\n    :param challbs: challenges from the authorization that can't be satisfied\n\n    :returns: An authorization error\n    :rtype: certbot.errors.AuthorizationError\n\n    "
    msg = 'Client with the currently selected authenticator does not support any combination of challenges that will satisfy the CA.'
    if len(challbs) == 1 and isinstance(challbs[0].chall, challenges.DNS01):
        msg += ' You may need to use an authenticator plugin that can do challenges over DNS.'
    logger.critical(msg)
    return errors.AuthorizationError(msg)

def _generate_failed_chall_msg(failed_achalls: List[achallenges.AnnotatedChallenge]) -> str:
    if False:
        return 10
    'Creates a user friendly error message about failed challenges.\n\n    :param list failed_achalls: A list of failed\n        :class:`certbot.achallenges.AnnotatedChallenge` with the same error\n        type.\n    :returns: A formatted error message for the client.\n    :rtype: str\n\n    '
    error = failed_achalls[0].error
    typ = error.typ
    if messages.is_acme_error(error):
        typ = error.code
    msg = []
    for achall in failed_achalls:
        msg.append('\n  Domain: %s\n  Type:   %s\n  Detail: %s\n' % (achall.domain, typ, achall.error.detail))
    return ''.join(msg)