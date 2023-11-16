"""Certbot Route53 authenticator plugin."""
import collections
import logging
import time
from typing import Any
from typing import Callable
from typing import DefaultDict
from typing import Dict
from typing import List
import boto3
from botocore.exceptions import ClientError
from botocore.exceptions import NoCredentialsError
from acme.challenges import ChallengeResponse
from certbot import achallenges
from certbot import errors
from certbot.achallenges import AnnotatedChallenge
from certbot.plugins import dns_common
from certbot.util import add_deprecated_argument
logger = logging.getLogger(__name__)
INSTRUCTIONS = 'To use certbot-dns-route53, configure credentials as described at https://boto3.readthedocs.io/en/latest/guide/configuration.html#best-practices-for-configuring-credentials and add the necessary permissions for Route53 access.'

class Authenticator(dns_common.DNSAuthenticator):
    """Route53 Authenticator

    This authenticator solves a DNS01 challenge by uploading the answer to AWS
    Route53.
    """
    description = 'Obtain certificates using a DNS TXT record (if you are using AWS Route53 for DNS).'
    ttl = 10

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        if False:
            print('Hello World!')
        super().__init__(*args, **kwargs)
        self.r53 = boto3.client('route53')
        self._resource_records: DefaultDict[str, List[Dict[str, str]]] = collections.defaultdict(list)

    def more_info(self) -> str:
        if False:
            i = 10
            return i + 15
        return 'Solve a DNS01 challenge using AWS Route53'

    @classmethod
    def add_parser_arguments(cls, add: Callable[..., None], default_propagation_seconds: int=10) -> None:
        if False:
            return 10
        add_deprecated_argument(add, 'propagation-seconds', 1)

    def auth_hint(self, failed_achalls: List[achallenges.AnnotatedChallenge]) -> str:
        if False:
            return 10
        return 'The Certificate Authority failed to verify the DNS TXT records created by --dns-route53. Ensure the above domains have their DNS hosted by AWS Route53.'

    def _setup_credentials(self) -> None:
        if False:
            return 10
        pass

    def _perform(self, domain: str, validation_name: str, validation: str) -> None:
        if False:
            return 10
        pass

    def perform(self, achalls: List[AnnotatedChallenge]) -> List[ChallengeResponse]:
        if False:
            return 10
        self._attempt_cleanup = True
        try:
            change_ids = [self._change_txt_record('UPSERT', achall.validation_domain_name(achall.domain), achall.validation(achall.account_key)) for achall in achalls]
            for change_id in change_ids:
                self._wait_for_change(change_id)
        except (NoCredentialsError, ClientError) as e:
            logger.debug('Encountered error during perform: %s', e, exc_info=True)
            raise errors.PluginError('\n'.join([str(e), INSTRUCTIONS]))
        return [achall.response(achall.account_key) for achall in achalls]

    def _cleanup(self, domain: str, validation_name: str, validation: str) -> None:
        if False:
            for i in range(10):
                print('nop')
        try:
            self._change_txt_record('DELETE', validation_name, validation)
        except (NoCredentialsError, ClientError) as e:
            logger.debug('Encountered error during cleanup: %s', e, exc_info=True)

    def _find_zone_id_for_domain(self, domain: str) -> str:
        if False:
            while True:
                i = 10
        'Find the zone id responsible a given FQDN.\n\n           That is, the id for the zone whose name is the longest parent of the\n           domain.\n        '
        paginator = self.r53.get_paginator('list_hosted_zones')
        zones = []
        target_labels = domain.rstrip('.').split('.')
        for page in paginator.paginate():
            for zone in page['HostedZones']:
                if zone['Config']['PrivateZone']:
                    continue
                candidate_labels = zone['Name'].rstrip('.').split('.')
                if candidate_labels == target_labels[-len(candidate_labels):]:
                    zones.append((zone['Name'], zone['Id']))
        if not zones:
            raise errors.PluginError('Unable to find a Route53 hosted zone for {0}'.format(domain))
        zones.sort(key=lambda z: len(z[0]), reverse=True)
        return zones[0][1]

    def _change_txt_record(self, action: str, validation_domain_name: str, validation: str) -> str:
        if False:
            print('Hello World!')
        zone_id = self._find_zone_id_for_domain(validation_domain_name)
        rrecords = self._resource_records[validation_domain_name]
        challenge = {'Value': '"{0}"'.format(validation)}
        if action == 'DELETE':
            rrecords.remove(challenge)
            if rrecords:
                action = 'UPSERT'
            else:
                rrecords = [challenge]
        else:
            rrecords.append(challenge)
        response = self.r53.change_resource_record_sets(HostedZoneId=zone_id, ChangeBatch={'Comment': 'certbot-dns-route53 certificate validation ' + action, 'Changes': [{'Action': action, 'ResourceRecordSet': {'Name': validation_domain_name, 'Type': 'TXT', 'TTL': self.ttl, 'ResourceRecords': rrecords}}]})
        return response['ChangeInfo']['Id']

    def _wait_for_change(self, change_id: str) -> None:
        if False:
            for i in range(10):
                print('nop')
        'Wait for a change to be propagated to all Route53 DNS servers.\n           https://docs.aws.amazon.com/Route53/latest/APIReference/API_GetChange.html\n        '
        for unused_n in range(0, 120):
            response = self.r53.get_change(Id=change_id)
            if response['ChangeInfo']['Status'] == 'INSYNC':
                return
            time.sleep(5)
        raise errors.PluginError('Timed out waiting for Route53 change. Current status: %s' % response['ChangeInfo']['Status'])