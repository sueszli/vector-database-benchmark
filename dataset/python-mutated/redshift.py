from __future__ import annotations
import logging
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from botocore.credentials import ReadOnlyCredentials
log = logging.getLogger(__name__)

def build_credentials_block(credentials: ReadOnlyCredentials) -> str:
    if False:
        for i in range(10):
            print('nop')
    'Generate AWS credentials block for Redshift COPY and UNLOAD commands.\n\n    See AWS docs for details:\n    https://docs.aws.amazon.com/redshift/latest/dg/copy-parameters-authorization.html#copy-credentials\n\n    :param credentials: ReadOnlyCredentials object from `botocore`\n    '
    if credentials.token:
        log.debug('STS token found in credentials, including it in the command')
        credentials_line = f'aws_access_key_id={credentials.access_key};aws_secret_access_key={credentials.secret_key};token={credentials.token}'
    else:
        credentials_line = f'aws_access_key_id={credentials.access_key};aws_secret_access_key={credentials.secret_key}'
    return credentials_line