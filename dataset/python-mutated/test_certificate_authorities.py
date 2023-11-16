import random
import re
import typing
import uuid
import backoff
import google.auth
from conftest import LOCATION
from create_ca_pool import create_ca_pool
from create_certificate_authority import create_certificate_authority
from delete_ca_pool import delete_ca_pool
from delete_certificate_authority import delete_certificate_authority
from disable_certificate_authority import disable_certificate_authority
from enable_certificate_authority import enable_certificate_authority
from monitor_certificate_authority import create_ca_monitor_policy
from undelete_certificate_authority import undelete_certificate_authority
from update_certificate_authority import update_ca_label
PROJECT = google.auth.default()[1]
COMMON_NAME = 'COMMON_NAME'
ORGANIZATION = 'ORGANIZATION'
CA_DURATION = 1000000

def generate_name() -> str:
    if False:
        return 10
    return 'i' + uuid.uuid4().hex[:10]

def backoff_expo_wrapper():
    if False:
        return 10
    for exp in backoff.expo(base=4):
        if exp is None:
            yield None
            continue
        yield (exp * (1 + random.random()))

@backoff.on_exception(backoff_expo_wrapper, Exception, max_tries=3)
def test_create_certificate(capsys: typing.Any, ca_pool_autodelete_name) -> None:
    if False:
        return 10
    CA_POOL_NAME = ca_pool_autodelete_name
    CA_NAME = generate_name()
    create_ca_pool(PROJECT, LOCATION, CA_POOL_NAME)
    create_certificate_authority(PROJECT, LOCATION, CA_POOL_NAME, CA_NAME, COMMON_NAME, ORGANIZATION, CA_DURATION)
    (out, _) = capsys.readouterr()
    assert re.search(f'Operation result: name: "projects/{PROJECT}/locations/{LOCATION}/caPools/{CA_POOL_NAME}/certificateAuthorities/{CA_NAME}"', out)

def test_enable_and_disable_certificate_authority(certificate_authority, capsys: typing.Any) -> None:
    if False:
        while True:
            i = 10
    (CA_POOL_NAME, CA_NAME) = certificate_authority
    enable_certificate_authority(PROJECT, LOCATION, CA_POOL_NAME, CA_NAME)
    disable_certificate_authority(PROJECT, LOCATION, CA_POOL_NAME, CA_NAME)
    (out, _) = capsys.readouterr()
    assert re.search(f'Enabled Certificate Authority: {CA_NAME}', out)
    assert re.search(f'Disabled Certificate Authority: {CA_NAME}', out)

def test_undelete_certificate_authority(deleted_certificate_authority, capsys: typing.Any) -> None:
    if False:
        for i in range(10):
            print('nop')
    (CA_POOL_NAME, CA_NAME) = deleted_certificate_authority
    undelete_certificate_authority(PROJECT, LOCATION, CA_POOL_NAME, CA_NAME)
    delete_certificate_authority(PROJECT, LOCATION, CA_POOL_NAME, CA_NAME)
    delete_ca_pool(PROJECT, LOCATION, CA_POOL_NAME)
    (out, _) = capsys.readouterr()
    assert re.search(f'Successfully undeleted Certificate Authority: {CA_NAME}', out)
    assert re.search(f'Successfully deleted Certificate Authority: {CA_NAME}', out)

def test_update_certificate_authority(certificate_authority, capsys: typing.Any) -> None:
    if False:
        print('Hello World!')
    (CA_POOL_NAME, CA_NAME) = certificate_authority
    update_ca_label(PROJECT, LOCATION, CA_POOL_NAME, CA_NAME)
    (out, _) = capsys.readouterr()
    assert 'Successfully updated the labels !' in out

@backoff.on_exception(backoff_expo_wrapper, Exception, max_tries=3)
def test_create_monitor_ca_policy(capsys: typing.Any) -> None:
    if False:
        print('Hello World!')
    create_ca_monitor_policy(PROJECT)
    (out, _) = capsys.readouterr()
    assert 'Monitoring policy successfully created!' in out