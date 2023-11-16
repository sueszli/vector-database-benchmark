import sys
import deeplake
import pytest
from io import StringIO
from contextlib import contextmanager
from click.testing import CliRunner
from deeplake.cli.auth import login, logout
from deeplake.client.client import DeepLakeBackendClient
from deeplake.util.exceptions import AgreementNotAcceptedError, NotLoggedInAgreementError

@contextmanager
def replace_stdin(target):
    if False:
        for i in range(10):
            print('nop')
    orig = sys.stdin
    sys.stdin = target
    yield
    sys.stdin = orig

def dont_agree(path):
    if False:
        print('Hello World!')
    'Load the Deep Lake cloud dataset at path and simulate disagreeing to the terms of access.'
    with pytest.raises(AgreementNotAcceptedError):
        with replace_stdin(StringIO("no, i don't agree!")):
            deeplake.load(path)

def agree(path):
    if False:
        i = 10
        return i + 15
    'Load the Deep Lake cloud dataset at path and simulate agreeing to the terms of access.'
    dataset_name = path.split('/')[-1]
    with replace_stdin(StringIO(dataset_name)):
        ds = deeplake.load(path)
    ds.images[0].numpy()
    ds = deeplake.load(path)
    ds.images[0].numpy()

def reject(path):
    if False:
        return 10
    client = DeepLakeBackendClient()
    (org_id, ds_name) = path.split('/')[-2:]
    client.reject_agreements(org_id, ds_name)

def test_agreement_logged_out(hub_cloud_dev_credentials):
    if False:
        i = 10
        return i + 15
    runner = CliRunner()
    runner.invoke(logout)
    path = 'hub://activeloop/imagenet-test'
    with pytest.raises(NotLoggedInAgreementError):
        agree(path)

@pytest.mark.slow
@pytest.mark.flaky(reruns=3)
def test_agreement_logged_in(hub_cloud_dev_credentials):
    if False:
        while True:
            i = 10
    runner = CliRunner()
    (username, password) = hub_cloud_dev_credentials
    runner.invoke(login, f'-u {username} -p {password}')
    path = 'hub://activeloop/imagenet-test'
    agree(path)
    reject(path)
    runner.invoke(logout)

@pytest.mark.flaky(reruns=3)
@pytest.mark.slow
def test_not_agreement_logged_in(hub_cloud_dev_credentials):
    if False:
        for i in range(10):
            print('nop')
    runner = CliRunner()
    (username, password) = hub_cloud_dev_credentials
    runner.invoke(login, f'-u {username} -p {password}')
    path = 'hub://activeloop/imagenet-test'
    dont_agree(path)
    runner.invoke(logout)