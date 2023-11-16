"""
FILE: write_to_ledger.py
DESCRIPTION:
    This sample demonstrates how to write to a Confidential Ledger. In this sample, we write some
    ledger entries and perform common retrieval operations.
USAGE:
    python write_to_ledger.py
    Set the environment variables with your own values before running the sample:
    1) CONFIDENTIALLEDGER_ENDPOINT - the endpoint of the Confidential Ledger.
"""
import logging
import os
import sys
import tempfile
from azure.confidentialledger import ConfidentialLedgerClient
from azure.confidentialledger.certificate import ConfidentialLedgerCertificateClient
from azure.core.exceptions import HttpResponseError
from azure.identity import DefaultAzureCredential
logging.basicConfig(level=logging.ERROR)
LOG = logging.getLogger()

def main():
    if False:
        print('Hello World!')
    try:
        ledger_endpoint = os.environ['CONFIDENTIALLEDGER_ENDPOINT']
    except KeyError:
        LOG.error("Missing environment variable 'CONFIDENTIALLEDGER_ENDPOINT' - please set it before running the example")
        sys.exit(1)
    ledger_id = ledger_endpoint.replace('https://', '').split('.')[0]
    identity_service_client = ConfidentialLedgerCertificateClient()
    ledger_certificate = identity_service_client.get_ledger_identity(ledger_id)
    with tempfile.TemporaryDirectory() as tempdir:
        ledger_cert_file = os.path.join(tempdir, f'{ledger_id}.pem')
        with open(ledger_cert_file, 'w') as outfile:
            outfile.write(ledger_certificate['ledgerTlsCertificate'])
        print(f'Ledger certificate has been written to {ledger_cert_file}. It will be deleted when the script completes.')
        ledger_client = ConfidentialLedgerClient(ledger_endpoint, credential=DefaultAzureCredential(), ledger_certificate_path=ledger_cert_file)
        try:
            post_entry_result = ledger_client.create_ledger_entry({'contents': 'Hello world!'})
            transaction_id = post_entry_result['transactionId']
            print(f'Successfully sent a ledger entry to be written. It will become durable at transaction id {transaction_id}')
        except HttpResponseError as e:
            print('Request failed: {}'.format(e.response.json()))
            raise
        try:
            print(f'Waiting for {transaction_id} to become durable. This may be skipped for when writing less important entries where client throughput is prioritized.')
            wait_poller = ledger_client.begin_wait_for_commit(transaction_id)
            wait_poller.wait()
            print(f'Ledger entry at transaction id {transaction_id} has been committed successfully')
        except HttpResponseError as e:
            print('Request failed: {}'.format(e.response.json()))
            raise
        try:
            current_ledger_entry = ledger_client.get_current_ledger_entry()['contents']
            print(f'The current ledger entry is {current_ledger_entry}')
        except HttpResponseError as e:
            print('Request failed: {}'.format(e.response.json()))
            raise
        try:
            print(f"Writing another entry. This time, we'll have the client method wait for commit.")
            post_poller = ledger_client.begin_create_ledger_entry({'contents': 'Hello world again!'})
            new_post_result = post_poller.result()
            print(f"The new ledger entry has been committed successfully at transaction id {new_post_result['transactionId']}")
        except HttpResponseError as e:
            print('Request failed: {}'.format(e.response.json()))
            raise
        try:
            current_ledger_entry = ledger_client.get_current_ledger_entry()['contents']
            print(f'The current ledger entry is {current_ledger_entry}')
        except HttpResponseError as e:
            print('Request failed: {}'.format(e.response.json()))
            raise
        try:
            get_entry_poller = ledger_client.begin_get_ledger_entry(transaction_id)
            get_entry_result = get_entry_poller.result()
            print(f"At transaction id {get_entry_result['entry']['transactionId']}, the ledger entry contains '{get_entry_result['entry']['contents']}'")
        except HttpResponseError as e:
            print('Request failed: {}'.format(e.response.json()))
            raise
if __name__ == '__main__':
    main()