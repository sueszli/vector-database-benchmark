"""
FILE: list_ledger_entries.py
DESCRIPTION:
    This sample demonstrates how to iteratively retrieve a batch of ledger entries. In this sample,
    we write many ledger entries before retrieving them at once.
USAGE:
    python list_ledger_entries.py
    Set the environment variables with your own values before running the sample:
    1) CONFIDENTIALLEDGER_ENDPOINT - the endpoint of the Confidential Ledger.
"""
import logging
import os
import sys
import tempfile
from azure.confidentialledger import ConfidentialLedgerClient
from azure.confidentialledger.certificate import ConfidentialLedgerCertificateClient
from azure.identity import DefaultAzureCredential
logging.basicConfig(level=logging.ERROR)
LOG = logging.getLogger()

def main():
    if False:
        for i in range(10):
            print('nop')
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
        post_poller = ledger_client.begin_create_ledger_entry({'contents': 'First message'})
        first_transaction_id = post_poller.result()['transactionId']
        print(f"Wrote 'First message' to the ledger. It is recorded at transaction id {first_transaction_id}.")
        for i in range(10):
            entry_contents = f'Message {i}'
            print(f"Writing '{entry_contents}' to the ledger.")
            ledger_client.create_ledger_entry({'contents': entry_contents})
        post_poller = ledger_client.begin_create_ledger_entry({'contents': 'Last message'})
        last_transaction_id = post_poller.result()['transactionId']
        print(f"Wrote 'Last message' to the ledger. It is recorded at transaction id {last_transaction_id}.")
        ranged_result = ledger_client.list_ledger_entries(from_transaction_id=first_transaction_id, to_transaction_id=last_transaction_id)
        for entry in ranged_result:
            print(f"Contents at {entry['transactionId']}: {entry['contents']}")
if __name__ == '__main__':
    main()