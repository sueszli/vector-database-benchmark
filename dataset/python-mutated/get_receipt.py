"""
FILE: get_receipt.py
DESCRIPTION:
    This sample demonstrates how to retrieve Confidential Ledger receipts. In this sample, we write
    a ledger entry and retrieve a receipt certifying that it was written correctly. 
USAGE:
    python get_receipt.py
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
        while True:
            i = 10
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
            entry_contents = 'Hello world!'
            post_poller = ledger_client.begin_create_ledger_entry({'contents': entry_contents})
            post_entry_result = post_poller.result()
            transaction_id = post_entry_result['transactionId']
            print(f"Wrote '{entry_contents}' to the ledger at transaction {transaction_id}.")
        except HttpResponseError as e:
            print('Request failed: {}'.format(e.response.json()))
            raise
        try:
            print(f'Retrieving a receipt for {transaction_id}. The receipt may be used to cryptographically verify the contents of the transaction.')
            print('For more information about receipts, please see https://microsoft.github.io/CCF/main/audit/receipts.html#receipts')
            get_receipt_poller = ledger_client.begin_get_receipt(transaction_id)
            get_receipt_result = get_receipt_poller.result()
            print(f'Receipt for transaction id {transaction_id}: {get_receipt_result}')
        except HttpResponseError as e:
            print('Request failed: {}'.format(e.response.json()))
            raise
if __name__ == '__main__':
    main()