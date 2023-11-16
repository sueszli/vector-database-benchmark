import unittest
import azure.mgmt.billing
from devtools_testutils import AzureMgmtTestCase

class MgmtBillingTest(AzureMgmtTestCase):

    def _validate_invoice(self, invoice, url_generated=False):
        if False:
            print('Hello World!')
        self.assertIsNotNone(invoice)
        self.assertIsNotNone(invoice.id)
        self.assertIsNotNone(invoice.name)
        self.assertIsNotNone(invoice.type)
        self.assertTrue(len(invoice.billing_period_ids) > 0)
        self.assertTrue(invoice.invoice_period_start_date <= invoice.invoice_period_end_date)
        if url_generated:
            self.assertIsNotNone(invoice.download_url.url)
            self.assertIsNotNone(invoice.download_url.expiry_time)
        else:
            self.assertIsNone(invoice.download_url)

    def _validate_billing_period(self, billing_period):
        if False:
            i = 10
            return i + 15
        self.assertIsNotNone(billing_period)
        self.assertIsNotNone(billing_period.id)
        self.assertIsNotNone(billing_period.name)
        self.assertIsNotNone(billing_period.type)
        self.assertTrue(billing_period.billing_period_start_date <= billing_period.billing_period_end_date)

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        super(MgmtBillingTest, self).setUp()
        self.billing_client = self.create_mgmt_client(azure.mgmt.billing.BillingManagementClient)

    @unittest.skip('skip')
    def test_billing_enrollment_accounts_list(self):
        if False:
            return 10
        output = list(self.billing_client.enrollment_accounts.list())
        self.assertTrue(len(output) > 0)

    @unittest.skip('skip')
    def test_billing_invoice_latest(self):
        if False:
            i = 10
            return i + 15
        output = self.billing_client.invoices.get_latest()
        self._validate_invoice(output, url_generated=True)

    @unittest.skip('skip')
    def test_billing_invoice_list_get(self):
        if False:
            return 10
        output = list(self.billing_client.invoices.list())
        self.assertTrue(len(output) > 0)
        self._validate_invoice(output[0], url_generated=False)
        invoice = self.billing_client.invoices.get(output[0].name)
        self._validate_invoice(invoice, url_generated=True)

    @unittest.skip('skip')
    def test_billing_invoice_list_generate_url(self):
        if False:
            print('Hello World!')
        output = list(self.billing_client.invoices.list(expand='downloadUrl'))
        self.assertTrue(len(output) > 0)
        self._validate_invoice(output[0], url_generated=True)

    @unittest.skip('skip')
    def test_billing_invoice_list_top(self):
        if False:
            return 10
        output = list(self.billing_client.invoices.list(expand='downloadUrl', top=1))
        self.assertEqual(1, len(output))
        self._validate_invoice(output[0], url_generated=True)

    @unittest.skip('skip')
    def test_billing_invoice_list_filter(self):
        if False:
            while True:
                i = 10
        output = list(self.billing_client.invoices.list(filter='invoicePeriodEndDate gt 2017-02-01'))
        self.assertTrue(len(output) > 0)
        self._validate_invoice(output[0], url_generated=False)

    @unittest.skip('skip')
    def test_billing_period_list_get(self):
        if False:
            while True:
                i = 10
        output = list(self.billing_client.billing_periods.list())
        self.assertTrue(len(output) > 0)
        self._validate_billing_period(output[0])
        billing_period = self.billing_client.billing_periods.get(output[0].name)
        self._validate_billing_period(billing_period)

    @unittest.skip('skip')
    def test_billing_period_list_top(self):
        if False:
            print('Hello World!')
        output = list(self.billing_client.billing_periods.list(top=1))
        self.assertEqual(1, len(output))
        self._validate_billing_period(output[0])

    @unittest.skip('skip')
    def test_billing_period_list_filter(self):
        if False:
            return 10
        output = list(self.billing_client.billing_periods.list(filter='billingPeriodEndDate gt 2017-02-01'))
        self.assertTrue(len(output) > 0)
        self._validate_billing_period(output[0])
if __name__ == '__main__':
    unittest.main()