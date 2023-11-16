from operator import attrgetter
from django.test.testcases import TestCase
from .models import Address, Contact, Customer

class TestLookupQuery(TestCase):

    @classmethod
    def setUpTestData(cls):
        if False:
            print('Hello World!')
        cls.address = Address.objects.create(company=1, customer_id=20)
        cls.customer1 = Customer.objects.create(company=1, customer_id=20)
        cls.contact1 = Contact.objects.create(company_code=1, customer_code=20)

    def test_deep_mixed_forward(self):
        if False:
            print('Hello World!')
        self.assertQuerySetEqual(Address.objects.filter(customer__contacts=self.contact1), [self.address.id], attrgetter('id'))

    def test_deep_mixed_backward(self):
        if False:
            while True:
                i = 10
        self.assertQuerySetEqual(Contact.objects.filter(customer__address=self.address), [self.contact1.id], attrgetter('id'))