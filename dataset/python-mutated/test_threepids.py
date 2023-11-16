from synapse.util.threepids import canonicalise_email
from tests.unittest import HomeserverTestCase

class CanonicaliseEmailTests(HomeserverTestCase):

    def test_no_at(self) -> None:
        if False:
            return 10
        with self.assertRaises(ValueError):
            canonicalise_email('address-without-at.bar')

    def test_two_at(self) -> None:
        if False:
            return 10
        with self.assertRaises(ValueError):
            canonicalise_email('foo@foo@test.bar')

    def test_bad_format(self) -> None:
        if False:
            while True:
                i = 10
        with self.assertRaises(ValueError):
            canonicalise_email('user@bad.example.net@good.example.com')

    def test_valid_format(self) -> None:
        if False:
            while True:
                i = 10
        self.assertEqual(canonicalise_email('foo@test.bar'), 'foo@test.bar')

    def test_domain_to_lower(self) -> None:
        if False:
            print('Hello World!')
        self.assertEqual(canonicalise_email('foo@TEST.BAR'), 'foo@test.bar')

    def test_domain_with_umlaut(self) -> None:
        if False:
            return 10
        self.assertEqual(canonicalise_email('foo@Öumlaut.com'), 'foo@öumlaut.com')

    def test_address_casefold(self) -> None:
        if False:
            while True:
                i = 10
        self.assertEqual(canonicalise_email('Strauß@Example.com'), 'strauss@example.com')

    def test_address_trim(self) -> None:
        if False:
            print('Hello World!')
        self.assertEqual(canonicalise_email(' foo@test.bar '), 'foo@test.bar')