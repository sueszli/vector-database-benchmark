from unittest import TestCase

from bs4 import BeautifulSoup

from tools.lib.capitalization import check_capitalization, get_safe_text, is_capitalized


class GetSafeTextTestCase(TestCase):
    def test_get_safe_text(self) -> None:
        string = "Zulip Zulip. Zulip some text!"
        safe_text = get_safe_text(string)
        self.assertEqual(safe_text, "Zulip zulip. Zulip some text!")

        string = "Zulip Zulip? Zulip some text!"
        safe_text = get_safe_text(string)
        self.assertEqual(safe_text, "Zulip zulip? Zulip some text!")

        string = "Zulip Zulip! Zulip some text!"
        safe_text = get_safe_text(string)
        self.assertEqual(safe_text, "Zulip zulip! Zulip some text!")

        string = "Zulip Zulip, Zulip some text!"
        safe_text = get_safe_text(string)
        self.assertEqual(safe_text, "Zulip zulip, zulip some text!")

        string = "Not Ignored Phrase"
        safe_text = get_safe_text(string)
        self.assertEqual(safe_text, "Not Ignored Phrase")

        string = "Not ignored phrase"
        safe_text = get_safe_text(string)
        self.assertEqual(safe_text, "Not ignored phrase")

        string = ""
        safe_text = get_safe_text(string)
        self.assertEqual(safe_text, "")

        string = """
        <p>Please re-enter your password to confirm your identity.
                (<a href="/accounts/password/reset/" target="_blank">Forgotten it?</a>)</p>
                """
        safe_text = get_safe_text(string)
        soup = BeautifulSoup(safe_text, "lxml")
        rendered_text = " ".join(soup.text.split())
        self.assertEqual(safe_text, rendered_text)

        string = "Edited (__last_edit_timestr__)"
        safe_text = get_safe_text(string)
        self.assertEqual(safe_text, string)


class IsCapitalizedTestCase(TestCase):
    def test_process_text(self) -> None:
        string = "Zulip zulip. Zulip some text!"
        capitalized = is_capitalized(string)
        self.assertTrue(capitalized)

        string = "Zulip zulip? Zulip some text!"
        capitalized = is_capitalized(string)
        self.assertTrue(capitalized)

        string = "Zulip zulip! Zulip some text!"
        capitalized = is_capitalized(string)
        self.assertTrue(capitalized)

        string = "Zulip zulip, Zulip some text!"
        capitalized = is_capitalized(string)
        self.assertTrue(capitalized)

        string = "Some number 25mib"
        capitalized = is_capitalized(string)
        self.assertTrue(capitalized)

        string = "Not Ignored Phrase"
        capitalized = is_capitalized(string)
        self.assertFalse(capitalized)

        string = "Not ignored phrase"
        capitalized = is_capitalized(string)
        self.assertTrue(capitalized)

        string = ""
        capitalized = is_capitalized(string)
        self.assertTrue(capitalized)

        string = "Please re-enter your password to confirm your identity. (Forgotten it?)"
        capitalized = is_capitalized(string)
        self.assertTrue(capitalized)

        string = "Edited (__last_edit_timestr__)"
        capitalized = is_capitalized(string)
        self.assertTrue(capitalized)

        string = "Iphone application"
        capitalized = is_capitalized(string)
        self.assertTrue(capitalized)

        string = "One two etc_ three"
        capitalized = is_capitalized(string)
        self.assertTrue(capitalized)


class CheckCapitalizationTestCase(TestCase):
    def test_check_capitalization(self) -> None:
        strings = [
            "Zulip Zulip. Zulip some text!",
            "Zulip Zulip? Zulip some text!",
            "Zulip Zulip! Zulip some text!",
            "Zulip Zulip, Zulip some text!",
            "Not Ignored Phrase",
            "Not ignored phrase",
            "Some text with realm in it",
            "Realm in capital case",
        ]
        errored, ignored, banned = check_capitalization(strings)
        self.assertEqual(errored, ["Not Ignored Phrase"])
        self.assertEqual(
            ignored,
            sorted(
                [
                    "Zulip Zulip. Zulip some text!",
                    "Zulip Zulip? Zulip some text!",
                    "Zulip Zulip! Zulip some text!",
                    "Zulip Zulip, Zulip some text!",
                ]
            ),
        )

        self.assertEqual(
            banned,
            sorted(
                [
                    "realm found in 'Some text with realm in it'. "
                    "The term realm should not appear in user-facing "
                    "strings. Use organization instead.",
                    "realm found in 'Realm in capital case'. "
                    "The term realm should not appear in user-facing "
                    "strings. Use organization instead.",
                ]
            ),
        )
