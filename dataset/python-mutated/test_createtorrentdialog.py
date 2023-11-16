from PyQt5.QtGui import QValidator
from tribler.gui.dialogs.createtorrentdialog import sanitize_filename, TorrentNameValidator

def test_torrent_name_validator():
    if False:
        return 10
    '\n    Tests if the torrent name validator marks the input as valid if there are no multiline characters.\n    Upon fixup, the invalid characters are accepted correctly.\n    '

    def assert_text_is_valid(validator: QValidator, original_text: str, expected_to_be_valid: bool):
        if False:
            while True:
                i = 10
        (state, text, pos) = validator.validate(original_text, len(original_text))
        assert state == QValidator.Acceptable if expected_to_be_valid else QValidator.Intermediate
        assert text == original_text
        assert pos == len(original_text)
    validator = TorrentNameValidator(None)
    invalid_name = 'line 1\n    line2.torrent\n    '
    assert_text_is_valid(validator, invalid_name, expected_to_be_valid=False)
    fixed_name = validator.fixup(invalid_name)
    assert_text_is_valid(validator, fixed_name, expected_to_be_valid=True)

def test_sanitize_filename():
    if False:
        return 10
    original_filename = 'This \nIs \r\nA \tTorrent Name.torrent'
    expected_sanitized_filename = 'This Is A Torrent Name.torrent'
    assert sanitize_filename(original_filename) == expected_sanitized_filename