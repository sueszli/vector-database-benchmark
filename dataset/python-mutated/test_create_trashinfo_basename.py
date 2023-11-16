from trashcli.put.janitor_tools.info_file_persister import create_trashinfo_basename

class TestCreateTrashinfoBasename:

    def test_when_file_name_is_not_too_long(self):
        if False:
            return 10
        assert 'basename_1.trashinfo' == create_trashinfo_basename('basename', '_1', False)

    def test_when_file_name_too_long(self):
        if False:
            i = 10
            return i + 15
        assert '12345678_1.trashinfo' == create_trashinfo_basename('12345678901234567890', '_1', True)

    def test_when_file_name_too_long_with_big_suffix(self):
        if False:
            while True:
                i = 10
        assert '12345_9999.trashinfo' == create_trashinfo_basename('12345678901234567890', '_9999', True)