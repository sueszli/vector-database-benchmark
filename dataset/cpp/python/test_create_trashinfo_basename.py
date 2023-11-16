from trashcli.put.janitor_tools.info_file_persister import \
    create_trashinfo_basename


class TestCreateTrashinfoBasename:
    def test_when_file_name_is_not_too_long(self):
        assert 'basename_1.trashinfo' == create_trashinfo_basename('basename',
                                                                   '_1',
                                                                   False)

    def test_when_file_name_too_long(self):
        assert '12345678_1.trashinfo' == create_trashinfo_basename(
            '12345678901234567890', '_1', True)

    def test_when_file_name_too_long_with_big_suffix(self):
        assert '12345_9999.trashinfo' == create_trashinfo_basename(
            '12345678901234567890', '_9999', True)
