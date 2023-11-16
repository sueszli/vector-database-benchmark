from source_sftp_bulk.client import SFTPClient

def test_get_files_matching_pattern_match():
    if False:
        print('Hello World!')
    files = [{'filepath': 'test.csv', 'last_modified': '2021-01-01 00:00:00'}, {'filepath': 'test2.csv', 'last_modified': '2021-01-01 00:00:00'}]
    result = SFTPClient.get_files_matching_pattern(files, 'test.csv')
    assert result == [{'filepath': 'test.csv', 'last_modified': '2021-01-01 00:00:00'}]

def test_get_files_matching_pattern_no_match():
    if False:
        return 10
    files = [{'filepath': 'test.csv', 'last_modified': '2021-01-01 00:00:00'}, {'filepath': 'test2.csv', 'last_modified': '2021-01-01 00:00:00'}]
    result = SFTPClient.get_files_matching_pattern(files, 'test3.csv')
    assert result == []

def test_get_files_matching_pattern_regex_match():
    if False:
        while True:
            i = 10
    files = [{'filepath': 'test.csv', 'last_modified': '2021-01-01 00:00:00'}, {'filepath': 'test2.csv', 'last_modified': '2021-01-01 00:00:00'}]
    result = SFTPClient.get_files_matching_pattern(files, 'test.*')
    assert result == [{'filepath': 'test.csv', 'last_modified': '2021-01-01 00:00:00'}, {'filepath': 'test2.csv', 'last_modified': '2021-01-01 00:00:00'}]