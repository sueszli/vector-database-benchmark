import os
import deidentify_table_infotypes as deid
import pytest
GCLOUD_PROJECT = os.getenv('GOOGLE_CLOUD_PROJECT')

def test_deidentify_table_replace_with_info_types(capsys: pytest.CaptureFixture) -> None:
    if False:
        for i in range(10):
            print('nop')
    table_data = {'header': ['age', 'patient', 'happiness_score', 'factoid'], 'rows': [['101', 'Charles Dickens', '95', 'Charles Dickens name was a curse invented by Shakespeare.'], ['22', 'Jane Austen', '21', "There are 14 kisses in Jane Austen's novels."], ['90', 'Mark Twain', '75', 'Mark Twain loved cats.']]}
    deid.deidentify_table_replace_with_info_types(GCLOUD_PROJECT, table_data, ['PERSON_NAME'], ['patient', 'factoid'])
    (out, _) = capsys.readouterr()
    assert 'string_value: "[PERSON_NAME]"' in out
    assert '[PERSON_NAME] name was a curse invented by [PERSON_NAME].' in out
    assert 'There are 14 kisses in [PERSON_NAME] novels.' in out
    assert '[PERSON_NAME] loved cats.' in out