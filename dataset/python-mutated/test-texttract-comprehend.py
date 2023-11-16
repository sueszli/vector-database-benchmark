from testbook import testbook
import boto3
session = boto3.Session()
creds = session.get_credentials()

@testbook('TextractAndComprehendNotebook.ipynb', execute=[6, 8])
def test_text_detection(tb):
    if False:
        print('Hello World!')
    detect_func = tb.ref('process_text_detection')
    aws_access_code = creds.access_key
    aws_secret_code = creds.secret_key
    region = session.region_name
    bucket = 'DOC-EXAMPLE-BUCKET'
    document = 'Your document name here'
    res = detect_func(bucket, document, aws_access_code, aws_secret_code, region)
    print(res)
    assert len(res)

@testbook('TextractAndComprehendNotebook.ipynb', execute=[6, 11])
def test_entity_detection(tb):
    if False:
        i = 10
        return i + 15
    entity_func = tb.ref('entity_detection')
    aws_access_code = creds.access_key
    aws_secret_code = creds.secret_key
    region = session.region_name
    entity_list = ['San Antonio', 'Dallas is in the state of Texas', 'Boston is in Massachusetts', 'the Metropolitan Museum of Art of is New York City ', "the NASA headquarters is in our nation's capital"]
    res = entity_func(entity_list, aws_access_code, aws_secret_code, region)
    print(res)
    assert len(res)
if __name__ == '__main__':
    test_text_detection()
    test_entity_detection()