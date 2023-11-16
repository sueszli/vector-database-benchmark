import sys
from google.cloud import language_v1

def sample_analyze_sentiment(content):
    if False:
        i = 10
        return i + 15
    client = language_v1.LanguageServiceClient()
    if isinstance(content, bytes):
        content = content.decode('utf-8')
    type_ = language_v1.Document.Type.PLAIN_TEXT
    document = {'type_': type_, 'content': content}
    response = client.analyze_sentiment(request={'document': document})
    sentiment = response.document_sentiment
    print(f'Score: {sentiment.score}')
    print(f'Magnitude: {sentiment.magnitude}')

def main():
    if False:
        for i in range(10):
            print('nop')
    sample_analyze_sentiment(*sys.argv[1:])
if __name__ == '__main__':
    main()