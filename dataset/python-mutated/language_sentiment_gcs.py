from google.cloud import language_v1

def sample_analyze_sentiment(gcs_content_uri):
    if False:
        while True:
            i = 10
    '\n    Analyzing Sentiment in text file stored in Cloud Storage\n\n    Args:\n      gcs_content_uri Google Cloud Storage URI where the file content is located.\n      e.g. gs://[Your Bucket]/[Path to File]\n    '
    client = language_v1.LanguageServiceClient()
    type_ = language_v1.Document.Type.PLAIN_TEXT
    language = 'en'
    document = {'gcs_content_uri': gcs_content_uri, 'type_': type_, 'language': language}
    encoding_type = language_v1.EncodingType.UTF8
    response = client.analyze_sentiment(request={'document': document, 'encoding_type': encoding_type})
    print(f'Document sentiment score: {response.document_sentiment.score}')
    print(f'Document sentiment magnitude: {response.document_sentiment.magnitude}')
    for sentence in response.sentences:
        print(f'Sentence text: {sentence.text.content}')
        print(f'Sentence sentiment score: {sentence.sentiment.score}')
        print(f'Sentence sentiment magnitude: {sentence.sentiment.magnitude}')
    print(f'Language of the text: {response.language}')

def main():
    if False:
        while True:
            i = 10
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--gcs_content_uri', type=str, default='gs://cloud-samples-data/language/sentiment-positive.txt')
    args = parser.parse_args()
    sample_analyze_sentiment(args.gcs_content_uri)
if __name__ == '__main__':
    main()