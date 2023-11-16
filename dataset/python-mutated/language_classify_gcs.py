from google.cloud import language_v1

def sample_classify_text(gcs_content_uri):
    if False:
        return 10
    '\n    Classifying Content in text file stored in Cloud Storage\n\n    Args:\n      gcs_content_uri Google Cloud Storage URI where the file content is located.\n      e.g. gs://[Your Bucket]/[Path to File]\n      The text file must include at least 20 words.\n    '
    client = language_v1.LanguageServiceClient()
    type_ = language_v1.Document.Type.PLAIN_TEXT
    language = 'en'
    document = {'gcs_content_uri': gcs_content_uri, 'type_': type_, 'language': language}
    response = client.classify_text(request={'document': document})
    for category in response.categories:
        print(f'Category name: {category.name}')
        print(f'Confidence: {category.confidence}')

def main():
    if False:
        return 10
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--gcs_content_uri', type=str, default='gs://cloud-samples-data/language/classify-entertainment.txt')
    args = parser.parse_args()
    sample_classify_text(args.gcs_content_uri)
if __name__ == '__main__':
    main()