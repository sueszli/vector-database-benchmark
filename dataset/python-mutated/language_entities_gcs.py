from google.cloud import language_v1

def sample_analyze_entities(gcs_content_uri):
    if False:
        while True:
            i = 10
    '\n    Analyzing Entities in text file stored in Cloud Storage\n\n    Args:\n      gcs_content_uri Google Cloud Storage URI where the file content is located.\n      e.g. gs://[Your Bucket]/[Path to File]\n    '
    client = language_v1.LanguageServiceClient()
    type_ = language_v1.Document.Type.PLAIN_TEXT
    language = 'en'
    document = {'gcs_content_uri': gcs_content_uri, 'type_': type_, 'language': language}
    encoding_type = language_v1.EncodingType.UTF8
    response = client.analyze_entities(request={'document': document, 'encoding_type': encoding_type})
    for entity in response.entities:
        print(f'Representative name for the entity: {entity.name}')
        print(f'Entity type: {language_v1.Entity.Type(entity.type_).name}')
        print(f'Salience score: {entity.salience}')
        for (metadata_name, metadata_value) in entity.metadata.items():
            print(f'{metadata_name}: {metadata_value}')
        for mention in entity.mentions:
            print(f'Mention text: {mention.text.content}')
            print('Mention type: {}'.format(language_v1.EntityMention.Type(mention.type_).name))
    print(f'Language of the text: {response.language}')

def main():
    if False:
        for i in range(10):
            print('nop')
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--gcs_content_uri', type=str, default='gs://cloud-samples-data/language/entity.txt')
    args = parser.parse_args()
    sample_analyze_entities(args.gcs_content_uri)
if __name__ == '__main__':
    main()