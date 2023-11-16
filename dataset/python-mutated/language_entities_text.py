from google.cloud import language_v1

def sample_analyze_entities(text_content):
    if False:
        i = 10
        return i + 15
    '\n    Analyzing Entities in a String\n\n    Args:\n      text_content The text content to analyze\n    '
    client = language_v1.LanguageServiceClient()
    type_ = language_v1.Document.Type.PLAIN_TEXT
    language = 'en'
    document = {'content': text_content, 'type_': type_, 'language': language}
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
        while True:
            i = 10
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--text_content', type=str, default='California is a state.')
    args = parser.parse_args()
    sample_analyze_entities(args.text_content)
if __name__ == '__main__':
    main()