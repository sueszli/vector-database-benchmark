from google.cloud import language_v1

def sample_analyze_entity_sentiment(text_content):
    if False:
        while True:
            i = 10
    '\n    Analyzing Entity Sentiment in a String\n\n    Args:\n      text_content The text content to analyze\n    '
    client = language_v1.LanguageServiceClient()
    type_ = language_v1.types.Document.Type.PLAIN_TEXT
    language = 'en'
    document = {'content': text_content, 'type_': type_, 'language': language}
    encoding_type = language_v1.EncodingType.UTF8
    response = client.analyze_entity_sentiment(request={'document': document, 'encoding_type': encoding_type})
    for entity in response.entities:
        print(f'Representative name for the entity: {entity.name}')
        print(f'Entity type: {language_v1.Entity.Type(entity.type_).name}')
        print(f'Salience score: {entity.salience}')
        sentiment = entity.sentiment
        print(f'Entity sentiment score: {sentiment.score}')
        print(f'Entity sentiment magnitude: {sentiment.magnitude}')
        for (metadata_name, metadata_value) in entity.metadata.items():
            print(f'{metadata_name} = {metadata_value}')
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
    parser.add_argument('--text_content', type=str, default='Grapes are good. Bananas are bad.')
    args = parser.parse_args()
    sample_analyze_entity_sentiment(args.text_content)
if __name__ == '__main__':
    main()