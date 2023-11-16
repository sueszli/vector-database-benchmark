from google.cloud import language_v1

def sample_analyze_syntax(gcs_content_uri):
    if False:
        i = 10
        return i + 15
    '\n    Analyzing Syntax in text file stored in Cloud Storage\n\n    Args:\n      gcs_content_uri Google Cloud Storage URI where the file content is located.\n      e.g. gs://[Your Bucket]/[Path to File]\n    '
    client = language_v1.LanguageServiceClient()
    type_ = language_v1.Document.Type.PLAIN_TEXT
    language = 'en'
    document = {'gcs_content_uri': gcs_content_uri, 'type_': type_, 'language': language}
    encoding_type = language_v1.EncodingType.UTF8
    response = client.analyze_syntax(request={'document': document, 'encoding_type': encoding_type})
    for token in response.tokens:
        text = token.text
        print(f'Token text: {text.content}')
        print(f'Location of this token in overall document: {text.begin_offset}')
        part_of_speech = token.part_of_speech
        print('Part of Speech tag: {}'.format(language_v1.PartOfSpeech.Tag(part_of_speech.tag).name))
        print('Voice: {}'.format(language_v1.PartOfSpeech.Voice(part_of_speech.voice).name))
        print('Tense: {}'.format(language_v1.PartOfSpeech.Tense(part_of_speech.tense).name))
        print(f'Lemma: {token.lemma}')
        dependency_edge = token.dependency_edge
        print(f'Head token index: {dependency_edge.head_token_index}')
        print('Label: {}'.format(language_v1.DependencyEdge.Label(dependency_edge.label).name))
    print(f'Language of the text: {response.language}')

def main():
    if False:
        i = 10
        return i + 15
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--gcs_content_uri', type=str, default='gs://cloud-samples-data/language/syntax-sentence.txt')
    args = parser.parse_args()
    sample_analyze_syntax(args.gcs_content_uri)
if __name__ == '__main__':
    main()