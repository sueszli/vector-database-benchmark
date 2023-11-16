from google.cloud import language_v1

def sample_classify_text(text_content):
    if False:
        i = 10
        return i + 15
    '\n    Classifying Content in a String\n\n    Args:\n      text_content The text content to analyze.\n    '
    client = language_v1.LanguageServiceClient()
    type_ = language_v1.Document.Type.PLAIN_TEXT
    language = 'en'
    document = {'content': text_content, 'type_': type_, 'language': language}
    content_categories_version = language_v1.ClassificationModelOptions.V2Model.ContentCategoriesVersion.V2
    response = client.classify_text(request={'document': document, 'classification_model_options': {'v2_model': {'content_categories_version': content_categories_version}}})
    for category in response.categories:
        print(f'Category name: {category.name}')
        print(f'Confidence: {category.confidence}')

def main():
    if False:
        while True:
            i = 10
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--text_content', type=str, default='That actor on TV makes movies in Hollywood and also stars in a variety of popular new TV shows.')
    args = parser.parse_args()
    sample_classify_text(args.text_content)
if __name__ == '__main__':
    main()