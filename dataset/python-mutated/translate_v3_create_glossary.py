from google.cloud import translate_v3 as translate

def create_glossary(project_id: str='YOUR_PROJECT_ID', input_uri: str='YOUR_INPUT_URI', glossary_id: str='YOUR_GLOSSARY_ID', timeout: int=180) -> translate.Glossary:
    if False:
        print('Hello World!')
    '\n    Create a equivalent term sets glossary. Glossary can be words or\n    short phrases (usually fewer than five words).\n    https://cloud.google.com/translate/docs/advanced/glossary#format-glossary\n    '
    client = translate.TranslationServiceClient()
    source_lang_code = 'en'
    target_lang_code = 'ja'
    location = 'us-central1'
    name = client.glossary_path(project_id, location, glossary_id)
    language_codes_set = translate.types.Glossary.LanguageCodesSet(language_codes=[source_lang_code, target_lang_code])
    gcs_source = translate.types.GcsSource(input_uri=input_uri)
    input_config = translate.types.GlossaryInputConfig(gcs_source=gcs_source)
    glossary = translate.types.Glossary(name=name, language_codes_set=language_codes_set, input_config=input_config)
    parent = f'projects/{project_id}/locations/{location}'
    operation = client.create_glossary(parent=parent, glossary=glossary)
    result = operation.result(timeout)
    print(f'Created: {result.name}')
    print(f'Input Uri: {result.input_config.gcs_source.input_uri}')
    return result