from google.cloud import translate_v3 as translate

def delete_glossary(project_id: str='YOUR_PROJECT_ID', glossary_id: str='YOUR_GLOSSARY_ID', timeout: int=180) -> translate.Glossary:
    if False:
        return 10
    'Delete a specific glossary based on the glossary ID.\n\n    Args:\n        project_id: The ID of the GCP project that owns the glossary.\n        glossary_id: The ID of the glossary to delete.\n        timeout: The timeout for this request.\n\n    Returns:\n        The glossary that was deleted.\n    '
    client = translate.TranslationServiceClient()
    name = client.glossary_path(project_id, 'us-central1', glossary_id)
    operation = client.delete_glossary(name=name)
    result = operation.result(timeout)
    print(f'Deleted: {result.name}')
    return result