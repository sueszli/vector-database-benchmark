from __future__ import annotations
from collections.abc import Sequence
from google.cloud import enterpriseknowledgegraph as ekg

def lookup_public_kg_sample(project_id: str, location: str, ids: Sequence[str], languages: Sequence[str]=None):
    if False:
        return 10
    client = ekg.EnterpriseKnowledgeGraphServiceClient()
    parent = client.common_location_path(project=project_id, location=location)
    request = ekg.LookupPublicKgRequest(parent=parent, ids=ids, languages=languages)
    response = client.lookup_public_kg(request=request)
    print(f'Lookup IDs: {ids}\n')
    print(response)
    for item in response.item_list_element:
        result = item.get('result')
        print(f"Name: {result.get('name')}")
        print(f"- Description: {result.get('description')}")
        print(f"- Types: {result.get('@type')}\n")
        detailed_description = result.get('detailedDescription')
        if detailed_description:
            print('- Detailed Description:')
            print(f"\t- Article Body: {detailed_description.get('articleBody')}")
            print(f"\t- URL: {detailed_description.get('url')}")
            print(f"\t- License: {detailed_description.get('license')}\n")
        print(f"- Cloud MID: {result.get('@id')}")
        for identifier in result.get('identifier'):
            print(f"\t- {identifier.get('name')}: {identifier.get('value')}")
        print('\n')