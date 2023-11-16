from ...utils import get_graphql_content
PAGE_TYPE_CREATE_MUTATION = '\nmutation PageTypeCreate($input: PageTypeCreateInput!) {\n  pageTypeCreate(input: $input) {\n    errors {\n      field\n      message\n      code\n    }\n    pageType {\n      id\n      name\n      slug\n      attributes{\n        id\n      }\n    }\n  }\n}\n'

def create_page_type(staff_api_client, name='test Page Type', add_attributes=None):
    if False:
        i = 10
        return i + 15
    variables = {'input': {'name': name, 'addAttributes': add_attributes}}
    response = staff_api_client.post_graphql(PAGE_TYPE_CREATE_MUTATION, variables)
    content = get_graphql_content(response)
    data = content['data']['pageTypeCreate']
    errors = data['errors']
    assert errors == []
    page = data['pageType']
    assert page['id'] is not None
    return page