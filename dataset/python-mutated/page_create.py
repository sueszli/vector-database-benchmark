from ...utils import get_graphql_content
PAGE_CREATE_MUTATION = '\nmutation PageCreate($input: PageCreateInput!) {\n  pageCreate(input: $input) {\n    errors {\n      field\n      message\n      code\n    }\n    page {\n      id\n      title\n      slug\n      isPublished\n      publishedAt\n      attributes{\n        attribute{\n          id\n        }\n        values{\n          id\n          name\n          slug\n          value\n          inputType\n          reference\n          file{\n            url\n            contentType\n          }\n          richText\n          plainText\n          boolean\n          date\n          dateTime\n        }\n      }\n    }\n  }\n}\n'

def create_page(staff_api_client, page_type_id, title='test Page', is_published=True, publication_date=None, content=None, attributes=None):
    if False:
        return 10
    variables = {'input': {'pageType': page_type_id, 'title': title, 'content': content, 'isPublished': is_published, 'publicationDate': publication_date, 'attributes': attributes}}
    response = staff_api_client.post_graphql(PAGE_CREATE_MUTATION, variables)
    content = get_graphql_content(response)
    data = content['data']['pageCreate']
    errors = data['errors']
    assert errors == []
    page = data['page']
    assert page['id'] is not None
    return page