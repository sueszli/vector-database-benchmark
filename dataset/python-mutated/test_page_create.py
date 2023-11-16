from datetime import datetime, timedelta
import graphene
import pytz
from django.utils.text import slugify
from .....page.error_codes import PageErrorCode
from .....tests.utils import dummy_editorjs
from ....tests.utils import get_graphql_content
CREATE_PAGE_MUTATION = '\n    mutation CreatePage(\n        $input: PageCreateInput!\n    ) {\n        pageCreate(input: $input) {\n            page {\n                id\n                title\n                content\n                slug\n                isPublished\n                publicationDate\n                pageType {\n                    id\n                }\n                attributes {\n                    attribute {\n                        slug\n                    }\n                    values {\n                        slug\n                        name\n                        reference\n                        date\n                        dateTime\n                        file {\n                            url\n                            contentType\n                        }\n                    }\n                }\n            }\n            errors {\n                field\n                code\n                message\n                attributes\n            }\n        }\n    }\n'

def test_page_create_mutation_with_publication_date(staff_api_client, permission_manage_pages, page_type):
    if False:
        for i in range(10):
            print('nop')
    page_slug = 'test-slug'
    page_content = dummy_editorjs('test content', True)
    page_title = 'test title'
    page_is_published = True
    publication_date = datetime.now(pytz.utc) + timedelta(days=5)
    page_type_id = graphene.Node.to_global_id('PageType', page_type.pk)
    tag_attr = page_type.page_attributes.get(name='tag')
    tag_value_slug = tag_attr.values.first().slug
    tag_value_name = tag_attr.values.first().name
    tag_attr_id = graphene.Node.to_global_id('Attribute', tag_attr.id)
    size_attr = page_type.page_attributes.get(name='Page size')
    size_attr_id = graphene.Node.to_global_id('Attribute', size_attr.id)
    non_existent_attr_value = 'New value'
    variables = {'input': {'title': page_title, 'content': page_content, 'isPublished': page_is_published, 'publicationDate': publication_date, 'slug': page_slug, 'pageType': page_type_id, 'attributes': [{'id': tag_attr_id, 'values': [tag_value_name]}, {'id': size_attr_id, 'values': [non_existent_attr_value]}]}}
    response = staff_api_client.post_graphql(CREATE_PAGE_MUTATION, variables, permissions=[permission_manage_pages])
    content = get_graphql_content(response)
    data = content['data']['pageCreate']
    assert data['errors'] == []
    assert data['page']['title'] == page_title
    assert data['page']['content'] == page_content
    assert data['page']['slug'] == page_slug
    assert data['page']['isPublished'] == page_is_published
    assert data['page']['publicationDate'] == publication_date.date().isoformat()
    assert data['page']['pageType']['id'] == page_type_id
    values = (data['page']['attributes'][0]['values'][0]['slug'], data['page']['attributes'][1]['values'][0]['slug'])
    assert slugify(non_existent_attr_value) in values
    assert tag_value_slug in values

def test_page_create_mutation_publication_date_and_published_at_provided(staff_api_client, permission_manage_pages, page_type):
    if False:
        print('Hello World!')
    'Test that setting publication date and time are mutually exclusive.'
    page_slug = 'test-slug'
    page_content = dummy_editorjs('test content', True)
    page_title = 'test title'
    page_is_published = True
    publication_date = datetime.now(pytz.utc) + timedelta(days=5)
    published_at = datetime.now(pytz.utc) + timedelta(days=5)
    page_type_id = graphene.Node.to_global_id('PageType', page_type.pk)
    variables = {'input': {'title': page_title, 'content': page_content, 'isPublished': page_is_published, 'publishedAt': published_at, 'publicationDate': publication_date, 'slug': page_slug, 'pageType': page_type_id}}
    response = staff_api_client.post_graphql(CREATE_PAGE_MUTATION, variables, permissions=[permission_manage_pages])
    content = get_graphql_content(response)
    data = content['data']['pageCreate']
    assert not data['page']
    assert len(data['errors']) == 1
    assert data['errors'][0]['field'] == 'publicationDate'
    assert data['errors'][0]['code'] == PageErrorCode.INVALID.name