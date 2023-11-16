from ......thumbnail.models import Thumbnail
from .....tests.utils import assert_no_permission, get_graphql_content
USER_AVATAR_DELETE_MUTATION = '\n    mutation userAvatarDelete {\n        userAvatarDelete {\n            user {\n                avatar {\n                    url\n                }\n            }\n        }\n    }\n'

def test_user_avatar_delete_mutation_permission(api_client):
    if False:
        i = 10
        return i + 15
    'Should raise error if user is not staff.'
    query = USER_AVATAR_DELETE_MUTATION
    response = api_client.post_graphql(query)
    assert_no_permission(response)

def test_user_avatar_delete_mutation(staff_api_client):
    if False:
        for i in range(10):
            print('nop')
    query = USER_AVATAR_DELETE_MUTATION
    user = staff_api_client.user
    Thumbnail.objects.create(user=staff_api_client.user, size=128)
    assert user.thumbnails.all()
    response = staff_api_client.post_graphql(query)
    content = get_graphql_content(response)
    user.refresh_from_db()
    assert not user.avatar
    assert not content['data']['userAvatarDelete']['user']['avatar']
    assert not user.thumbnails.exists()