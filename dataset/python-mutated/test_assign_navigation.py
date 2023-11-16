import graphene
from ....menu.enums import NavigationType
from ....tests.utils import assert_no_permission, get_graphql_content

def test_assign_menu(staff_api_client, menu, permission_manage_menus, permission_manage_settings, site_settings):
    if False:
        print('Hello World!')
    query = '\n    mutation AssignMenu($menu: ID, $navigationType: NavigationType!) {\n        assignNavigation(menu: $menu, navigationType: $navigationType) {\n            errors {\n                field\n                message\n            }\n            menu {\n                name\n            }\n        }\n    }\n    '
    menu_id = graphene.Node.to_global_id('Menu', menu.pk)
    variables = {'menu': menu_id, 'navigationType': NavigationType.MAIN.name}
    response = staff_api_client.post_graphql(query, variables)
    assert_no_permission(response)
    staff_api_client.user.user_permissions.add(permission_manage_menus)
    staff_api_client.user.user_permissions.add(permission_manage_settings)
    response = staff_api_client.post_graphql(query, variables)
    content = get_graphql_content(response)
    assert content['data']['assignNavigation']['menu']['name'] == menu.name
    site_settings.refresh_from_db()
    assert site_settings.top_menu.name == menu.name
    variables = {'menu': menu_id, 'navigationType': NavigationType.SECONDARY.name}
    response = staff_api_client.post_graphql(query, variables)
    content = get_graphql_content(response)
    assert content['data']['assignNavigation']['menu']['name'] == menu.name
    site_settings.refresh_from_db()
    assert site_settings.bottom_menu.name == menu.name
    variables = {'id': None, 'navigationType': NavigationType.MAIN.name}
    response = staff_api_client.post_graphql(query, variables)
    content = get_graphql_content(response)
    assert not content['data']['assignNavigation']['menu']
    site_settings.refresh_from_db()
    assert site_settings.top_menu is None