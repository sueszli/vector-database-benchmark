import itertools
import re
from django import template
from wagtail import hooks
from wagtail.users.permission_order import CONTENT_TYPE_ORDER
register = template.Library()

@register.inclusion_tag('wagtailusers/groups/includes/formatted_permissions.html')
def format_permissions(permission_bound_field):
    if False:
        return 10
    "\n    Given a bound field with a queryset of Permission objects - which must be using\n    the CheckboxSelectMultiple widget - construct a list of dictionaries for 'objects':\n\n    'objects': [\n        {\n            'object': name_of_some_content_object,\n            'add': checkbox,\n            'change': checkbox,\n            'delete': checkbox,\n            'publish': checkbox,  # only if the model extends DraftStateMixin\n            'custom': list_of_checkboxes_for_custom_permissions\n        },\n    ]\n\n    and a list of other permissions:\n\n    'others': [\n        (any_non_add_change_delete_permission, checkbox),\n    ]\n\n    (where 'checkbox' is an object with a tag() method that renders the checkbox as HTML;\n    this is a BoundWidget on Django >=1.11)\n\n    - and returns a table template formatted with this list.\n\n    "
    permissions = permission_bound_field.field._queryset
    content_type_ids = sorted(dict.fromkeys(permissions.values_list('content_type_id', flat=True)), key=lambda ct: CONTENT_TYPE_ORDER.get(ct, float('inf')))
    checkboxes_by_id = {int(checkbox.data['value'].value): checkbox for checkbox in permission_bound_field}
    object_perms = []
    other_perms = []
    main_permission_names = ['add', 'change', 'delete', 'publish', 'lock', 'unlock']
    extra_perms_exist = {'publish': False, 'lock': False, 'unlock': False, 'custom': False}
    for content_type_id in content_type_ids:
        content_perms = permissions.filter(content_type_id=content_type_id)
        content_perms_dict = {}
        custom_perms = []
        if content_perms[0].content_type.name == 'admin':
            perm = content_perms[0]
            other_perms.append((perm, checkboxes_by_id[perm.id]))
            continue
        for perm in content_perms:
            content_perms_dict['object'] = perm.content_type.name
            checkbox = checkboxes_by_id[perm.id]
            permission_action = perm.codename.split('_')[0]
            if permission_action in main_permission_names:
                if permission_action in extra_perms_exist:
                    extra_perms_exist[permission_action] = True
                content_perms_dict[permission_action] = {'perm': perm, 'checkbox': checkbox}
            else:
                extra_perms_exist['custom'] = True
                custom_perms.append({'perm': perm, 'name': re.sub(f'{perm.content_type.name}$', '', perm.name, flags=re.I).strip(), 'selected': checkbox.data['selected']})
        content_perms_dict['custom'] = custom_perms
        object_perms.append(content_perms_dict)
    return {'object_perms': object_perms, 'other_perms': other_perms, 'extra_perms_exist': extra_perms_exist}

@register.inclusion_tag('wagtailadmin/shared/buttons.html', takes_context=True)
def user_listing_buttons(context, user):
    if False:
        i = 10
        return i + 15
    button_hooks = hooks.get_hooks('register_user_listing_buttons')
    buttons = sorted(itertools.chain.from_iterable((hook(context, user) for hook in button_hooks)))
    return {'user': user, 'buttons': buttons}