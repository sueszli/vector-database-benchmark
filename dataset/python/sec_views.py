from flask_appbuilder.security.views import UserDBModelView
from flask_babel import lazy_gettext


class MyUserDBModelView(UserDBModelView):
    """
        View that add DB specifics to User view.
        Override to implement your own custom view.
        Then override userdbmodelview property on SecurityManager
    """

    show_fieldsets = [
        (
            lazy_gettext("User info"),
            {"fields": ["username", "active", "roles", "login_count", "emp_number"]},
        ),
        (
            lazy_gettext("Personal Info"),
            {"fields": ["first_name", "last_name", "email"], "expanded": True},
        ),
        (
            lazy_gettext("Audit Info"),
            {
                "fields": [
                    "last_login",
                    "fail_login_count",
                    "created_on",
                    "created_by",
                    "changed_on",
                    "changed_by",
                ],
                "expanded": False,
            },
        ),
    ]

    user_show_fieldsets = [
        (
            lazy_gettext("User info"),
            {"fields": ["username", "active", "roles", "login_count", "emp_number"]},
        ),
        (
            lazy_gettext("Personal Info"),
            {"fields": ["first_name", "last_name", "email"], "expanded": True},
        ),
    ]

    add_columns = [
        "first_name",
        "last_name",
        "username",
        "active",
        "email",
        "roles",
        "companies",
        "emp_number",
        "password",
        "conf_password",
    ]
    list_columns = [
        "first_name",
        "last_name",
        "username",
        "email",
        "active",
        "roles",
        "companies",
    ]
    edit_columns = [
        "first_name",
        "last_name",
        "username",
        "active",
        "email",
        "roles",
        "companies",
        "emp_number",
    ]
