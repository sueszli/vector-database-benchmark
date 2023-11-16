from __future__ import annotations
from flask import current_app
from flask_login import AnonymousUserMixin
from airflow.auth.managers.models.base_user import BaseUser

class AnonymousUser(AnonymousUserMixin, BaseUser):
    """User object used when no active user is logged in."""
    _roles: set[tuple[str, str]] = set()
    _perms: set[tuple[str, str]] = set()

    @property
    def roles(self):
        if False:
            i = 10
            return i + 15
        if not self._roles:
            public_role = current_app.appbuilder.get_app.config['AUTH_ROLE_PUBLIC']
            self._roles = {current_app.appbuilder.sm.find_role(public_role)} if public_role else set()
        return self._roles

    @roles.setter
    def roles(self, roles):
        if False:
            return 10
        self._roles = roles
        self._perms = set()

    @property
    def perms(self):
        if False:
            return 10
        if not self._perms:
            self._perms = {(perm.action.name, perm.resource.name) for role in self.roles for perm in role.permissions}
        return self._perms