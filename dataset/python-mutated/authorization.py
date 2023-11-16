import warnings
from zope.interface import implementer
from pyramid.interfaces import IAuthorizationPolicy
from pyramid.location import lineage
from pyramid.util import is_nonstr_iter
with warnings.catch_warnings():
    warnings.simplefilter('ignore')
    from pyramid.security import ACLAllowed as _ACLAllowed, ACLDenied as _ACLDenied, Allow, AllPermissionsList as _AllPermissionsList, Authenticated, Deny, Everyone
Everyone = Everyone
Authenticated = Authenticated
Allow = Allow
Deny = Deny

class AllPermissionsList(_AllPermissionsList):
    pass

class ACLAllowed(_ACLAllowed):
    pass

class ACLDenied(_ACLDenied):
    pass
ALL_PERMISSIONS = AllPermissionsList()
DENY_ALL = (Deny, Everyone, ALL_PERMISSIONS)

@implementer(IAuthorizationPolicy)
class ACLAuthorizationPolicy:
    """An :term:`authorization policy` which consults an :term:`ACL`
    object attached to a :term:`context` to determine authorization
    information about a :term:`principal` or multiple principals.
    This class is a wrapper around :class:`.ACLHelper`, refer to that class for
    more detailed documentation.

    Objects of this class implement the
    :class:`pyramid.interfaces.IAuthorizationPolicy` interface.

    .. deprecated:: 2.0

        Authorization policies have been deprecated by the new security system.
        See :ref:`upgrading_auth_20` for more information.

    """

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        self.helper = ACLHelper()

    def permits(self, context, principals, permission):
        if False:
            i = 10
            return i + 15
        'Return an instance of\n        :class:`pyramid.authorization.ACLAllowed` instance if the policy\n        permits access, return an instance of\n        :class:`pyramid.authorization.ACLDenied` if not.'
        return self.helper.permits(context, principals, permission)

    def principals_allowed_by_permission(self, context, permission):
        if False:
            for i in range(10):
                print('nop')
        'Return the set of principals explicitly granted the\n        permission named ``permission`` according to the ACL directly\n        attached to the ``context`` as well as inherited ACLs based on\n        the :term:`lineage`.'
        return self.helper.principals_allowed_by_permission(context, permission)

class ACLHelper:
    """A helper for use with constructing a :term:`security policy` which
    consults an :term:`ACL` object attached to a :term:`context` to determine
    authorization information about a :term:`principal` or multiple principals.
    If the context is part of a :term:`lineage`, the context's parents are
    consulted for ACL information too.

    """

    def permits(self, context, principals, permission):
        if False:
            i = 10
            return i + 15
        "Return an instance of :class:`pyramid.authorization.ACLAllowed` if\n        the ACL allows access a user with the given principals, return an\n        instance of :class:`pyramid.authorization.ACLDenied` if not.\n\n        When checking if principals are allowed, the security policy consults\n        the ``context`` for an ACL first.  If no ACL exists on the context, or\n        one does exist but the ACL does not explicitly allow or deny access for\n        any of the effective principals, consult the context's parent ACL, and\n        so on, until the lineage is exhausted or we determine that the policy\n        permits or denies.\n\n        During this processing, if any :data:`pyramid.authorization.Deny`\n        ACE is found matching any principal in ``principals``, stop\n        processing by returning an\n        :class:`pyramid.authorization.ACLDenied` instance (equals\n        ``False``) immediately.  If any\n        :data:`pyramid.authorization.Allow` ACE is found matching any\n        principal, stop processing by returning an\n        :class:`pyramid.authorization.ACLAllowed` instance (equals\n        ``True``) immediately.  If we exhaust the context's\n        :term:`lineage`, and no ACE has explicitly permitted or denied\n        access, return an instance of\n        :class:`pyramid.authorization.ACLDenied` (equals ``False``).\n\n        "
        acl = '<No ACL found on any object in resource lineage>'
        for location in lineage(context):
            try:
                acl = location.__acl__
            except AttributeError:
                continue
            if acl and callable(acl):
                acl = acl()
            for ace in acl:
                (ace_action, ace_principal, ace_permissions) = ace
                if ace_principal in principals:
                    if not is_nonstr_iter(ace_permissions):
                        ace_permissions = [ace_permissions]
                    if permission in ace_permissions:
                        if ace_action == Allow:
                            return ACLAllowed(ace, acl, permission, principals, location)
                        else:
                            return ACLDenied(ace, acl, permission, principals, location)
        return ACLDenied('<default deny>', acl, permission, principals, context)

    def principals_allowed_by_permission(self, context, permission):
        if False:
            while True:
                i = 10
        "Return the set of principals explicitly granted the permission\n        named ``permission`` according to the ACL directly attached to the\n        ``context`` as well as inherited ACLs based on the :term:`lineage`.\n\n        When computing principals allowed by a permission, we compute the set\n        of principals that are explicitly granted the ``permission`` in the\n        provided ``context``.  We do this by walking 'up' the object graph\n        *from the root* to the context.  During this walking process, if we\n        find an explicit :data:`pyramid.authorization.Allow` ACE for a\n        principal that matches the ``permission``, the principal is included in\n        the allow list.  However, if later in the walking process that\n        principal is mentioned in any :data:`pyramid.authorization.Deny` ACE\n        for the permission, the principal is removed from the allow list.  If\n        a :data:`pyramid.authorization.Deny` to the principal\n        :data:`pyramid.authorization.Everyone` is encountered during the\n        walking process that matches the ``permission``, the allow list is\n        cleared for all principals encountered in previous ACLs.  The walking\n        process ends after we've processed the any ACL directly attached to\n        ``context``; a set of principals is returned.\n\n        "
        allowed = set()
        for location in reversed(list(lineage(context))):
            try:
                acl = location.__acl__
            except AttributeError:
                continue
            allowed_here = set()
            denied_here = set()
            if acl and callable(acl):
                acl = acl()
            for (ace_action, ace_principal, ace_permissions) in acl:
                if not is_nonstr_iter(ace_permissions):
                    ace_permissions = [ace_permissions]
                if ace_action == Allow and permission in ace_permissions:
                    if ace_principal not in denied_here:
                        allowed_here.add(ace_principal)
                if ace_action == Deny and permission in ace_permissions:
                    denied_here.add(ace_principal)
                    if ace_principal == Everyone:
                        allowed = set()
                        break
                    elif ace_principal in allowed:
                        allowed.remove(ace_principal)
            allowed.update(allowed_here)
        return allowed