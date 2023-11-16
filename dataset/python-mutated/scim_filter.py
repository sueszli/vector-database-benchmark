from typing import List, Optional, Tuple
from django.http import HttpRequest
from django_scim.filters import UserFilterQuery
from zerver.lib.request import RequestNotes

class ZulipUserFilterQuery(UserFilterQuery):
    """This class implements the filter functionality of SCIM2.
    E.g. requests such as
    /scim/v2/Users?filter=userName eq "hamlet@zulip.com"
    can be made to refer to resources via their properties.
    This gets fairly complicated in its full scope
    (https://datatracker.ietf.org/doc/html/rfc7644#section-3.4.2.2)
    and django-scim2 implements an entire mechanism of converting
    this SCIM2 filter syntax into SQL queries.

    What we have to do in this class is to customize django-scim2 so
    that it knows which SCIM attributes map to which UserProfile
    fields.  We can assume that get_extra_model_filter_kwargs_getter
    has already ensured that we will only interact with non-bot user
    accounts in the realm associated with this SCIM configuration.
    """
    attr_map = {('userName', None, None): 'zerver_userprofile.delivery_email', ('name', 'formatted', None): 'zerver_userprofile.full_name', ('active', None, None): 'zerver_userprofile.is_active'}
    joins = ('INNER JOIN zerver_realm ON zerver_realm.id = realm_id',)

    @classmethod
    def get_extras(cls, q: str, request: Optional[HttpRequest]=None) -> Tuple[str, List[object]]:
        if False:
            i = 10
            return i + 15
        "\n        Return extra SQL and params to be attached to end of current Query's\n        SQL and params. The return format matches the format that should be used\n        for providing raw SQL with params to Django's .raw():\n        https://docs.djangoproject.com/en/3.2/topics/db/sql/#passing-parameters-into-raw\n\n        Here we ensure that results are limited to the subdomain of the request\n        and also exclude bots, as we currently don't want them to be managed by SCIM2.\n        "
        assert request is not None
        realm = RequestNotes.get_notes(request).realm
        assert realm is not None
        return ('AND zerver_realm.id = %s AND zerver_userprofile.is_bot = False ORDER BY zerver_userprofile.id', [realm.id])