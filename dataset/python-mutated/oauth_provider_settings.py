import frappe
from frappe import _
from frappe.model.document import Document

class OAuthProviderSettings(Document):
    from typing import TYPE_CHECKING
    if TYPE_CHECKING:
        from frappe.types import DF
        skip_authorization: DF.Literal['Force', 'Auto']
    pass

def get_oauth_settings():
    if False:
        for i in range(10):
            print('nop')
    'Returns oauth settings'
    return frappe._dict({'skip_authorization': frappe.db.get_single_value('OAuth Provider Settings', 'skip_authorization')})