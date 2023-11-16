import frappe
from frappe.model.document import Document

class OAuthBearerToken(Document):
    from typing import TYPE_CHECKING
    if TYPE_CHECKING:
        from frappe.types import DF
        access_token: DF.Data | None
        client: DF.Link | None
        expiration_time: DF.Datetime | None
        expires_in: DF.Int
        refresh_token: DF.Data | None
        scopes: DF.Text | None
        status: DF.Literal['Active', 'Revoked']
        user: DF.Link | None

    def validate(self):
        if False:
            print('Hello World!')
        if not self.expiration_time:
            self.expiration_time = frappe.utils.datetime.datetime.strptime(self.creation, '%Y-%m-%d %H:%M:%S.%f') + frappe.utils.datetime.timedelta(seconds=self.expires_in)