import json
import frappe
from frappe.model.document import Document
from frappe.utils.jinja import validate_template

class EmailTemplate(Document):
    from typing import TYPE_CHECKING
    if TYPE_CHECKING:
        from frappe.types import DF
        response: DF.TextEditor | None
        response_html: DF.Code | None
        subject: DF.Data
        use_html: DF.Check

    @property
    def response_(self):
        if False:
            print('Hello World!')
        return self.response_html if self.use_html else self.response

    def validate(self):
        if False:
            print('Hello World!')
        validate_template(self.subject)
        validate_template(self.response_)

    def get_formatted_subject(self, doc):
        if False:
            for i in range(10):
                print('nop')
        return frappe.render_template(self.subject, doc)

    def get_formatted_response(self, doc):
        if False:
            while True:
                i = 10
        return frappe.render_template(self.response_, doc)

    def get_formatted_email(self, doc):
        if False:
            print('Hello World!')
        if isinstance(doc, str):
            doc = json.loads(doc)
        return {'subject': self.get_formatted_subject(doc), 'message': self.get_formatted_response(doc)}

@frappe.whitelist()
def get_email_template(template_name, doc):
    if False:
        for i in range(10):
            print('nop')
    'Returns the processed HTML of a email template with the given doc'
    email_template = frappe.get_doc('Email Template', template_name)
    return email_template.get_formatted_email(doc)