import frappe
from frappe.core.doctype.report.report import is_prepared_report_enabled
from frappe.model.document import Document
from frappe.permissions import ALL_USER_ROLE
from frappe.utils import cint

class RolePermissionforPageandReport(Document):
    from typing import TYPE_CHECKING
    if TYPE_CHECKING:
        from frappe.core.doctype.has_role.has_role import HasRole
        from frappe.types import DF
        enable_prepared_report: DF.Check
        page: DF.Link | None
        report: DF.Link | None
        roles: DF.Table[HasRole]
        set_role_for: DF.Literal['', 'Page', 'Report']

    @frappe.whitelist()
    def set_report_page_data(self):
        if False:
            for i in range(10):
                print('nop')
        self.set_custom_roles()
        self.check_prepared_report_disabled()

    def set_custom_roles(self):
        if False:
            while True:
                i = 10
        args = self.get_args()
        self.set('roles', [])
        name = frappe.db.get_value('Custom Role', args, 'name')
        if name:
            doc = frappe.get_doc('Custom Role', name)
            roles = doc.roles
        else:
            roles = self.get_standard_roles()
        self.set('roles', roles)

    def check_prepared_report_disabled(self):
        if False:
            i = 10
            return i + 15
        if self.report:
            self.enable_prepared_report = is_prepared_report_enabled(self.report)

    def get_standard_roles(self):
        if False:
            print('Hello World!')
        doctype = self.set_role_for
        docname = self.page if self.set_role_for == 'Page' else self.report
        doc = frappe.get_doc(doctype, docname)
        return doc.roles

    @frappe.whitelist()
    def reset_roles(self):
        if False:
            print('Hello World!')
        roles = self.get_standard_roles()
        self.set('roles', roles)
        self.update_custom_roles()
        self.update_disable_prepared_report()

    @frappe.whitelist()
    def update_report_page_data(self):
        if False:
            i = 10
            return i + 15
        self.update_custom_roles()
        self.update_disable_prepared_report()

    def update_custom_roles(self):
        if False:
            for i in range(10):
                print('nop')
        args = self.get_args()
        name = frappe.db.get_value('Custom Role', args, 'name')
        args.update({'doctype': 'Custom Role', 'roles': self.get_roles()})
        if self.report:
            args.update({'ref_doctype': frappe.db.get_value('Report', self.report, 'ref_doctype')})
        if name:
            custom_role = frappe.get_doc('Custom Role', name)
            custom_role.set('roles', self.get_roles())
            custom_role.save()
        else:
            frappe.get_doc(args).insert()

    def update_disable_prepared_report(self):
        if False:
            while True:
                i = 10
        if self.report:
            frappe.db.sql('update `tabReport` set prepared_report = %s\n\t\t\t\twhere name = %s', (self.enable_prepared_report, self.report))

    def get_args(self, row=None):
        if False:
            return 10
        name = self.page if self.set_role_for == 'Page' else self.report
        check_for_field = self.set_role_for.replace(' ', '_').lower()
        return {check_for_field: name}

    def get_roles(self):
        if False:
            while True:
                i = 10
        return [{'role': data.role, 'parenttype': 'Custom Role'} for data in self.roles if data.role != ALL_USER_ROLE]

    def update_status(self):
        if False:
            while True:
                i = 10
        return frappe.render_template