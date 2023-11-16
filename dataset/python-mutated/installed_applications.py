import json
import frappe
from frappe import _
from frappe.model.document import Document

class InvalidAppOrder(frappe.ValidationError):
    pass

class InstalledApplications(Document):
    from typing import TYPE_CHECKING
    if TYPE_CHECKING:
        from frappe.core.doctype.installed_application.installed_application import InstalledApplication
        from frappe.types import DF
        installed_applications: DF.Table[InstalledApplication]

    def update_versions(self):
        if False:
            return 10
        self.delete_key('installed_applications')
        for app in frappe.utils.get_installed_apps_info():
            self.append('installed_applications', {'app_name': app.get('app_name'), 'app_version': app.get('version') or 'UNVERSIONED', 'git_branch': app.get('branch') or 'UNVERSIONED'})
        self.save()

@frappe.whitelist()
def update_installed_apps_order(new_order: list[str] | str):
    if False:
        return 10
    "Change the ordering of `installed_apps` global\n\n\tThis list is used to resolve hooks and by default it's order of installation on site.\n\n\tSometimes it might not be the ordering you want, so thie function is provided to override it.\n\t"
    frappe.only_for('System Manager')
    if isinstance(new_order, str):
        new_order = json.loads(new_order)
    frappe.local.request_cache and frappe.local.request_cache.clear()
    existing_order = frappe.get_installed_apps(_ensure_on_bench=True)
    if set(existing_order) != set(new_order) or not isinstance(new_order, list):
        frappe.throw(_('You are only allowed to update order, do not remove or add apps.'), exc=InvalidAppOrder)
    if 'frappe' in new_order:
        new_order.remove('frappe')
    new_order.insert(0, 'frappe')
    frappe.db.set_global('installed_apps', json.dumps(new_order))
    _create_version_log_for_change(existing_order, new_order)

def _create_version_log_for_change(old, new):
    if False:
        i = 10
        return i + 15
    version = frappe.new_doc('Version')
    version.ref_doctype = 'DefaultValue'
    version.docname = 'installed_apps'
    version.data = frappe.as_json({'changed': [['current', json.dumps(old), json.dumps(new)]]})
    version.flags.ignore_links = True
    version.flags.ignore_permissions = True
    version.insert()

@frappe.whitelist()
def get_installed_app_order() -> list[str]:
    if False:
        i = 10
        return i + 15
    frappe.only_for('System Manager')
    return frappe.get_installed_apps(_ensure_on_bench=True)