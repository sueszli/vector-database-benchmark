import frappe
from frappe.model.document import Document

class PortalSettings(Document):
    from typing import TYPE_CHECKING
    if TYPE_CHECKING:
        from frappe.types import DF
        from frappe.website.doctype.portal_menu_item.portal_menu_item import PortalMenuItem
        custom_menu: DF.Table[PortalMenuItem]
        default_portal_home: DF.Data | None
        default_role: DF.Link | None
        hide_standard_menu: DF.Check
        menu: DF.Table[PortalMenuItem]

    def add_item(self, item):
        if False:
            for i in range(10):
                print('nop')
        'insert new portal menu item if route is not set, or role is different'
        exists = [d for d in self.get('menu', []) if d.get('route') == item.get('route')]
        if exists and item.get('role'):
            if exists[0].role != item.get('role'):
                exists[0].role = item.get('role')
                return True
        elif not exists:
            item['enabled'] = 1
            self.append('menu', item)
            return True

    @frappe.whitelist()
    def reset(self):
        if False:
            for i in range(10):
                print('nop')
        'Restore defaults'
        self.menu = []
        self.sync_menu()

    def sync_menu(self):
        if False:
            while True:
                i = 10
        'Sync portal menu items'
        dirty = False
        for item in frappe.get_hooks('standard_portal_menu_items'):
            if item.get('role') and (not frappe.db.exists('Role', item.get('role'))):
                frappe.get_doc({'doctype': 'Role', 'role_name': item.get('role'), 'desk_access': 0}).insert()
            if self.add_item(item):
                dirty = True
        if dirty:
            self.remove_deleted_doctype_items()
            self.save()

    def on_update(self):
        if False:
            for i in range(10):
                print('nop')
        self.clear_cache()

    def clear_cache(self):
        if False:
            for i in range(10):
                print('nop')
        frappe.clear_cache(user='Guest')
        from frappe.website.utils import clear_cache
        clear_cache()
        frappe.clear_cache()

    def remove_deleted_doctype_items(self):
        if False:
            return 10
        existing_doctypes = set(frappe.get_list('DocType', pluck='name'))
        for menu_item in list(self.get('menu')):
            if menu_item.reference_doctype not in existing_doctypes:
                self.remove(menu_item)