from frappe.model.document import Document

class ModuleProfile(Document):
    from typing import TYPE_CHECKING
    if TYPE_CHECKING:
        from frappe.core.doctype.block_module.block_module import BlockModule
        from frappe.types import DF
        block_modules: DF.Table[BlockModule]
        module_profile_name: DF.Data

    def onload(self):
        if False:
            i = 10
            return i + 15
        from frappe.config import get_modules_from_all_apps
        self.set_onload('all_modules', sorted((m.get('module_name') for m in get_modules_from_all_apps())))