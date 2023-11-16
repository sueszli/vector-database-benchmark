import json
import os
import subprocess
import frappe
from frappe.desk.form.load import get_attachments
from frappe.model.document import Document
from frappe.model.sync import get_doc_files
from frappe.modules.import_file import import_doc, import_file_by_path
from frappe.utils import get_files_path

class PackageImport(Document):
    from typing import TYPE_CHECKING
    if TYPE_CHECKING:
        from frappe.types import DF
        activate: DF.Check
        attach_package: DF.Attach | None
        force: DF.Check
        log: DF.Code | None

    def validate(self):
        if False:
            while True:
                i = 10
        if self.activate:
            self.import_package()

    def import_package(self):
        if False:
            print('Hello World!')
        attachment = get_attachments(self.doctype, self.name)
        if not attachment:
            frappe.throw(frappe._('Please attach the package'))
        attachment = attachment[0]
        package_name = attachment.file_name.split('.', 1)[0].rsplit('-', 1)[0]
        if not os.path.exists(frappe.get_site_path('packages')):
            os.makedirs(frappe.get_site_path('packages'))
        subprocess.check_output(['tar', 'xzf', get_files_path(attachment.file_name, is_private=attachment.is_private), '-C', frappe.get_site_path('packages')])
        package_path = frappe.get_site_path('packages', package_name)
        with open(os.path.join(package_path, package_name + '.json')) as packagefile:
            doc_dict = json.loads(packagefile.read())
        frappe.flags.package = import_doc(doc_dict)
        files = []
        log = []
        for module in os.listdir(package_path):
            module_path = os.path.join(package_path, module)
            if os.path.isdir(module_path):
                files = get_doc_files(files, module_path)
        for file in files:
            import_file_by_path(file, force=self.force, ignore_version=True)
            log.append(f'Imported {file}')
        self.log = '\n'.join(log)