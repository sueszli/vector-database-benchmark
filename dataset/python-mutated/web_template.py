import os
from shutil import rmtree
import frappe
from frappe import _
from frappe.model.document import Document
from frappe.modules.export_file import get_module_path, scrub_dt_dn, write_document_file
from frappe.website.utils import clear_cache

class WebTemplate(Document):
    from typing import TYPE_CHECKING
    if TYPE_CHECKING:
        from frappe.types import DF
        from frappe.website.doctype.web_template_field.web_template_field import WebTemplateField
        fields: DF.Table[WebTemplateField]
        module: DF.Link | None
        standard: DF.Check
        template: DF.Code | None
        type: DF.Literal['Component', 'Section', 'Navbar', 'Footer']

    def validate(self):
        if False:
            for i in range(10):
                print('nop')
        if self.standard and (not (frappe.conf.developer_mode or frappe.flags.in_patch)):
            frappe.throw(_('Enable developer mode to create a standard Web Template'))
        for field in self.fields:
            if not field.fieldname:
                field.fieldname = frappe.scrub(field.label)

    def before_save(self):
        if False:
            print('Hello World!')
        if frappe.conf.developer_mode:
            if self.standard:
                self.export_to_files()
            was_standard = (self.get_doc_before_save() or {}).get('standard')
            if was_standard and (not self.standard):
                self.import_from_files()

    def on_update(self):
        if False:
            print('Hello World!')
        'Clear cache for all Web Pages in which this template is used'
        routes = frappe.get_all('Web Page', filters=[['Web Page Block', 'web_template', '=', self.name], ['Web Page', 'published', '=', 1]], pluck='route')
        for route in routes:
            clear_cache(route)

    def on_trash(self):
        if False:
            return 10
        if frappe.conf.developer_mode and self.standard:
            rmtree(self.get_template_folder())

    def export_to_files(self):
        if False:
            return 10
        'Export Web Template to a new folder.\n\n\t\tDoc is exported as JSON. The content of the `template` field gets\n\t\twritten into a separate HTML file. The template should not be contained\n\t\tin the JSON.\n\t\t'
        (html, self.template) = (self.template, '')
        write_document_file(self, create_init=True)
        self.create_template_file(html)

    def import_from_files(self):
        if False:
            return 10
        self.template = self.get_template(standard=True)
        rmtree(self.get_template_folder())

    def create_template_file(self, html=None):
        if False:
            i = 10
            return i + 15
        'Touch a HTML file for the Web Template and add existing content, if any.'
        if self.standard:
            path = self.get_template_path()
            if not os.path.exists(path):
                with open(path, 'w') as template_file:
                    if html:
                        template_file.write(html)

    def get_template_folder(self):
        if False:
            return 10
        "Return the absolute path to the template's folder."
        module = self.module or 'Website'
        module_path = get_module_path(module)
        (doctype, docname) = scrub_dt_dn(self.doctype, self.name)
        return os.path.join(module_path, doctype, docname)

    def get_template_path(self):
        if False:
            i = 10
            return i + 15
        "Return the absolute path to the template's HTML file."
        folder = self.get_template_folder()
        file_name = frappe.scrub(self.name) + '.html'
        return os.path.join(folder, file_name)

    def get_template(self, standard=False):
        if False:
            print('Hello World!')
        'Get the jinja template string.\n\n\t\tParams:\n\t\tstandard - if True, look on the disk instead of in the database.\n\t\t'
        if standard:
            template = self.get_template_path()
            with open(template) as template_file:
                template = template_file.read()
        else:
            template = self.template
        return template

    def render(self, values=None):
        if False:
            while True:
                i = 10
        if not values:
            values = {}
        values = frappe.parse_json(values)
        values.update({'values': values})
        template = self.get_template(self.standard)
        return frappe.render_template(template, values)