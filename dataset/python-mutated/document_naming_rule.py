import frappe
from frappe import _
from frappe.model.document import Document
from frappe.model.naming import parse_naming_series
from frappe.utils.data import evaluate_filters

class DocumentNamingRule(Document):
    from typing import TYPE_CHECKING
    if TYPE_CHECKING:
        from frappe.core.doctype.document_naming_rule_condition.document_naming_rule_condition import DocumentNamingRuleCondition
        from frappe.types import DF
        conditions: DF.Table[DocumentNamingRuleCondition]
        counter: DF.Int
        disabled: DF.Check
        document_type: DF.Link
        prefix: DF.Data
        prefix_digits: DF.Int
        priority: DF.Int

    def validate(self):
        if False:
            for i in range(10):
                print('nop')
        self.validate_fields_in_conditions()

    def clear_doctype_map(self):
        if False:
            for i in range(10):
                print('nop')
        frappe.cache_manager.clear_doctype_map(self.doctype, self.document_type)

    def on_update(self):
        if False:
            return 10
        self.clear_doctype_map()

    def on_trash(self):
        if False:
            while True:
                i = 10
        self.clear_doctype_map()

    def validate_fields_in_conditions(self):
        if False:
            print('Hello World!')
        if self.has_value_changed('document_type'):
            docfields = [x.fieldname for x in frappe.get_meta(self.document_type).fields]
            for condition in self.conditions:
                if condition.field not in docfields:
                    frappe.throw(_('{0} is not a field of doctype {1}').format(frappe.bold(condition.field), frappe.bold(self.document_type)))

    def apply(self, doc):
        if False:
            return 10
        '\n\t\tApply naming rules for the given document. Will set `name` if the rule is matched.\n\t\t'
        if self.conditions:
            if not evaluate_filters(doc, [(self.document_type, d.field, d.condition, d.value) for d in self.conditions]):
                return
        counter = frappe.db.get_value(self.doctype, self.name, 'counter', for_update=True) or 0
        naming_series = parse_naming_series(self.prefix, doc=doc)
        doc.name = naming_series + ('%0' + str(self.prefix_digits) + 'd') % (counter + 1)
        frappe.db.set_value(self.doctype, self.name, 'counter', counter + 1)