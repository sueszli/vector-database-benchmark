import json
import frappe
from frappe.model.document import Document
from frappe.utils.safe_exec import read_sql, safe_exec

class SystemConsole(Document):
    from typing import TYPE_CHECKING
    if TYPE_CHECKING:
        from frappe.types import DF
        commit: DF.Check
        console: DF.Code | None
        output: DF.Code | None
        show_processlist: DF.Check
        type: DF.Literal['Python', 'SQL']

    def run(self):
        if False:
            return 10
        frappe.only_for('System Manager')
        try:
            frappe.local.debug_log = []
            if self.type == 'Python':
                safe_exec(self.console)
                self.output = '\n'.join(frappe.debug_log)
            elif self.type == 'SQL':
                self.output = frappe.as_json(read_sql(self.console, as_dict=1))
        except Exception:
            self.commit = False
            self.output = frappe.get_traceback()
        if self.commit:
            frappe.db.commit()
        else:
            frappe.db.rollback()
        frappe.get_doc(dict(doctype='Console Log', script=self.console, type=self.type, committed=self.commit)).insert()
        frappe.db.commit()

@frappe.whitelist()
def execute_code(doc):
    if False:
        print('Hello World!')
    console = frappe.get_doc(json.loads(doc))
    console.run()
    return console.as_dict()

@frappe.whitelist()
def show_processlist():
    if False:
        i = 10
        return i + 15
    frappe.only_for('System Manager')
    return frappe.db.multisql({'postgres': '\n\t\t\tSELECT pid AS "Id",\n\t\t\t\tquery_start AS "Time",\n\t\t\t\tstate AS "State",\n\t\t\t\tquery AS "Info",\n\t\t\t\twait_event AS "Progress"\n\t\t\tFROM pg_stat_activity', 'mariadb': 'show full processlist'}, as_dict=True)