import frappe
from frappe import _
from frappe.model.document import Document

class NetworkPrinterSettings(Document):
    from typing import TYPE_CHECKING
    if TYPE_CHECKING:
        from frappe.types import DF
        port: DF.Int
        printer_name: DF.Literal
        server_ip: DF.Data

    @frappe.whitelist()
    def get_printers_list(self, ip='127.0.0.1', port=631):
        if False:
            for i in range(10):
                print('nop')
        printer_list = []
        try:
            import cups
        except ImportError:
            frappe.throw(_('This feature can not be used as dependencies are missing.\n\t\t\t\tPlease contact your system manager to enable this by installing pycups!'))
            return
        try:
            cups.setServer(self.server_ip)
            cups.setPort(self.port)
            conn = cups.Connection()
            printers = conn.getPrinters()
            printer_list.extend(({'value': printer_id, 'label': printer['printer-make-and-model']} for (printer_id, printer) in printers.items()))
        except RuntimeError:
            frappe.throw(_('Failed to connect to server'))
        except frappe.ValidationError:
            frappe.throw(_('Failed to connect to server'))
        return printer_list

@frappe.whitelist()
def get_network_printer_settings():
    if False:
        while True:
            i = 10
    return frappe.db.get_list('Network Printer Settings', pluck='name')