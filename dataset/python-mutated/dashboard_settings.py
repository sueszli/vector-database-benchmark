import json
import frappe
from frappe.model.document import Document

class DashboardSettings(Document):
    from typing import TYPE_CHECKING
    if TYPE_CHECKING:
        from frappe.types import DF
        chart_config: DF.Code | None
        user: DF.Link | None
    pass

@frappe.whitelist()
def create_dashboard_settings(user):
    if False:
        while True:
            i = 10
    if not frappe.db.exists('Dashboard Settings', user):
        doc = frappe.new_doc('Dashboard Settings')
        doc.name = user
        doc.insert(ignore_permissions=True)
        frappe.db.commit()
        return doc

def get_permission_query_conditions(user):
    if False:
        return 10
    if not user:
        user = frappe.session.user
    return f'(`tabDashboard Settings`.name = {frappe.db.escape(user)})'

@frappe.whitelist()
def save_chart_config(reset, config, chart_name):
    if False:
        for i in range(10):
            print('nop')
    reset = frappe.parse_json(reset)
    doc = frappe.get_doc('Dashboard Settings', frappe.session.user)
    chart_config = frappe.parse_json(doc.chart_config) or {}
    if reset:
        chart_config[chart_name] = {}
    else:
        config = frappe.parse_json(config)
        if not chart_name in chart_config:
            chart_config[chart_name] = {}
        chart_config[chart_name].update(config)
    frappe.db.set_value('Dashboard Settings', frappe.session.user, 'chart_config', json.dumps(chart_config))