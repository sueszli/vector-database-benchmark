import frappe
import frappe.share

def execute():
    if False:
        return 10
    for user in frappe.STANDARD_USERS:
        frappe.share.remove('User', user, user)